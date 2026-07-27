"""Tests for pipeline/tools/fetch_page.py's fetch_page() function.

All tests monkeypatch the four tier helpers (_try_jina, _try_obscura,
_try_playwright, _try_archive_is) so nothing touches the network or spawns
external binaries.
"""
import pipeline.tools.fetch_page as fp


def _patch_tiers(monkeypatch, jina=None, obscura=None, playwright=None, archive=None):
    """Monkeypatch the four tier helpers on the fetch_page module.

    Each keyword arg is one of:
      - None: the tier "fails cleanly" -- returns (None, <tier name>)
      - an exception class/instance: the tier raises it (simulates a missing
        binary/package or a network error)
      - a callable(url, timeout=30, debug_dir=None) -> (content, name): the
        tier succeeds with custom content
    Unspecified tiers default to the clean-failure case.
    """
    def _wrap(name, spec):
        if spec is None:
            def fn(url, timeout=30, debug_dir=None):
                return None, name
            return fn
        if isinstance(spec, BaseException) or (
                isinstance(spec, type) and issubclass(spec, BaseException)):
            def fn(url, timeout=30, debug_dir=None):
                raise spec
            return fn
        return spec

    monkeypatch.setattr(fp, "_try_jina", _wrap("jina.ai", jina))
    monkeypatch.setattr(fp, "_try_obscura", _wrap("obscura", obscura))
    monkeypatch.setattr(fp, "_try_playwright", _wrap("playwright", playwright))
    monkeypatch.setattr(fp, "_try_archive_is", _wrap("archive.is", archive))


def test_all_tiers_fail_writes_marker(tmp_path, monkeypatch):
    _patch_tiers(monkeypatch)
    result = fp.fetch_page("http://example.com/a", "target-1", cache_dir=str(tmp_path))
    assert result["ok"] is False
    assert result["method"] is None
    page = tmp_path / "target-1" / "page.md"
    assert page.exists()
    assert "(failed to fetch http://example.com/a)" in page.read_text()
    assert "(failed to fetch http://example.com/a)" in result["content"]


def test_raising_tier_falls_through_and_writes_marker(tmp_path, monkeypatch):
    """Regression test for the bug where an exception mid-tier (e.g. a missing
    optional package) escaped fetch_page() entirely, skipping remaining tiers
    and never writing page.md. This must fail without the _safe_tier fix."""
    _patch_tiers(monkeypatch, obscura=ModuleNotFoundError("no module named 'nonexistent'"))
    result = fp.fetch_page("http://example.com/b", "target-2", cache_dir=str(tmp_path))
    assert result["ok"] is False
    assert result["method"] is None
    page = tmp_path / "target-2" / "page.md"
    assert page.exists()
    assert "(failed to fetch http://example.com/b)" in page.read_text()


def test_raising_tier_then_succeeding_tier_wins(tmp_path, monkeypatch):
    def _playwright_ok(url, timeout=30, debug_dir=None):
        return "playwright content http://example.com/c", "playwright"

    _patch_tiers(monkeypatch, obscura=ModuleNotFoundError("boom"), playwright=_playwright_ok)
    result = fp.fetch_page("http://example.com/c", "target-3", cache_dir=str(tmp_path))
    assert result["ok"] is True
    assert result["method"] == "playwright"
    assert "playwright content" in result["content"]


def test_jina_success_verbatim_no_prefix(tmp_path, monkeypatch):
    def _jina_ok(url, timeout=30, debug_dir=None):
        return "# Title\n\nJina content here.", "jina.ai"

    _patch_tiers(monkeypatch, jina=_jina_ok)
    result = fp.fetch_page("http://example.com/d", "target-4", cache_dir=str(tmp_path))
    assert result["ok"] is True
    assert result["method"] == "jina.ai"
    assert result["content"] == "# Title\n\nJina content here."
    assert "**Source URL:**" not in result["content"]
    page = tmp_path / "target-4" / "page.md"
    assert page.read_text() == result["content"]


def test_non_jina_success_without_url_gets_prefix(tmp_path, monkeypatch):
    def _obscura_ok(url, timeout=30, debug_dir=None):
        return "Some page body with no link in the leading text.", "obscura"

    _patch_tiers(monkeypatch, obscura=_obscura_ok)
    result = fp.fetch_page("http://example.com/e", "target-5", cache_dir=str(tmp_path))
    assert result["ok"] is True
    assert result["content"].startswith("**Source URL:** http://example.com/e")
    assert "Some page body" in result["content"]


def test_paywall_signal_sets_warning_but_keeps_full_content(tmp_path, monkeypatch):
    body = "Subscribe to continue reading this article. " + ("more text. " * 50)

    def _obscura_paywall(url, timeout=30, debug_dir=None):
        return body, "obscura"

    _patch_tiers(monkeypatch, obscura=_obscura_paywall)
    result = fp.fetch_page("http://example.com/f", "target-6", cache_dir=str(tmp_path))
    assert result["ok"] is True
    assert result["warning"] is not None
    assert "paywall" in result["warning"].lower()
    # Full body must still be present -- the paywall check only warns, never truncates.
    assert body in result["content"]
