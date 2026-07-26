"""Tests for pipeline/stage_d_sources.py."""
import html as html_mod


def test_mark_excerpts_no_spans_returns_escaped_text():
    from pipeline.stage_d_sources import mark_excerpts

    text = "Plain text with <a tag> in it & an ampersand."
    result = mark_excerpts(text, [])
    assert result == html_mod.escape(text, quote=False)
    assert "<mark" not in result


def test_mark_excerpts_single_span():
    from pipeline.stage_d_sources import mark_excerpts

    text = "The facility injects 20000 barrels a day into the well."
    start = text.index("injects 20000 barrels a day")
    end = start + len("injects 20000 barrels a day")
    result = mark_excerpts(text, [(start, end, "fn3", "WARNING")])

    assert '<mark id="exc-fn3" data-findings="fn3" data-severity="WARNING">' in result
    assert "injects 20000 barrels a day" in result
    assert result.startswith("The facility ")
    assert result.count("<mark") == 1


def test_mark_excerpts_multiple_non_overlapping_spans():
    from pipeline.stage_d_sources import mark_excerpts

    text = "First claim here. Middle text. Second claim there."
    s1 = text.index("First claim here")
    e1 = s1 + len("First claim here")
    s2 = text.index("Second claim there")
    e2 = s2 + len("Second claim there")
    result = mark_excerpts(text, [
        (s1, e1, "fn1", "PASS"),
        (s2, e2, "fn2", "CRITICAL"),
    ])

    assert 'id="exc-fn1" data-findings="fn1" data-severity="PASS"' in result
    assert 'id="exc-fn2" data-findings="fn2" data-severity="CRITICAL"' in result
    assert result.count("<mark") == 2
    assert "Middle text." in result
    # Both marks' content still present and in original order
    assert result.index("First claim here") < result.index("Second claim there")


def test_mark_excerpts_overlapping_spans_split_into_three_segments():
    from pipeline.stage_d_sources import mark_excerpts

    text = "the quick brown fox jumps over the lazy dog"
    # Span A: "quick brown fox" (fn1), Span B: "brown fox jumps" (fn2) -- overlap on "brown fox"
    s1 = text.index("quick brown fox")
    e1 = s1 + len("quick brown fox")
    s2 = text.index("brown fox jumps")
    e2 = s2 + len("brown fox jumps")
    result = mark_excerpts(text, [
        (s1, e1, "fn1", "WARNING"),
        (s2, e2, "fn2", "CRITICAL"),
    ])

    # Three marked segments: "quick " (fn1 only), "brown fox" (both), " jumps" (fn2 only)
    assert result.count("<mark") == 3
    assert 'data-findings="fn1"' in result
    assert 'data-findings="fn2"' in result
    assert 'data-findings="fn1,fn2"' in result
    # Order preserved
    assert result.index('data-findings="fn1"') < result.index('data-findings="fn1,fn2"')
    assert result.index('data-findings="fn1,fn2"') < result.index('data-findings="fn2"')


def test_mark_excerpts_escapes_html_special_chars_in_and_around_marks():
    from pipeline.stage_d_sources import mark_excerpts

    text = "Cost < $5 & rising, per the <report>."
    start = text.index("<report>")
    end = start + len("<report>")
    result = mark_excerpts(text, [(start, end, "fn1", "PASS")])

    assert "Cost &lt; $5 &amp; rising" in result
    assert "&lt;report&gt;" in result
    assert "<report>" not in result  # never appears unescaped


def _finding(finding_id, source_path=None, source_url=None, source_excerpt="a quote",
             severity="PASS"):
    return {
        "finding_id": finding_id, "severity": severity,
        "source_path": source_path, "source_url": source_url,
        "source_excerpt": source_excerpt,
    }


def test_resolve_cited_sources_corpus_path(tmp_path):
    from pipeline.stage_d_sources import resolve_cited_sources

    corpus = tmp_path / "corpus"
    (corpus / "docs").mkdir(parents=True)
    (corpus / "docs" / "a.md").write_text("Some corpus content.")

    findings = [_finding("fn1", source_path="docs/a.md", source_excerpt="corpus content")]
    resolved = resolve_cited_sources(findings, str(corpus), str(tmp_path / "web_cache"))

    assert "docs/a.md" in resolved
    entry = resolved["docs/a.md"]
    assert entry["kind"] == "corpus"
    assert entry["local_path"] == str(corpus / "docs" / "a.md")
    assert entry["excerpts"] == [("corpus content", "fn1", "PASS")]


def test_resolve_cited_sources_web_cache_snapshot(tmp_path):
    from pipeline.stage_d_sources import resolve_cited_sources

    wc = tmp_path / "web_cache" / "fn2"
    wc.mkdir(parents=True)
    (wc / "page.md").write_text("Real fetched page content.")

    findings = [_finding("fn2", source_url="https://example.com/x",
                         source_excerpt="fetched page content", severity="WARNING")]
    resolved = resolve_cited_sources(findings, str(tmp_path / "corpus"), str(tmp_path / "web_cache"))

    assert "fn2" in resolved
    entry = resolved["fn2"]
    assert entry["kind"] == "web"
    assert entry["local_path"] == str(wc / "page.md")
    assert entry["excerpts"] == [("fetched page content", "fn2", "WARNING")]


def test_resolve_cited_sources_failed_fetch_placeholder_not_resolved(tmp_path):
    from pipeline.stage_d_sources import resolve_cited_sources

    wc = tmp_path / "web_cache" / "fn3"
    wc.mkdir(parents=True)
    (wc / "page.md").write_text("(failed to fetch https://example.com/dead)\n")

    findings = [_finding("fn3", source_url="https://example.com/dead")]
    resolved = resolve_cited_sources(findings, str(tmp_path / "corpus"), str(tmp_path / "web_cache"))

    assert "fn3" not in resolved


def test_resolve_cited_sources_multiple_findings_same_document(tmp_path):
    from pipeline.stage_d_sources import resolve_cited_sources

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("First bit. Second bit.")

    findings = [
        _finding("fn1", source_path="a.md", source_excerpt="First bit", severity="PASS"),
        _finding("fn2", source_path="a.md", source_excerpt="Second bit", severity="CRITICAL"),
    ]
    resolved = resolve_cited_sources(findings, str(corpus), str(tmp_path / "web_cache"))

    assert len(resolved) == 1
    assert resolved["a.md"]["excerpts"] == [
        ("First bit", "fn1", "PASS"),
        ("Second bit", "fn2", "CRITICAL"),
    ]


def test_resolve_cited_sources_skips_findings_with_no_source(tmp_path):
    from pipeline.stage_d_sources import resolve_cited_sources

    findings = [_finding("fn1", source_path=None, source_url=None)]
    resolved = resolve_cited_sources(findings, str(tmp_path / "corpus"), str(tmp_path / "web_cache"))

    assert resolved == {}


def test_resolve_cited_sources_missing_corpus_file_not_resolved(tmp_path):
    from pipeline.stage_d_sources import resolve_cited_sources

    findings = [_finding("fn1", source_path="does/not/exist.md")]
    resolved = resolve_cited_sources(findings, str(tmp_path / "corpus"), str(tmp_path / "web_cache"))

    assert resolved == {}


def test_render_source_document_highlights_found_excerpt():
    from pipeline.stage_d_sources import render_source_document

    text = "First paragraph here.\n\nThe facility injects 20000 barrels a day."
    html, not_found = render_source_document(text, [("injects 20000 barrels a day", "fn1", "WARNING")])

    assert not_found == []
    assert '<mark id="exc-fn1"' in html
    assert "<p>First paragraph here.</p>" in html


def test_render_source_document_reports_not_found_excerpt_without_failing():
    from pipeline.stage_d_sources import render_source_document

    text = "This document says nothing relevant."
    html, not_found = render_source_document(text, [("a phrase that is not in the text", "fn2", "CRITICAL")])

    assert not_found == ["fn2"]
    assert "This document says nothing relevant." in html
    assert "<mark" not in html


def test_render_source_document_multiple_excerpts_some_found_some_not():
    from pipeline.stage_d_sources import render_source_document

    text = "The vote passed six to one. Nothing else notable happened."
    html, not_found = render_source_document(text, [
        ("vote passed six to one", "fn1", "PASS"),
        ("a completely absent phrase", "fn2", "CRITICAL"),
    ])

    assert not_found == ["fn2"]
    assert '<mark id="exc-fn1"' in html


def test_backstop_fetch_missing_skips_already_resolved(tmp_path, monkeypatch):
    from pipeline.stage_d_sources import backstop_fetch_missing

    wc = tmp_path / "web_cache" / "fn1"
    wc.mkdir(parents=True)
    (wc / "page.md").write_text("Already have this one.")

    calls = []
    monkeypatch.setattr(
        "pipeline.stage_d_sources._run_fetch_page",
        lambda url, target_id, cache_dir: calls.append((url, target_id)),
    )

    findings = [_finding("fn1", source_url="https://example.com/a")]
    still_missing = backstop_fetch_missing(findings, str(tmp_path / "corpus"), str(tmp_path / "web_cache"))

    assert calls == []  # never attempted -- already resolved
    assert still_missing == []


def test_backstop_fetch_missing_attempts_fetch_and_succeeds(tmp_path, monkeypatch):
    from pipeline.stage_d_sources import backstop_fetch_missing

    def fake_fetch(url, target_id, cache_dir):
        d = __import__("pathlib").Path(cache_dir) / target_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "page.md").write_text("Freshly fetched content.")

    monkeypatch.setattr("pipeline.stage_d_sources._run_fetch_page", fake_fetch)

    findings = [_finding("fn2", source_url="https://example.com/b")]
    still_missing = backstop_fetch_missing(findings, str(tmp_path / "corpus"), str(tmp_path / "web_cache"))

    assert still_missing == []
    assert (tmp_path / "web_cache" / "fn2" / "page.md").read_text() == "Freshly fetched content."


def test_backstop_fetch_missing_reports_still_failing(tmp_path, monkeypatch):
    from pipeline.stage_d_sources import backstop_fetch_missing

    def fake_fetch(url, target_id, cache_dir):
        d = __import__("pathlib").Path(cache_dir) / target_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "page.md").write_text(f"(failed to fetch {url})\n")

    monkeypatch.setattr("pipeline.stage_d_sources._run_fetch_page", fake_fetch)

    findings = [_finding("fn3", source_url="https://example.com/dead")]
    still_missing = backstop_fetch_missing(findings, str(tmp_path / "corpus"), str(tmp_path / "web_cache"))

    assert still_missing == [("fn3", "https://example.com/dead")]


def test_write_missing_snapshots_report_creates_and_removes(tmp_path):
    from pipeline.stage_d_sources import write_missing_snapshots_report

    out = tmp_path / "output"
    out.mkdir()
    report = out / "MISSING_SNAPSHOTS.md"

    write_missing_snapshots_report(str(out), [("fn3", "https://example.com/dead")])
    assert report.exists()
    assert "fn3" in report.read_text()
    assert "https://example.com/dead" in report.read_text()

    write_missing_snapshots_report(str(out), [])
    assert not report.exists()
