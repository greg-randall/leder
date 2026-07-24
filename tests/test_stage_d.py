"""Tests for Stage D HTML output with severity colors."""
from pathlib import Path
from types import SimpleNamespace


def _make_claim(check_type="fact_check", severity="PASS", claim_text="Test claim text.",
                 rationale="Found in source.", source_path="test/doc.md"):
    return SimpleNamespace(
        claim_text=claim_text, rationale=rationale, check_type=check_type,
        severity=severity, source_proximity=None, source_path=source_path,
        source_url=None, source_excerpt=None, human_review=False, reconciled=False,
    )


def test_severity_colors_in_html(tmp_path):
    """Generated HTML must use severity classes derived from REAL stage_c
    output, not just any text containing the word 'pass' or a leaked CSS
    variable name (the bug this test replaces: both old fixtures used a
    delimiter stage_c never actually produces, so the real parsing path
    never ran and the assertions passed on accident)."""
    from pipeline.stage_c_rebuild import build_footnote_block
    from pipeline.stage_d_html import run_stage_d

    footnote_block = build_footnote_block([_make_claim(severity="PASS")])
    md = tmp_path / "sourced.md"
    md.write_text("# Test Article\n\nSome text here.[^1]\n" + footnote_block)

    html = Path(run_stage_d(str(md), str(tmp_path / "out.html"))).read_text()

    assert 'class="fn-ref pass"' in html
    assert 'class="source pass"' in html
    assert 'data-severity="PASS"' in html
    assert "--supported" not in html


def test_critical_severity_in_html(tmp_path):
    """CRITICAL severity gets the critical class end to end, not just red
    CSS present anywhere on the page."""
    from pipeline.stage_c_rebuild import build_footnote_block
    from pipeline.stage_d_html import run_stage_d

    footnote_block = build_footnote_block([_make_claim(severity="CRITICAL")])
    md = tmp_path / "sourced.md"
    md.write_text("# Test Article\n\nSome text.[^1]\n" + footnote_block)

    html = Path(run_stage_d(str(md), str(tmp_path / "out.html"))).read_text()

    assert 'class="fn-ref critical"' in html
    assert 'class="source critical"' in html
    assert 'data-severity="CRITICAL"' in html


def test_warning_severity_in_html(tmp_path):
    """WARNING severity (the default/fallback badge) gets the warning class."""
    from pipeline.stage_c_rebuild import build_footnote_block
    from pipeline.stage_d_html import run_stage_d

    footnote_block = build_footnote_block([_make_claim(severity="WARNING")])
    md = tmp_path / "sourced.md"
    md.write_text("# Test Article\n\nSome text.[^1]\n" + footnote_block)

    html = Path(run_stage_d(str(md), str(tmp_path / "out.html"))).read_text()

    assert 'class="fn-ref warning"' in html
    assert 'class="source warning"' in html
    assert 'data-severity="WARNING"' in html
