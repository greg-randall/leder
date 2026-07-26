"""Tests for Stage D HTML output with severity colors."""
from types import SimpleNamespace


def _make_claim(check_type="fact_check", severity="PASS", claim_text="Test claim text.",
                rationale="Found in source.", source_path="test/doc.md"):
    return SimpleNamespace(
        claim_text=claim_text, rationale=rationale, check_type=check_type,
        severity=severity, source_proximity=None, source_path=source_path,
        source_url=None, source_excerpt=None, human_review=False, reconciled=False,
    )


def _run_and_read(tmp_path, md, findings_json='{"article_file": "a.md", "article_summary": "", "findings": []}'):
    """Helper: write findings.json, run stage_d, return article.html content."""
    from pipeline.stage_d_html import run_stage_d

    findings_path = tmp_path / "findings.json"
    findings_path.write_text(findings_json)
    output_dir = tmp_path / "out"
    run_stage_d(str(md), str(findings_path), str(output_dir),
                corpus_root=str(tmp_path), web_cache_dir=str(tmp_path / "web_cache"))
    return (output_dir / "article.html").read_text()


def test_severity_colors_in_html(tmp_path):
    """Generated HTML must use severity classes derived from REAL stage_c
    output, not just any text containing the word 'pass' or a leaked CSS
    variable name (the bug this test replaces: both old fixtures used a
    delimiter stage_c never actually produces, so the real parsing path
    never ran and the assertions passed on accident)."""
    from pipeline.stage_c_rebuild import build_footnote_block

    footnote_block = build_footnote_block([_make_claim(severity="PASS")])
    md = tmp_path / "sourced.md"
    md.write_text("# Test Article\n\nSome text here.[^1]\n" + footnote_block)

    html = _run_and_read(tmp_path, md)

    assert 'class="fn-ref pass"' in html
    assert 'class="source pass"' in html
    assert 'data-severity="PASS"' in html
    assert "--supported" not in html


def test_critical_severity_in_html(tmp_path):
    """CRITICAL severity gets the critical class end to end, not just red
    CSS present anywhere on the page."""
    from pipeline.stage_c_rebuild import build_footnote_block

    footnote_block = build_footnote_block([_make_claim(severity="CRITICAL")])
    md = tmp_path / "sourced.md"
    md.write_text("# Test Article\n\nSome text.[^1]\n" + footnote_block)

    html = _run_and_read(tmp_path, md)

    assert 'class="fn-ref critical"' in html
    assert 'class="source critical"' in html
    assert 'data-severity="CRITICAL"' in html


def test_warning_severity_in_html(tmp_path):
    """WARNING severity (the default/fallback badge) gets the warning class."""
    from pipeline.stage_c_rebuild import build_footnote_block

    footnote_block = build_footnote_block([_make_claim(severity="WARNING")])
    md = tmp_path / "sourced.md"
    md.write_text("# Test Article\n\nSome text.[^1]\n" + footnote_block)

    html = _run_and_read(tmp_path, md)

    assert 'class="fn-ref warning"' in html
    assert 'class="source warning"' in html
    assert 'data-severity="WARNING"' in html


def test_unplaced_claims_block_is_stripped_and_rendered_separately(tmp_path):
    """Regression test: stage_c_rebuild's unplaced-claims block (heading +
    intro + claim bullets) must be fully removed from the visible article
    body and rendered into its own .unplaced box -- not leak through as
    raw markdown text, and not render as an empty box missing the actual
    claim content. Uses real build_unplaced_warning() output assembled the
    same way run_stage_c actually joins it (unplaced block + '\\n---\\n\\n'
    separator + article), so this can't drift out of sync with the real
    producer the way the original bug did."""
    from pipeline.stage_c_rebuild import build_unplaced_warning
    from pipeline.models import Verdict

    unplaced_claim = SimpleNamespace(
        claim_id="c099", claim_text="An unplaceable claim.", verdict=Verdict.SUPPORTED,
        source_path="orphan.md", source_url=None, rationale="Could not locate in article.",
        source_quote="a quote that does not appear anywhere",
    )
    unplaced_block = build_unplaced_warning([unplaced_claim])

    md = tmp_path / "sourced.md"
    md.write_text(
        unplaced_block + "\n---\n\n"
        "# Test Article\n\nOrdinary article text.\n"
    )

    html = _run_and_read(tmp_path, md)

    # The raw markdown heading/intro must not leak into the page as literal
    # unprocessed text outside the dedicated unplaced-claims box.
    assert "MANUAL REVIEW REQUIRED" not in html.split('<div class="unplaced">')[0]
    assert "# ⚠️ UNPLACED CLAIMS" not in html

    # The dedicated box must actually contain the real claim content, not
    # render empty (the bug: the strip regex only captured the heading
    # line, so _render_unplaced received nothing to work with).
    assert '<div class="unplaced">' in html
    assert "An unplaceable claim." in html
    assert "Could not locate in article." in html

    # The ordinary article body must still render normally afterward.
    assert "Ordinary article text." in html


def test_source_div_gets_data_source_html_when_resolved(tmp_path):
    from pipeline.stage_c_rebuild import build_footnote_block

    (tmp_path / "doc.md").write_text("The facility injects 20000 barrels a day.")

    claim = _make_claim(severity="WARNING", source_path="doc.md")
    footnote_block = build_footnote_block([claim])
    md = tmp_path / "sourced.md"
    md.write_text("# Test Article\n\nSome text here.[^1]\n" + footnote_block)

    findings_json = (
        '{"article_file": "a.md", "article_summary": "", "findings": ['
        '{"finding_id": "1", "check_type": "fact_check", "severity": "WARNING", '
        '"target_text": "t", "anchor_text": "Some text here", "agent_summary": "s", '
        '"source_path": "doc.md", "source_excerpt": "injects 20000 barrels a day"}'
        ']}'
    )
    html = _run_and_read(tmp_path, md, findings_json=findings_json)

    assert 'data-source-html="sources/doc.html"' in html
    assert (tmp_path / "out" / "sources" / "doc.html").exists()


def test_source_div_gets_data_is_summary_for_summary_paths(tmp_path):
    from pipeline.stage_c_rebuild import build_footnote_block

    (tmp_path / "ALL_SUMMARIES.md").write_text("A rollup summary document.")

    claim = _make_claim(severity="WARNING", source_path="ALL_SUMMARIES.md")
    footnote_block = build_footnote_block([claim])
    md = tmp_path / "sourced.md"
    md.write_text("# Test Article\n\nSome text here.[^1]\n" + footnote_block)

    findings_json = (
        '{"article_file": "a.md", "article_summary": "", "findings": ['
        '{"finding_id": "1", "check_type": "fact_check", "severity": "WARNING", '
        '"target_text": "t", "anchor_text": "Some text here", "agent_summary": "s", '
        '"source_path": "ALL_SUMMARIES.md", "source_excerpt": "rollup summary"}'
        ']}'
    )
    html = _run_and_read(tmp_path, md, findings_json=findings_json)

    assert 'data-is-summary="true"' in html


def test_article_html_includes_modal_shell_and_explore_button_script(tmp_path):
    from pipeline.stage_c_rebuild import build_footnote_block

    footnote_block = build_footnote_block([_make_claim(severity="PASS")])
    md = tmp_path / "sourced.md"
    md.write_text("# Test Article\n\nSome text here.[^1]\n" + footnote_block)
    findings_path = tmp_path / "findings.json"
    findings_path.write_text('{"article_file": "a.md", "article_summary": "", "findings": []}')
    output_dir = tmp_path / "out"
    from pipeline.stage_d_html import run_stage_d
    run_stage_d(str(md), str(findings_path), str(output_dir),
                corpus_root=str(tmp_path), web_cache_dir=str(tmp_path / "web_cache"))

    html = (output_dir / "article.html").read_text()
    assert 'id="srcModalOverlay"' in html
    assert "Explore the source material" in html
    assert "function openSourceModal" in html


def test_article_html_includes_keyboard_and_deep_link_script(tmp_path):
    from pipeline.stage_c_rebuild import build_footnote_block
    from pipeline.stage_d_html import run_stage_d

    footnote_block = build_footnote_block([_make_claim(severity="PASS")])
    md = tmp_path / "sourced.md"
    md.write_text("# Test Article\n\nSome text here.[^1]\n" + footnote_block)
    findings_path = tmp_path / "findings.json"
    findings_path.write_text('{"article_file": "a.md", "article_summary": "", "findings": []}')
    output_dir = tmp_path / "out"
    run_stage_d(str(md), str(findings_path), str(output_dir),
                corpus_root=str(tmp_path), web_cache_dir=str(tmp_path / "web_cache"))

    html = (output_dir / "article.html").read_text()
    assert "function navigateModal" in html
    assert "e.key === 'Escape'" in html
    assert "_explorableFindingIds" in html


def test_locate_excerpt_exact_match_only():
    """_locate_excerpt now only does exact substring match."""
    from pipeline.stage_d_sources import _locate_excerpt
    text = "the road commission issued a 68 page notice of violation"
    pos = _locate_excerpt("road commission issued", text)
    assert pos == (4, 26)


def test_locate_excerpt_no_fuzzy_fallback():
    """A candidate that doesn't appear verbatim must return None -- the
    short-doc Levenshtein and long-doc chunk-scoring paths are deleted."""
    from pipeline.stage_d_sources import _locate_excerpt
    text = "the road commission issued a 68 page notice"
    pos = _locate_excerpt("Railroad Commission issued", text)
    assert pos is None


def test_render_source_document_uses_offset_directly(tmp_path):
    """When offset data is present and valid, use it directly -- no search."""
    from pipeline.stage_d_sources import render_source_document
    text = "some text before the road commission issued a notice and text after"
    # 4-tuple: (excerpt, finding_id, severity, offset)
    excerpts = [("this excerpt text is never searched -- offset used instead",
                 "fact_check-1", "PASS", [21, 52])]
    html, not_found = render_source_document(text, excerpts)
    assert not_found == []
    assert "<mark" in html
    assert "road commission issued a notice" in html


def test_render_source_document_falls_back_when_offset_out_of_range(tmp_path):
    """Out-of-range offset falls back to exact-match (which may or may not succeed)."""
    from pipeline.stage_d_sources import render_source_document
    text = "short doc"
    excerpts = [("short doc", "fact_check-1", "PASS", [0, 999])]  # out of range
    html, not_found = render_source_document(text, excerpts)
    # Falls back to exact-match on "short doc" -- which succeeds
    assert not_found == []
    assert "<mark" in html


def test_render_source_document_no_offset_no_match_is_not_found(tmp_path):
    """When no offset and no exact match, the excerpt is not_found."""
    from pipeline.stage_d_sources import render_source_document
    text = "some unrelated text"
    excerpts = [("text not in source", "fact_check-1", "PASS", None)]
    html, not_found = render_source_document(text, excerpts)
    assert not_found == ["fact_check-1"]
