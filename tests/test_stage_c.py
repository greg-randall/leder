"""Tests for Stage C: Article rebuild with source footnotes."""
from pipeline.stage_c_rebuild import (
    normalize_text,
    find_quote_position,
    insert_footnote_markers,
    build_footnote_block,
)
from pipeline.models import Claim

SAMPLE_ARTICLE = """# Test Article

LA-0304 was originally permitted in 2001 to Koch Midstream Services.
The facility irrigates 165 acres in Karnes County using sprinklers.

Another paragraph about water quality."""


def test_normalize_text():
    assert normalize_text("Hello  World") == "hello world"
    assert normalize_text("  LA-0304...  ") == "la-0304"
    assert normalize_text("Line1\nLine2\tTab") == "line1 line2 tab"


def test_find_quote_position_exact():
    pos = find_quote_position("irrigates 165 acres in Karnes County", SAMPLE_ARTICLE)
    assert pos is not None
    assert pos[0] > 0
    assert pos[1] > pos[0]


def test_find_quote_position_normalized():
    # Extra whitespace should still match
    pos = find_quote_position("  irrigates  165  acres  ", SAMPLE_ARTICLE)
    assert pos is not None


def test_find_quote_position_not_found():
    pos = find_quote_position("this text does not exist in the article", SAMPLE_ARTICLE)
    assert pos is None


def test_insert_footnote_markers():
    pos1 = find_quote_position("originally permitted in 2001 to Koch Midstream Services", SAMPLE_ARTICLE)
    pos2 = find_quote_position("irrigates 165 acres in Karnes County using sprinklers", SAMPLE_ARTICLE)
    placed = [("c001", pos1), ("c002", pos2)]

    result = insert_footnote_markers(SAMPLE_ARTICLE, placed)
    assert "[^1]" in result
    assert "[^2]" in result
    assert result.index("[^1]") > SAMPLE_ARTICLE.index("Koch Midstream Services")


def test_build_footnote_block():
    claims = [
        Claim(claim_id="c001", claim_text="A supported claim.", source_quote="q",
              claim_type="attribution", verdict="supported", source_proximity="original",
              source_path="test/doc.md", rationale="Found in the document.",
              human_review=False, reconciled=False),
        Claim(claim_id="c002", claim_text="A contradicted claim.", source_quote="q",
              claim_type="generalization", verdict="contradicted", source_proximity="derived",
              source_path="overview.md", rationale="Data contradicts.",
              human_review=True, reconciled=False),
        Claim(claim_id="c003", claim_text="A reconciled claim.", source_quote="q",
              claim_type="legal", verdict="supported", source_proximity="original",
              source_url="https://example.com", rationale="Found via web.",
              human_review=False, reconciled=True),
    ]
    block = build_footnote_block(claims)
    assert "[^1]" in block
    assert "[^3]" in block
    assert "Supported" in block
    assert "Contradicted" in block
    assert "HUMAN REVIEW" in block
    assert "RECONCILED" in block
    assert "example.com" in block


def test_rebuild_reads_findings_json(tmp_path):
    import json
    findings = tmp_path / "f.json"
    findings.write_text(json.dumps({
        "article_file": "a.md", "article_summary": "Test.", "total_findings": 1,
        "findings": [{
            "finding_id": "fc-001", "check_type": "fact_check", "severity": "PASS",
            "target_text": "Acme polluted.", "anchor_text": "Acme polluted the river.",
            "agent_summary": "EPA confirms.",
        }]
    }))
    article = tmp_path / "a.md"
    article.write_text("Acme polluted the river.")
    out = tmp_path / "s.md"
    from pipeline.stage_c_rebuild import run_stage_c
    run_stage_c(str(article), findings_path=str(findings), output_path=str(out))
    result = out.read_text()
    assert "PASS" in result or "✓" in result
    assert "Acme" in result


def test_find_quote_position_disambiguates_via_context():
    """Same phrase (letters-only) appears twice; context should pick the right one."""
    article = (
        "# Report\n\n"
        "In Karnes County, the facility irrigates 165 acres using sprinklers.\n\n"
        "Meanwhile in Live Oak County, a different facility irrigates 165 acres "
        "using sprinklers as well.\n\n"
        "The Live Oak facility was cited for overspray in 2023."
    )
    quote = "irrigates 165 acres using sprinklers"

    # Without context: first occurrence (Karnes) wins -- unchanged behavior.
    pos_default = find_quote_position(quote, article)
    assert pos_default is not None
    assert "Karnes" in article[:pos_default[0]]

    # With context pointing at the Live Oak paragraph: second occurrence wins.
    pos_context = find_quote_position(
        quote, article,
        context=("Meanwhile in Live Oak County, a different facility irrigates 165 "
                  "acres using sprinklers as well."),
    )
    assert pos_context is not None
    assert "Live Oak" in article[:pos_context[0]]
    assert pos_context != pos_default


def test_find_quote_position_levenshtein_does_not_confidently_pick_wrong_sentence():
    """Two near-identical sentences differing in 4 swappable words; a blended
    quote drawing 2 words from each side produces a genuine near-tie between
    the two candidate windows (top two Levenshtein ratios within ~1% of each
    other -- see printed output). This test characterizes real behavior: even
    at that near-tie margin, the fallback must land ENTIRELY within one real
    sentence, not straddle/blend both."""
    sentence1 = ("In 2021, the northern plant was cited for a spill that affected "
                 "local wildlife.")
    sentence2 = ("In 2022, the southern refinery was cited for a leak that affected "
                 "nearby wildlife.")
    article = "# Report\n\n" + sentence1 + "\n\n" + sentence2
    blended_quote = "southern refinery was cited for a spill that affected local wildlife"

    pos = find_quote_position(blended_quote, article)
    assert pos is not None, "expected the fallback to find SOME match for this near-tie quote"

    s1_start = article.index(sentence1)
    s1_end = s1_start + len(sentence1)
    s2_start = article.index(sentence2)
    s2_end = s2_start + len(sentence2)

    matched_text = article[pos[0]:pos[1]]
    print(f"\nMatched: {matched_text!r} at {pos}")

    within_sentence1 = s1_start <= pos[0] and pos[1] <= s1_end
    within_sentence2 = s2_start <= pos[0] and pos[1] <= s2_end
    assert within_sentence1 or within_sentence2, (
        f"match at {pos} ({matched_text!r}) straddles both sentences -- "
        f"sentence1 span={(s1_start, s1_end)}, sentence2 span={(s2_start, s2_end)}"
    )
