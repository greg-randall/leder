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
