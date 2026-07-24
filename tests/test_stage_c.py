"""Tests for Stage C: Article rebuild with source footnotes."""
from types import SimpleNamespace

from pipeline.stage_c_rebuild import (
    normalize_text,
    find_quote_position,
    insert_footnote_markers,
    build_footnote_block,
    build_unplaced_warning,
)
import pipeline.stage_c_rebuild as sc
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


class _FakeTextBlock:
    def __init__(self, text):
        self.text = text


class _FakeResponse:
    def __init__(self, text):
        self.content = [_FakeTextBlock(text)]


def _make_reconcile_claim(quote="broken quote text"):
    return Claim(
        claim_id="c001", claim_text="Some claim.", source_quote=quote,
        claim_type="attribution", verdict="supported", source_proximity="original",
        source_path="test.md", rationale="r", human_review=False, reconciled=False,
    )


def test_reconcile_unmatched_quotes_corrects(monkeypatch):
    claim = _make_reconcile_claim()

    class FakeClient:
        def __init__(self, api_key=None):
            pass

        class messages:
            @staticmethod
            def create(**kwargs):
                return _FakeResponse(
                    '{"corrected_quote": "the real sentence in the article", "status": "corrected"}'
                )

    monkeypatch.setattr("anthropic.Anthropic", FakeClient)
    result = sc.reconcile_unmatched_quotes([claim], "article text", model="claude-sonnet-5")
    assert result[0].source_quote == "the real sentence in the article"
    assert result[0].reconciled is True


def test_reconcile_unmatched_quotes_no_match(monkeypatch):
    claim = _make_reconcile_claim()

    class FakeClient:
        def __init__(self, api_key=None):
            pass

        class messages:
            @staticmethod
            def create(**kwargs):
                return _FakeResponse('{"corrected_quote": null, "status": "no_match"}')

    monkeypatch.setattr("anthropic.Anthropic", FakeClient)
    result = sc.reconcile_unmatched_quotes([claim], "article text")
    assert result[0].source_quote == "broken quote text"
    assert result[0].reconciled is False


def test_reconcile_unmatched_quotes_malformed_json(monkeypatch):
    claim = _make_reconcile_claim()

    class FakeClient:
        def __init__(self, api_key=None):
            pass

        class messages:
            @staticmethod
            def create(**kwargs):
                return _FakeResponse("not valid json at all")

    monkeypatch.setattr("anthropic.Anthropic", FakeClient)
    result = sc.reconcile_unmatched_quotes([claim], "article text")
    assert result[0].source_quote == "broken quote text"
    assert len(result) == 1


def test_reconcile_unmatched_quotes_api_exception(monkeypatch):
    claim = _make_reconcile_claim()

    class FakeClient:
        def __init__(self, api_key=None):
            pass

        class messages:
            @staticmethod
            def create(**kwargs):
                raise RuntimeError("API down")

    monkeypatch.setattr("anthropic.Anthropic", FakeClient)
    result = sc.reconcile_unmatched_quotes([claim], "article text")
    assert result[0].source_quote == "broken quote text"
    assert len(result) == 1


def test_reconcile_unmatched_quotes_empty_list_short_circuits(monkeypatch):
    def _explode(**kwargs):
        raise AssertionError("should not construct a client for an empty list")

    monkeypatch.setattr("anthropic.Anthropic", _explode)
    result = sc.reconcile_unmatched_quotes([], "article text")
    assert result == []


def test_merge_placed_groups_single_position_unchanged():
    claim_lookup = {"c001": SimpleNamespace(claim_id="c001", check_type="fact_check", severity="PASS")}
    placed = [("c001", (10, 20))]
    deduped, merged = sc._merge_placed_groups(placed, claim_lookup)
    assert deduped == [("c001", (10, 20))]
    assert len(merged) == 1
    assert merged[0] is claim_lookup["c001"]


def test_merge_placed_groups_combines_badges_at_same_position():
    claim_lookup = {
        "c001": SimpleNamespace(claim_id="c001", check_type="fact_check", severity="PASS"),
        "c002": SimpleNamespace(claim_id="c002", check_type="quote_precision", severity="CRITICAL"),
    }
    placed = [("c001", (10, 20)), ("c002", (10, 20))]
    deduped, merged = sc._merge_placed_groups(placed, claim_lookup)

    assert deduped == [("c001", (10, 20))]
    assert len(merged) == 1
    primary = merged[0]
    assert primary is claim_lookup["c001"]
    assert primary._merged_badges == ["✓ fact_check", "✗ quote_precision"]


def test_merge_placed_groups_different_positions_stay_separate():
    claim_lookup = {
        "c001": SimpleNamespace(claim_id="c001", check_type="fact_check", severity="PASS"),
        "c002": SimpleNamespace(claim_id="c002", check_type="fact_check", severity="WARNING"),
    }
    placed = [("c001", (10, 20)), ("c002", (30, 40))]
    deduped, merged = sc._merge_placed_groups(placed, claim_lookup)
    assert deduped == [("c001", (10, 20)), ("c002", (30, 40))]
    assert len(merged) == 2


# ── Unplaced-claims heading: producer/consumer contract ──────────

def test_build_unplaced_warning_heading_matches_downstream_expectations():
    """stage_d_html.py and stage_e_docx.py strip this block by searching
    for the literal heading '# ⚠️ UNPLACED CLAIMS' (with the warning
    emoji). If this producer ever emits a heading that doesn't match --
    exactly what happened before this test existed -- the strip regex in
    both consumers silently fails to match, and unplaced-claims content
    (including internal review notes like 'MANUAL REVIEW REQUIRED') leaks
    straight into the published HTML/DOCX output unstripped."""
    claim = Claim(
        claim_id="c001", claim_text="A claim.", source_quote="q",
        claim_type="attribution", verdict="supported", source_proximity="original",
        rationale="r", human_review=False, reconciled=False,
    )
    block = build_unplaced_warning([claim])
    assert block.startswith("# ⚠️ UNPLACED CLAIMS\n")
