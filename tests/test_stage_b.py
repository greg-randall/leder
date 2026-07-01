"""Tests for Stage B: Claim verification via Claude Agent SDK."""
import pytest
import os
from pipeline.stage_b_verify import (
    AGENT_SYSTEM_PROMPT,
    agent_failure_result,
    parse_verdict,
)
from pipeline.models import Claim


def test_agent_system_prompt_contains_key_rules():
    assert "SEARCH STRATEGY" in AGENT_SYSTEM_PROMPT
    assert "Summaries are a map" in AGENT_SYSTEM_PROMPT
    assert "ORIGINALS (MANDATORY VERIFICATION STEP)" in AGENT_SYSTEM_PROMPT
    assert "OUTPUT" in AGENT_SYSTEM_PROMPT
    assert '"verdict"' in AGENT_SYSTEM_PROMPT


def test_parse_verdict_valid_json():
    claim = Claim(
        claim_id="c0001",
        claim_text="Test claim.",
        source_quote="Test claim.",
        claim_type="numeric",
    )
    agent_output = """
I searched the corpus and found the relevant permit.

{"verdict":"supported","source_proximity":"original","source_path":"LA-0304/permits/2020.md","source_url":null,"rationale":"The permit confirms 165 acres of irrigation.","human_review":false,"confidence":0.95}
"""
    result = parse_verdict(claim, agent_output)
    assert result.verdict == "supported"
    assert result.source_proximity == "original"
    assert result.source_path == "LA-0304/permits/2020.md"
    assert result.rationale == "The permit confirms 165 acres of irrigation."
    assert result.human_review is False
    assert result.confidence == 0.95


def test_parse_verdict_multiline_json():
    claim = Claim(
        claim_id="c0002",
        claim_text="Another claim.",
        source_quote="Another claim.",
        claim_type="attribution",
    )
    agent_output = """Some preamble text.

{
  "verdict": "contradicted",
  "source_proximity": "derived",
  "source_path": "overview.md",
  "source_url": null,
  "rationale": "The overview says something different.",
  "human_review": true,
  "confidence": 0.5
}

Some trailing text."""
    result = parse_verdict(claim, agent_output)
    assert result.verdict == "contradicted"
    assert result.human_review is True
    assert result.confidence == 0.5


def test_parse_verdict_no_json():
    claim = Claim(
        claim_id="c0003",
        claim_text="No JSON here.",
        source_quote="No JSON here.",
        claim_type="generalization",
    )
    result = parse_verdict(claim, "Just some text without any JSON object.")
    assert result.verdict == "unsupported"
    assert result.source_proximity == "unverifiable"
    assert result.human_review is True


def test_parse_verdict_invalid_json():
    claim = Claim(
        claim_id="c0004",
        claim_text="Bad JSON.",
        source_quote="Bad JSON.",
        claim_type="legal",
    )
    result = parse_verdict(claim, '{"verdict": "supported", invalid}')
    assert result.verdict == "unsupported"
    assert result.human_review is True


def test_parse_verdict_missing_field():
    claim = Claim(
        claim_id="c0005",
        claim_text="Missing field.",
        source_quote="Missing field.",
        claim_type="numeric",
    )
    result = parse_verdict(claim, '{"verdict":"supported","rationale":"ok.","human_review":false,"confidence":0.9}')
    assert result.verdict == "unsupported"
    assert result.human_review is True


def test_parse_verdict_empty_text():
    claim = Claim(
        claim_id="c0006",
        claim_text="Empty.",
        source_quote="Empty.",
        claim_type="numeric",
    )
    result = parse_verdict(claim, "")
    assert result.verdict == "unsupported"
    assert result.human_review is True


def test_agent_failure_fallback():
    claim = Claim(
        claim_id="c9999",
        claim_text="Test claim for failure.",
        source_quote="Test claim for failure.",
        claim_type="generalization",
    )
    result = agent_failure_result(claim)
    assert result.verdict == "unsupported"
    assert result.source_proximity == "unverifiable"
    assert result.human_review is True
    assert result.rationale is not None


@pytest.mark.integration
def test_verify_claim_integration():
    """Integration test: requires Claude Code CLI and API key."""
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")):
        pytest.skip("No API key configured")
    corpus_root = "/mnt/e/Google Drive/rrc-permit-files/g-journalism-run/source-docs-and-summaries"
    if not os.path.exists(corpus_root):
        pytest.skip("Corpus not available")

    import asyncio
    from pipeline.stage_b_verify import _verify_claim_async

    claim = Claim(
        claim_id="c_test",
        claim_text="LA-0304 irrigates 165 acres in Karnes County via sprinkler.",
        source_quote="irrigates 165 acres in Karnes County via sprinkler",
        claim_type="numeric",
    )

    result = asyncio.run(_verify_claim_async(claim, corpus_root, AGENT_SYSTEM_PROMPT))
    assert result.verdict is not None
    assert result.rationale is not None
    assert len(result.rationale) > 5
