"""Tests for Stage A: Claim extraction."""
import json
import os
import tempfile

import pytest

from pipeline.stage_a_extract import (
    build_extraction_prompt,
    build_quality_gate_prompt,
    extract_claims,
    run_stage_a,
)

SAMPLE_ARTICLE = """# Test Article

LA-0304 was originally permitted in 2001 to Koch Midstream Services.
The facility irrigates 165 acres in Karnes County using sprinklers.
Produced water chlorides range from 80 to 200 mg/L at this site.

This is an opinion statement that should be skipped.

SB 1145 transfers jurisdiction from RRC to TCEQ.
"""


def test_build_extraction_prompt():
    prompt = build_extraction_prompt(SAMPLE_ARTICLE)
    assert "LA-0304" in prompt
    assert "STANDALONE" not in prompt  # It's in system prompt, not user prompt


def test_build_quality_gate_prompt():
    existing = ["LA-0304 was originally permitted in 2001 to Koch Midstream Services."]
    prompt = build_quality_gate_prompt(SAMPLE_ARTICLE, existing)
    assert "Koch Midstream" in prompt
    assert "MISSED" in prompt


@pytest.mark.integration
def test_extract_claims_integration():
    """Integration test: requires Anthropic API key in env."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    doc = extract_claims(SAMPLE_ARTICLE, model="claude-sonnet-5", quality_gate=False)
    assert len(doc.claims) >= 3
    assert all(c.claim_id for c in doc.claims)
    assert all(c.source_quote for c in doc.claims)
    for c in doc.claims:
        assert len(c.claim_text) > 20, f"Claim too short/generic: {c.claim_text}"


@pytest.mark.integration
def test_run_stage_a_writes_file():
    """Integration test: writes claims.json."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    with tempfile.TemporaryDirectory() as tmp:
        article_path = os.path.join(tmp, "article.md")
        with open(article_path, "w") as f:
            f.write(SAMPLE_ARTICLE)
        output_path = os.path.join(tmp, "claims.json")
        doc = run_stage_a(
            article_path=article_path,
            output_path=output_path,
            corpus_root=tmp,
            project_name="test",
            model="claude-sonnet-5",
            quality_gate=False,
        )
        assert os.path.exists(output_path)
        loaded = json.loads(open(output_path).read())
        assert len(loaded["claims"]) >= 3


def test_missing_article_raises():
    with pytest.raises(FileNotFoundError):
        run_stage_a("/nonexistent/article.md", "/tmp/out.json", "/tmp", "test")


def test_empty_article_raises():
    with tempfile.TemporaryDirectory() as tmp:
        article_path = os.path.join(tmp, "empty.md")
        with open(article_path, "w") as f:
            f.write("")
        with pytest.raises(ValueError):
            run_stage_a(article_path, os.path.join(tmp, "out.json"), tmp, "test")
