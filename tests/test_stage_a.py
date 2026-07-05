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


def test_extract_targets_from_text_uses_playbook_prompt(monkeypatch):
    from pipeline.playbook import Playbook
    from pipeline.stage_a_extract import _extract_targets_from_text, _extraction_tool_for

    pb = Playbook(
        name="test_check",
        extraction_prompt="Extract things from: {{article_text}}",
        verification_prompt="Verify.",
    )

    class FakeText:
        type = "tool_use"
        input = {"targets": [{"target_text": "t1", "anchor_text": "a1"}],
                 "article_title": "Title", "article_summary": "Summary"}

    fake_client = type("c", (), {
        "messages": type("m", (), {
            "create": lambda self, **kw: type("r", (), {"content": [FakeText()]})()
        })()
    })()
    monkeypatch.setattr("anthropic.Anthropic", lambda **kw: fake_client)

    targets, title, summary = _extract_targets_from_text(
        "some article text", "m", pb, _extraction_tool_for(pb))
    assert len(targets) == 1
    assert targets[0].target_text == "t1"
    assert targets[0].playbook == "test_check"
    assert title == "Title"
    assert summary == "Summary"


def test_run_stage_a_playbook_path_writes_targets_json(tmp_path, monkeypatch):
    """run_stage_a with playbook_names writes targets.json tagged with playbook."""
    article = tmp_path / "article.md"
    article.write_text(
        "Some article text about a very important topic. "
        "Another sentence with more context. "
        "Yet another sentence to reach the minimum word threshold required."
    )
    output = str(tmp_path / "targets.json")

    # Build a minimal playbook in a temp dir
    pd = tmp_path / "playbooks"
    pd.mkdir()
    (pd / "test_check.yaml").write_text("""
        name: Test Check
        extraction:
          prompt: "Extract from: {{article_text}}"
        verification:
          prompt: Verify.
    """)

    # Mock the LLM so no real API call
    class FakeTargetTool:
        type = "tool_use"
        input = {"targets": [{"target_text": "t1", "anchor_text": "a1"}],
                 "article_title": "T", "article_summary": "S"}

    fake = type("c", (), {
        "messages": type("m", (), {
            "create": lambda self, **kw: type("r", (), {"content": [FakeTargetTool()]})()
        })()
    })()
    monkeypatch.setattr("anthropic.Anthropic", lambda **kw: fake)

    from pipeline.stage_a_extract import run_stage_a
    run_stage_a(
        article_path=str(article),
        output_path=output,
        corpus_root="",
        project_name="",
        model="m",
        quality_gate=False,
        playbook_dir=str(pd),
        playbook_names=["test_check"],
    )

    import json
    data = json.loads(open(output).read())
    assert data["article_file"] == str(article)
    assert len(data["targets"]) == 1
    assert data["targets"][0]["playbook"] == "Test Check"
    assert data["targets"][0]["target_text"] == "t1"
