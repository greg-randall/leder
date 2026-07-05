"""Tests for the playbook YAML loader."""
from __future__ import annotations

import textwrap

import pytest
from pipeline.playbook import Playbook, load_playbook


def test_load_playbook_parses_minimal(tmp_path):
    y = tmp_path / "test.yaml"
    y.write_text(textwrap.dedent("""\
        name: "Test Check"
        extraction:
          prompt: "Extract things."
        verification:
          prompt: "Verify things."
    """))
    pb = load_playbook(str(y))
    assert pb.name == "Test Check"
    assert pb.extraction_prompt == "Extract things."
    assert pb.verification_prompt == "Verify things."
    assert pb.quality_gate_enabled is False
    assert pb.quality_gate_prompt == ""
    assert pb.allowed_tools == ["Read"]
    assert pb.display_template == ""


def test_load_playbook_full(tmp_path):
    y = tmp_path / "test.yaml"
    y.write_text(textwrap.dedent("""\
        name: "Full Check"
        description: "Does stuff."
        extraction:
          prompt: "Extract."
          quality_gate:
            enabled: true
            prompt: "Missed anything?"
        verification:
          prompt: "Verify."
          allowed_tools: [Read, Bash, WebSearch]
        display:
          template: "{{severity_badge}} **{{target_text}}**"
    """))
    pb = load_playbook(str(y))
    assert pb.quality_gate_enabled is True
    assert pb.quality_gate_prompt == "Missed anything?"
    assert pb.allowed_tools == ["Read", "Bash", "WebSearch"]
    assert "severity_badge" in pb.display_template


def test_load_playbook_quality_gate_disabled_by_default(tmp_path):
    y = tmp_path / "test.yaml"
    y.write_text(textwrap.dedent("""\
        name: "No QC"
        extraction:
          prompt: "E"
        verification:
          prompt: "V"
    """))
    pb = load_playbook(str(y))
    assert pb.quality_gate_enabled is False
    assert pb.quality_gate_prompt == ""


def test_load_playbook_missing_extraction_prompt_raises(tmp_path):
    y = tmp_path / "test.yaml"
    y.write_text("name: Bad\nverification:\n  prompt: V\n")
    with pytest.raises(ValueError, match="extraction.prompt"):
        load_playbook(str(y))


def test_load_playbook_missing_verification_prompt_raises(tmp_path):
    y = tmp_path / "test.yaml"
    y.write_text("name: Bad\nextraction:\n  prompt: E\n")
    with pytest.raises(ValueError, match="verification.prompt"):
        load_playbook(str(y))
