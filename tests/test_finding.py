"""Tests for the unified finding schema."""
from __future__ import annotations

import json

from pipeline.finding import Severity, Target, Finding, FindingsDocument


def test_severity_values():
    assert Severity.PASS.value == "PASS"
    assert Severity.WARNING.value == "WARNING"
    assert Severity.CRITICAL.value == "CRITICAL"


def test_target_round_trip():
    t = Target(
        target_text="LA-0304 irrigates 165 acres.",
        anchor_text="LA-0304 irrigates 165 acres in Karnes County via sprinkler.",
        context="The Rickaway Energy facility...",
        claim_type="numeric",
        playbook="fact_check",
    )
    d = t.to_dict()
    t2 = Target.from_dict(d)
    assert t2.target_text == t.target_text
    assert t2.playbook == "fact_check"


def test_finding_round_trip():
    f = Finding(
        finding_id="fc-001",
        check_type="fact_check",
        severity=Severity.PASS,
        target_text="LA-0304 irrigates 165 acres.",
        anchor_text="LA-0304 irrigates 165 acres in Karnes County via sprinkler.",
        agent_summary="The 2020 permit renewal states...",
        source_path="LA-0304/Permits/renewal.md",
        source_excerpt='"shall irrigate 165 acres via sprinkler"',
        confidence=0.95,
        human_review=False,
        metadata={"claim_type": "numeric", "source_proximity": "original"},
    )
    d = f.to_dict()
    f2 = Finding.from_dict(d)
    assert f2.finding_id == "fc-001"
    assert f2.severity == Severity.PASS
    assert f2.metadata["claim_type"] == "numeric"


def test_findings_document_round_trip():
    findings = [
        Finding(
            finding_id="fc-001", check_type="fact_check", severity=Severity.PASS,
            target_text="x", anchor_text="x", agent_summary="ok",
        ),
    ]
    doc = FindingsDocument(
        article_file="article.md",
        article_summary="An investigation into...",
        findings=findings,
    )
    raw = doc.to_json()
    doc2 = FindingsDocument.from_json(raw)
    assert doc2.total_findings == 1
    assert doc2.findings[0].finding_id == "fc-001"


def test_finding_defaults():
    f = Finding(
        finding_id="fc-001", check_type="fact_check", severity=Severity.WARNING,
        target_text="x", anchor_text="x", agent_summary="ok",
    )
    assert f.source_path is None
    assert f.source_url is None
    assert f.source_excerpt is None
    assert f.recommended_action is None
    assert f.confidence is None
    assert f.human_review is None
    assert f.metadata == {}


def test_target_from_dict_defaults():
    t = Target.from_dict({"target_text": "x", "anchor_text": "x", "playbook": "fc"})
    assert t.context == ""
    assert t.claim_type is None
