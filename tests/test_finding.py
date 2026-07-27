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


def _finding(**kw):
    from pipeline.finding import Finding
    base = dict(finding_id="f1", check_type="fact_check", severity="PASS",
                target_text="T", anchor_text="A", agent_summary="S")
    base.update(kw)
    return Finding(**base)


def test_finding_round_trips_excerpt_fields():
    from pipeline.finding import Finding
    f = _finding(source_excerpt="real text", source_excerpt_offset=[10, 19],
                 source_excerpt_similarity=0.82, excerpt_status="repaired")
    d = f.to_dict()
    assert d["source_excerpt_offset"] == [10, 19]
    assert d["source_excerpt_similarity"] == 0.82
    assert d["excerpt_status"] == "repaired"
    back = Finding.from_dict(d)
    assert back.source_excerpt_offset == [10, 19]
    assert back.source_excerpt_similarity == 0.82
    assert back.excerpt_status == "repaired"


def test_finding_omits_excerpt_fields_when_unset():
    d = _finding().to_dict()
    assert "source_excerpt_offset" not in d
    assert "source_excerpt_similarity" not in d
    assert "excerpt_status" not in d


def test_finding_from_dict_without_excerpt_fields():
    """Findings written before the excerpt gate existed must still load."""
    from pipeline.finding import Finding
    f = Finding.from_dict({
        "finding_id": "f1", "check_type": "fact_check", "severity": "PASS",
        "target_text": "T", "anchor_text": "A", "agent_summary": "S",
    })
    assert f.source_excerpt_offset is None
    assert f.excerpt_status is None


def test_finding_offset_survives_findings_document_json():
    """The fields must survive the FindingsDocument JSON round-trip, which is
    the actual wire format between stage B and stages C/D -- not just to_dict."""
    from pipeline.finding import FindingsDocument
    doc = FindingsDocument(
        article_file="a.md", article_summary="S",
        findings=[_finding(source_excerpt="real text",
                           source_excerpt_offset=[10, 19],
                           source_excerpt_similarity=1.0,
                           excerpt_status="exact")],
    )
    back = FindingsDocument.from_json(doc.to_json())
    assert back.findings[0].source_excerpt_offset == [10, 19]
    assert back.findings[0].source_excerpt_similarity == 1.0
    assert back.findings[0].excerpt_status == "exact"


def test_finding_real_findings_json_still_loads():
    """A findings.json that predates these fields must still parse --
    this is the backward-compatibility guarantee that lets us fix forward."""
    from pathlib import Path as _Path
    from pipeline.finding import FindingsDocument
    fixture = _Path(__file__).resolve().parent / "fixtures" / "legacy-findings.json"
    doc = FindingsDocument.from_json(fixture.read_text(encoding="utf-8"))
    assert doc.findings
    for f in doc.findings:
        assert f.source_excerpt_offset is None
        assert f.source_excerpt_similarity is None
        assert f.excerpt_status is None
