"""Unified finding schema — the data contract between all pipeline stages."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


class Severity(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class Target:
    """A single extracted item from Stage 1, before verification."""
    target_text: str
    anchor_text: str
    playbook: str
    context: str = ""
    claim_type: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict) -> "Target":
        return cls(
            target_text=d["target_text"],
            anchor_text=d["anchor_text"],
            playbook=d["playbook"],
            context=d.get("context", ""),
            claim_type=d.get("claim_type"),
        )


@dataclass
class Finding:
    """A verified editorial finding from Stage 2."""
    finding_id: str
    check_type: str
    severity: Severity
    target_text: str
    anchor_text: str
    agent_summary: str
    context: str = ""
    recommended_action: Optional[str] = None
    source_path: Optional[str] = None
    source_url: Optional[str] = None
    source_excerpt: Optional[str] = None
    confidence: Optional[float] = None
    human_review: Optional[bool] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        VALID = frozenset({"PASS", "WARNING", "CRITICAL"})
        if isinstance(self.severity, Severity):
            pass
        elif isinstance(self.severity, str):
            if self.severity not in VALID:
                raise ValueError(
                    f"Invalid severity: {self.severity}. "
                    f"Must be one of {sorted(VALID)}")
            self.severity = Severity(self.severity)
        else:
            raise ValueError(
                f"severity must be string or Severity, got {type(self.severity)}")

        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be between 0.0 and 1.0, got {self.confidence}")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["severity"] = self.severity.value
        return {k: v for k, v in d.items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict) -> "Finding":
        return cls(
            finding_id=d["finding_id"],
            check_type=d["check_type"],
            severity=d["severity"],
            target_text=d["target_text"],
            anchor_text=d["anchor_text"],
            agent_summary=d["agent_summary"],
            context=d.get("context", ""),
            recommended_action=d.get("recommended_action"),
            source_path=d.get("source_path"),
            source_url=d.get("source_url"),
            source_excerpt=d.get("source_excerpt"),
            confidence=d.get("confidence"),
            human_review=d.get("human_review"),
            metadata=d.get("metadata", {}),
        )


@dataclass
class FindingsDocument:
    """Top-level document: the contract between Stages 2 -> C/D/E."""
    article_file: str
    article_summary: str
    findings: list[Finding] = field(default_factory=list)

    @property
    def total_findings(self) -> int:
        return len(self.findings)

    def to_json(self) -> str:
        return json.dumps(self._to_dict(), indent=2, ensure_ascii=False)

    def _to_dict(self) -> dict:
        return {
            "article_file": self.article_file,
            "article_summary": self.article_summary,
            "total_findings": self.total_findings,
            "findings": [f.to_dict() for f in self.findings],
        }

    @classmethod
    def from_json(cls, raw: str) -> "FindingsDocument":
        data = json.loads(raw)
        return cls(
            article_file=data["article_file"],
            article_summary=data["article_summary"],
            findings=[Finding.from_dict(f) for f in data.get("findings", [])],
        )
