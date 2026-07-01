"""Data model shared across all three pipeline stages."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional
from enum import Enum


class ClaimType(str, Enum):
    NUMERIC = "numeric"
    ATTRIBUTION = "attribution"
    LEGAL = "legal"
    GENERALIZATION = "generalization"


class Verdict(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNSUPPORTED = "unsupported"


class SourceProximity(str, Enum):
    ORIGINAL = "original"
    DERIVED = "derived"
    UNVERIFIABLE = "unverifiable"


@dataclass
class Article:
    path: str
    title: str
    generated_at: str = ""


@dataclass
class Corpus:
    root: str
    project: str


@dataclass
class Claim:
    """A single verifiable claim extracted from the article."""
    claim_id: str
    claim_text: str
    source_quote: str
    claim_type: ClaimType

    # Stage B output (nullable until populated)
    verdict: Optional[Verdict] = None
    source_proximity: Optional[SourceProximity] = None
    source_path: Optional[str] = None
    source_url: Optional[str] = None
    rationale: Optional[str] = None
    human_review: Optional[bool] = None
    confidence: Optional[float] = None

    # Stage C reconciliation
    reconciled: bool = False

    def __post_init__(self):
        # Convert and validate claim_type
        if isinstance(self.claim_type, str):
            if self.claim_type not in ClaimType._value2member_map_:
                raise ValueError(f"Invalid claim_type: {self.claim_type}")
            self.claim_type = ClaimType(self.claim_type)
        elif not isinstance(self.claim_type, ClaimType):
            raise ValueError(f"Invalid claim_type: {self.claim_type}")

        # Convert and validate verdict
        if self.verdict is not None:
            if isinstance(self.verdict, str):
                if self.verdict not in Verdict._value2member_map_:
                    raise ValueError(f"Invalid verdict: {self.verdict}")
                self.verdict = Verdict(self.verdict)
            elif not isinstance(self.verdict, Verdict):
                raise ValueError(f"Invalid verdict: {self.verdict}")

        # Convert and validate source_proximity
        if self.source_proximity is not None:
            if isinstance(self.source_proximity, str):
                if self.source_proximity not in SourceProximity._value2member_map_:
                    raise ValueError(f"Invalid source_proximity: {self.source_proximity}")
                self.source_proximity = SourceProximity(self.source_proximity)
            elif not isinstance(self.source_proximity, SourceProximity):
                raise ValueError(f"Invalid source_proximity: {self.source_proximity}")

        # Validate confidence range
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be between 0.0 and 1.0, got {self.confidence}"
            )


@dataclass
class ClaimsDocument:
    """Top-level document: the contract between all pipeline stages."""
    article: Article
    corpus: Corpus
    claims: list[Claim] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(self._to_dict(), indent=2, ensure_ascii=False)

    def _to_dict(self) -> dict:
        return {
            "article": asdict(self.article),
            "corpus": asdict(self.corpus),
            "claims": [asdict(c) for c in self.claims],
        }

    @classmethod
    def from_json(cls, raw: str) -> ClaimsDocument:
        data = json.loads(raw)
        return cls(
            article=Article(**data["article"]),
            corpus=Corpus(**data["corpus"]),
            claims=[Claim(**c) for c in data["claims"]],
        )
