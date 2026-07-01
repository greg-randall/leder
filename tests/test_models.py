import json
import pytest
from pipeline.models import Claim, ClaimsDocument, Article, Corpus


def test_claim_defaults():
    c = Claim(claim_id="c001", claim_text="Test claim.", source_quote="Test claim.", claim_type="attribution")
    assert c.verdict is None
    assert c.source_proximity is None
    assert c.source_path is None
    assert c.human_review is None


def test_claims_document_round_trip():
    doc = ClaimsDocument(
        article=Article(path="test.md", title="Test"),
        corpus=Corpus(root="corpus/", project="test"),
        claims=[
            Claim(
                claim_id="c001",
                claim_text="A test claim.",
                source_quote="A test claim.",
                claim_type="numeric",
                verdict="supported",
                source_proximity="original",
                source_path="test/doc.md",
                source_url=None,
                rationale="Found in doc.",
                human_review=None,
                confidence=0.95,
            )
        ],
    )
    data = json.loads(doc.to_json())
    assert data["claims"][0]["verdict"] == "supported"
    loaded = ClaimsDocument.from_json(json.dumps(data))
    assert loaded.claims[0].claim_text == "A test claim."


def test_claim_type_validation():
    with pytest.raises(ValueError):
        Claim(claim_id="c001", claim_text="X", source_quote="X", claim_type="invalid")


def test_verdict_validation():
    with pytest.raises(ValueError):
        Claim(claim_id="c001", claim_text="X", source_quote="X", claim_type="numeric", verdict="invalid")


def test_proximity_validation():
    with pytest.raises(ValueError):
        Claim(claim_id="c001", claim_text="X", source_quote="X", claim_type="numeric", source_proximity="invalid")


def test_reconciled_default():
    c = Claim(claim_id="c001", claim_text="X", source_quote="X", claim_type="generalization")
    assert c.reconciled is False
