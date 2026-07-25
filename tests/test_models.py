import pytest
from pipeline.models import Claim


def test_claim_defaults():
    c = Claim(claim_id="c001", claim_text="Test claim.", source_quote="Test claim.", claim_type="attribution")
    assert c.verdict is None
    assert c.source_proximity is None
    assert c.source_path is None
    assert c.human_review is None


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
