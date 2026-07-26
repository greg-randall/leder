"""Tests for the shared prompt-ruleset module used by all playbooks."""
from __future__ import annotations

from pipeline.prompts import build_extraction_system_prompt, build_verification_rules_block


def test_extraction_prompt_contains_core_rules():
    p = build_extraction_system_prompt("")
    for rule in ("STANDALONE", "CONTEXT INJECTION", "GRANULARITY", "NO OPINIONS",
                 "PRESERVE PRECISION", "EXHAUSTIVE", "SOURCE QUOTE"):
        assert rule in p


def test_extraction_prompt_defines_claim_types():
    p = build_extraction_system_prompt("")
    for claim_type in ("numeric", "attribution", "legal", "generalization"):
        assert claim_type in p


def test_extraction_prompt_has_attribution_framing_rule():
    p = build_extraction_system_prompt("")
    assert "attribution framing" in p.lower()
    assert "X testified that Y" in p


def test_extraction_prompt_has_unique_anchor_rule():
    p = build_extraction_system_prompt("")
    assert "anchor_text must be" in p or "unique" in p.lower()
    assert "same sentence may share the same anchor" in p.lower() \
        or "share the same anchor" in p.lower()


def test_extraction_prompt_injects_corpus_description():
    p = build_extraction_system_prompt("Transcripts of city council meetings.")
    assert "Transcripts of city council meetings." in p


def test_extraction_prompt_well_formed_with_empty_description():
    p = build_extraction_system_prompt("")
    assert "{corpus_description}" not in p
    assert "{{" not in p and "}}" not in p


def test_verification_rules_has_sandbox_rule():
    p = build_verification_rules_block("")
    assert "SANDBOX" in p
    assert "relative paths only" in p.lower()


def test_verification_rules_has_tiered_search_strategy():
    p = build_verification_rules_block("")
    assert "_FOLDER_SUMMARY.md" in p
    assert "web_cache" in p
    assert "SEARCH STRATEGY" in p


def test_verification_rules_has_source_path_format_spec():
    p = build_verification_rules_block("")
    assert "source_path" in p
    assert "relative to the corpus root" in p


def test_verification_rules_has_corroboration_directive():
    p = build_verification_rules_block("")
    assert "corpus_contradicted_by_external" in p
    assert "corroboration" in p.lower()


def test_verification_rules_has_agent_summary_example():
    p = build_verification_rules_block("")
    assert "BAD:" in p and "GOOD:" in p


def test_verification_rules_has_full_confidence_rubric():
    p = build_verification_rules_block("")
    for band in ("0.95", "0.8", "0.6", "0.4", "0.2"):
        assert band in p
    assert "Hard caps" in p or "hard caps" in p.lower()
    assert "Worked examples" in p or "worked examples" in p.lower()


def test_verification_rules_has_date_discipline_rule():
    p = build_verification_rules_block("")
    assert "scheduled" in p.lower() or "future events" in p.lower()


def test_verification_rules_injects_corpus_description():
    p = build_verification_rules_block("Transcripts of city council meetings.")
    assert "Transcripts of city council meetings." in p


def test_verification_rules_well_formed_with_empty_description():
    p = build_verification_rules_block("")
    assert "{corpus_description}" not in p
    assert "{{" not in p and "}}" not in p


def test_verification_rules_has_validate_excerpt_step():
    p = build_verification_rules_block("")
    assert "VALIDATE EXCERPT (MANDATORY)" in p
    assert "validate_excerpt.py" in p
