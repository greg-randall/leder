"""Tests for prepare-3 recursive rollup."""
from __future__ import annotations

from pathlib import Path

import pipeline.prepare_3_rollup as p3


def _mk(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_recursive_folder_summaries(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    # nested tree: corpus/A/sub/doc_summary.md and corpus/A/top_summary.md
    _mk(corpus / "A" / "sub" / "doc_summary.md", "**Summary:** sub doc.")
    _mk(corpus / "A" / "top_summary.md", "**Summary:** top doc.")
    _mk(corpus / "B" / "only_summary.md", "**Summary:** lone doc.")

    calls = []

    def fake_llm(system, user, model, max_tokens=4096):
        calls.append(model)
        return f"FOLDER SUMMARY ({len(user)} chars)"

    monkeypatch.setattr(p3, "call_text_llm", fake_llm)
    p3.run_prepare_3(corpus_root=str(corpus), model="pro",
                     big_call_model="pro[1m]", workers=2,
                     crosscutting=True, force=True)

    assert (corpus / "A" / "sub" / "_FOLDER_SUMMARY.md").exists()
    assert (corpus / "A" / "_FOLDER_SUMMARY.md").exists()
    # B has a single summary + no subfolders -> reuse, content equals the child summary
    assert (corpus / "B" / "_FOLDER_SUMMARY.md").read_text() == "**Summary:** lone doc."
    # root overview + crosscutting + concat exist
    assert (corpus / "CORPUS_OVERVIEW.md").exists()
    assert (corpus / "CORPUS_CROSSCUTTING.md").exists()
    assert (corpus / "ALL_SUMMARIES.md").exists()


def test_crosscutting_uses_big_call_model(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    _mk(corpus / "A" / "doc_summary.md", "**Summary:** a.")
    seen = {}

    def fake_llm(system, user, model, max_tokens=4096, stream=False):
        seen[model] = max_tokens
        return "x"

    monkeypatch.setattr(p3, "call_text_llm", fake_llm)
    p3.run_prepare_3(corpus_root=str(corpus), model="pro",
                     big_call_model="pro[1m]", workers=1,
                     crosscutting=True, force=True)
    assert "pro[1m]" in seen  # crosscutting call used the big-call model


def test_build_folder_prompt_de_domained_and_cites_sources():
    from pipeline.prepare_3_rollup import build_folder_prompt

    p = build_folder_prompt("Transcripts of Waskom city council meetings.")
    assert "RAD" not in p
    assert "permit" not in p.lower()
    assert "Transcripts of Waskom city council meetings." in p
    assert "source document" in p.lower()
    # Citation is delimited by the literal text "Source:", not by enclosing
    # brackets -- corpus filenames themselves already contain brackets (e.g.
    # YouTube-ID suffixes), so wrapping the citation in brackets would nest
    # and create ambiguity. The worked example still contains a real
    # bracket-bearing filename, demonstrating the format is collision-safe.
    assert "Source:" in p
    assert "[" in p and "]" in p


def test_build_crosscutting_prompt_has_entity_index_and_topic_map():
    from pipeline.prepare_3_rollup import build_crosscutting_prompt

    p = build_crosscutting_prompt("A test corpus.")
    assert "Entity index" in p
    assert "Topic" in p and "document map" in p.lower()
    assert "Suggested next analyses" not in p
    assert "Cross-cutting patterns" in p
    assert "Data highlights" in p
    assert "A test corpus." in p


def test_run_prepare_3_threads_corpus_description(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    # A: one subdir -> exercises summarize_folder's own sequential ("else")
    # recursion branch, which only fires when a folder has exactly one subdir.
    _mk(corpus / "A" / "sub" / "e1_summary.md", "**Summary:** e1.")
    _mk(corpus / "A" / "sub" / "e2_summary.md", "**Summary:** e2.")
    # B: two subdirs -> exercises summarize_folder's own ThreadPoolExecutor
    # ("_TPE") parallel-recursion branch.
    _mk(corpus / "B" / "sub1" / "g1_summary.md", "**Summary:** g1.")
    _mk(corpus / "B" / "sub1" / "g2_summary.md", "**Summary:** g2.")
    _mk(corpus / "B" / "sub2" / "h1_summary.md", "**Summary:** h1.")
    _mk(corpus / "B" / "sub2" / "h2_summary.md", "**Summary:** h2.")

    captured = []

    def fake_llm(system, user, model, max_tokens=4096, stream=False):
        captured.append(system)
        return "x"

    monkeypatch.setattr(p3, "call_text_llm", fake_llm)
    p3.run_prepare_3(corpus_root=str(corpus), model="pro", big_call_model="pro[1m]",
                     workers=1, crosscutting=False, force=True,
                     corpus_description="A test corpus about widgets.")

    # Calls happen at multiple nesting levels, including ones reached only via
    # summarize_folder's own recursive call sites (A/sub via the sequential
    # branch, B/sub1 and B/sub2 via the ThreadPoolExecutor branch). Every one
    # of them must carry corpus_description through -- if either recursive
    # call site dropped the parameter, at least one captured system prompt
    # would be missing it.
    assert len(captured) >= 5
    assert all("A test corpus about widgets." in s for s in captured)


def test_failed_folder_summary_not_persisted(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    _mk(corpus / "A" / "d1_summary.md", "**Summary:** one.")
    _mk(corpus / "A" / "d2_summary.md", "**Summary:** two.")  # 2 summaries -> LLM path

    def boom(system, user, model, max_tokens=4096):
        raise RuntimeError("nope")

    monkeypatch.setattr(p3, "call_text_llm", boom)
    p3.run_prepare_3(corpus_root=str(corpus), model="pro", big_call_model="pro[1m]",
                     workers=1, crosscutting=False, force=True)
    # failed folder summary must NOT be written, so a re-run retries (no poison string)
    assert not (corpus / "A" / "_FOLDER_SUMMARY.md").exists()
    assert not (corpus / "CORPUS_OVERVIEW.md").exists()
