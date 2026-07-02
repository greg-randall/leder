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

    def fake_llm(system, user, model, max_tokens=4096):
        seen[model] = max_tokens
        return "x"

    monkeypatch.setattr(p3, "call_text_llm", fake_llm)
    p3.run_prepare_3(corpus_root=str(corpus), model="pro",
                     big_call_model="pro[1m]", workers=1,
                     crosscutting=True, force=True)
    assert "pro[1m]" in seen  # crosscutting call used the big-call model


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
