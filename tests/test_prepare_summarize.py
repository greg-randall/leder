"""Tests for prepare-2 (summarize) and the shared text-LLM helper."""
from __future__ import annotations

from pathlib import Path
from unittest import mock

import pipeline.prepare_2_summarize as p2


def test_call_text_llm_concats_text_blocks(monkeypatch):
    from pipeline import llm

    class _Block:
        def __init__(self, text): self.type = "text"; self.text = text

    class _Resp:
        content = [_Block("hello "), _Block("world")]

    fake_client = mock.Mock()
    fake_client.messages.create.return_value = _Resp()
    monkeypatch.setattr(llm.anthropic, "Anthropic", lambda **kw: fake_client)

    out = llm.call_text_llm("sys", "user", model="m", max_tokens=100)
    assert out == "hello world"
    assert fake_client.messages.create.call_args.kwargs["model"] == "m"


def test_run_prepare_2_writes_summaries(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    (corpus / "CaseA").mkdir(parents=True)
    doc = corpus / "CaseA" / "doc1.md"
    doc.write_text("This is a real document with enough words to summarize. " * 5)

    monkeypatch.setattr(p2, "call_text_llm", lambda *a, **k: "**Summary:** ok\n\n**Facts:** none\n")
    p2.run_prepare_2(corpus_root=str(corpus), model="m", workers=2, force=False)

    summary = corpus / "CaseA" / "doc1_summary.md"
    assert summary.exists()
    assert "Summary" in summary.read_text()


def test_run_prepare_2_skips_short_files(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "tiny.md").write_text("x")
    called = []
    monkeypatch.setattr(p2, "call_text_llm", lambda *a, **k: called.append(1) or "x")
    p2.run_prepare_2(corpus_root=str(corpus), model="m", workers=1, force=False)
    # stub/empty summary written, LLM not called
    assert (corpus / "tiny_summary.md").exists()
    assert called == []
