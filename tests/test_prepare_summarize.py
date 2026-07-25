"""Tests for prepare-2 (summarize) and the shared text-LLM helper."""
from __future__ import annotations

from unittest import mock

import pipeline.prepare_2_summarize as p2


def test_call_text_llm_concats_text_blocks(monkeypatch):
    from pipeline import llm

    class _Block:
        def __init__(self, text):
            self.type = "text"
            self.text = text

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


def test_collect_targets_skips_page_md(tmp_path):
    """web_cache page.md files should not be sent to the LLM summarizer."""
    corpus = tmp_path / "corpus"
    (corpus / "normal").mkdir(parents=True)
    (corpus / "normal" / "doc.md").write_text("A" * 200)
    (corpus / "web_cache").mkdir()
    (corpus / "web_cache" / "claim-x").mkdir()
    (corpus / "web_cache" / "claim-x" / "page.md").write_text("B" * 200)

    targets = p2.collect_targets(corpus)
    target_paths = [str(t.relative_to(corpus)) for t in targets]

    assert "normal/doc.md" in target_paths
    assert "web_cache/claim-x/page.md" not in target_paths


def test_collect_targets_skips_page_md_at_any_depth(tmp_path):
    """page.md is skipped regardless of nesting depth."""
    corpus = tmp_path / "corpus"
    # Deeply nested web_cache
    deep = corpus / "a" / "b" / "web_cache" / "deep" / "claim-z"
    deep.mkdir(parents=True)
    (deep / "page.md").write_text("C" * 200)

    # Also a real doc that should be included
    (corpus / "real.md").write_text("D" * 200)

    targets = p2.collect_targets(corpus)
    target_names = [t.name for t in targets]

    assert "real.md" in target_names
    assert "page.md" not in target_names


def test_build_prepare_2_prompt_de_domained_and_retrieval_oriented():
    from pipeline.prepare_2_summarize import build_prepare_2_prompt

    p = build_prepare_2_prompt("Transcripts of Waskom city council meetings.")
    assert "Texas RRC" not in p
    assert "LA-0304" not in p
    assert "Transcripts of Waskom city council meetings." in p
    assert "search index" in p.lower() or "retrieval" in p.lower()
    assert "speaker" in p.lower()
    assert "self-correct" in p.lower() or "contradict" in p.lower()
    assert "variant" in p.lower() or "spelling" in p.lower()
    assert "verbatim" in p.lower()
    assert "locations" in p.lower()
    assert "deadlines" in p.lower()


def test_build_prepare_2_prompt_empty_corpus_description_no_stray_text():
    from pipeline.prepare_2_summarize import build_prepare_2_prompt

    p = build_prepare_2_prompt("")
    assert "following collection" not in p


def test_run_prepare_2_uses_corpus_description(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    doc = corpus / "doc1.md"
    doc.write_text("This is a real document with enough words to summarize. " * 5)

    captured = {}

    def fake_call_text_llm(system, user, model, max_tokens=4096):
        captured["system"] = system
        return "**Summary:** ok\n\n**Facts:** none\n"

    monkeypatch.setattr(p2, "call_text_llm", fake_call_text_llm)
    p2.run_prepare_2(corpus_root=str(corpus), model="m", workers=1, force=False,
                     corpus_description="A test corpus about widgets.")

    assert "A test corpus about widgets." in captured["system"]


def test_collect_targets_includes_non_page_md_in_web_cache(tmp_path):
    """Files named other than page.md inside web_cache are still summarized."""
    corpus = tmp_path / "corpus"
    (corpus / "web_cache").mkdir(parents=True)
    # This is unlikely but should be handled — only "page.md" is skipped
    (corpus / "web_cache" / "notes.md").write_text("E" * 200)

    targets = p2.collect_targets(corpus)
    target_names = [t.name for t in targets]

    assert "notes.md" in target_names  # not skipped — not named page.md
