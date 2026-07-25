"""Tests for Stage A: Claim extraction."""
import json
import os
import tempfile

import pytest

from pipeline.stage_a_extract import (
    _chunk_article,
    build_extraction_prompt,
    build_quality_gate_prompt,
    extract_claims,
    run_stage_a,
)

SAMPLE_ARTICLE = """# Test Article

LA-0304 was originally permitted in 2001 to Koch Midstream Services.
The facility irrigates 165 acres in Karnes County using sprinklers.
Produced water chlorides range from 80 to 200 mg/L at this site.

This is an opinion statement that should be skipped.

SB 1145 transfers jurisdiction from RRC to TCEQ.
"""


def test_build_extraction_prompt():
    prompt = build_extraction_prompt(SAMPLE_ARTICLE)
    assert "LA-0304" in prompt
    assert "STANDALONE" not in prompt  # It's in system prompt, not user prompt


def test_build_quality_gate_prompt():
    existing = ["LA-0304 was originally permitted in 2001 to Koch Midstream Services."]
    prompt = build_quality_gate_prompt(SAMPLE_ARTICLE, existing)
    assert "Koch Midstream" in prompt
    assert "MISSED" in prompt


@pytest.mark.integration
def test_extract_claims_integration():
    """Integration test: requires Anthropic API key in env."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    doc = extract_claims(SAMPLE_ARTICLE, model="claude-sonnet-5", quality_gate=False)
    assert len(doc.claims) >= 3
    assert all(c.claim_id for c in doc.claims)
    assert all(c.source_quote for c in doc.claims)
    for c in doc.claims:
        assert len(c.claim_text) > 20, f"Claim too short/generic: {c.claim_text}"


@pytest.mark.integration
def test_run_stage_a_writes_file():
    """Integration test: writes claims.json."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    with tempfile.TemporaryDirectory() as tmp:
        article_path = os.path.join(tmp, "article.md")
        with open(article_path, "w") as f:
            f.write(SAMPLE_ARTICLE)
        output_path = os.path.join(tmp, "claims.json")
        doc = run_stage_a(
            article_path=article_path,
            output_path=output_path,
            corpus_root=tmp,
            project_name="test",
            model="claude-sonnet-5",
            quality_gate=False,
        )
        assert os.path.exists(output_path)
        loaded = json.loads(open(output_path).read())
        assert len(loaded["claims"]) >= 3


def test_missing_article_raises():
    with pytest.raises(FileNotFoundError):
        run_stage_a("/nonexistent/article.md", "/tmp/out.json", "/tmp", "test")


def test_empty_article_raises():
    with tempfile.TemporaryDirectory() as tmp:
        article_path = os.path.join(tmp, "empty.md")
        with open(article_path, "w") as f:
            f.write("")
        with pytest.raises(ValueError):
            run_stage_a(article_path, os.path.join(tmp, "out.json"), tmp, "test")


def test_extract_targets_from_text_uses_shared_system_prompt(monkeypatch):
    from pipeline.playbook import Playbook
    from pipeline.stage_a_extract import _extract_targets_from_text, _extraction_tool_for

    pb = Playbook(
        name="test_check",
        extraction_prompt="Extract things from: {{article_text}}",
        verification_prompt="Verify.",
    )

    class FakeText:
        type = "tool_use"
        input = {"targets": [{"target_text": "t1", "anchor_text": "a1"}],
                 "article_title": "Title", "article_summary": "Summary"}

    captured = {}

    def fake_create(self, **kw):
        captured.update(kw)
        return type("r", (), {"content": [FakeText()]})()

    fake_client = type("c", (), {
        "messages": type("m", (), {"create": fake_create})()
    })()
    monkeypatch.setattr("anthropic.Anthropic", lambda **kw: fake_client)

    targets, title, summary = _extract_targets_from_text(
        "some article text", "m", pb, _extraction_tool_for(pb),
        system_prompt="SHARED SYSTEM PROMPT TEXT")

    assert len(targets) == 1
    assert targets[0].target_text == "t1"
    assert targets[0].playbook == "test_check"
    assert title == "Title"
    assert summary == "Summary"
    # The fix: system is the shared prompt, not the raw playbook template
    assert captured["system"] == "SHARED SYSTEM PROMPT TEXT"
    assert "{{article_text}}" not in captured["system"]
    # The user message still has the substituted article text
    assert "some article text" in captured["messages"][0]["content"]
    assert "{{article_text}}" not in captured["messages"][0]["content"]


def test_run_stage_a_playbook_path_writes_targets_json(tmp_path, monkeypatch):
    """run_stage_a with playbook_names writes targets.json tagged with playbook."""
    article = tmp_path / "article.md"
    article.write_text(
        "Some article text about a very important topic. "
        "Another sentence with more context. "
        "Yet another sentence to reach the minimum word threshold required."
    )
    output = str(tmp_path / "targets.json")

    # Build a minimal playbook in a temp dir
    pd = tmp_path / "playbooks"
    pd.mkdir()
    (pd / "test_check.yaml").write_text("""
        name: Test Check
        extraction:
          prompt: "Extract from: {{article_text}}"
        verification:
          prompt: Verify.
    """)

    # Mock the LLM so no real API call
    class FakeTargetTool:
        type = "tool_use"
        input = {"targets": [{"target_text": "t1", "anchor_text": "a1"}],
                 "article_title": "T", "article_summary": "S"}

    fake = type("c", (), {
        "messages": type("m", (), {
            "create": lambda self, **kw: type("r", (), {"content": [FakeTargetTool()]})()
        })()
    })()
    monkeypatch.setattr("anthropic.Anthropic", lambda **kw: fake)

    from pipeline.stage_a_extract import run_stage_a
    run_stage_a(
        article_path=str(article),
        output_path=output,
        corpus_root="",
        project_name="",
        model="m",
        quality_gate=False,
        playbook_dir=str(pd),
        playbook_names=["test_check"],
    )

    import json
    data = json.loads(open(output).read())
    assert data["article_file"] == str(article)
    assert len(data["targets"]) == 1
    assert data["targets"][0]["playbook"] == "test_check"  # slug, not display name
    assert data["targets"][0]["target_text"] == "t1"


def test_chunk_article_merges_small_chunks():
    text = "\n\n".join([f"Paragraph {i} " + "word " * 50 for i in range(3)])
    chunks = _chunk_article(text, target_words=100, max_words=1000)
    assert len(chunks) == 1


def test_chunk_article_oversized_splits_on_sentences():
    long_sentence_block = " ".join(
        [f"Sentence number {i} has some words in it." for i in range(1, 200)]
    )
    chunks = _chunk_article(long_sentence_block, target_words=50, max_words=100)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 100 for c in chunks)


def test_chunk_article_still_oversized_hard_splits():
    giant_single_sentence = "word " * 500
    chunks = _chunk_article(giant_single_sentence, target_words=50, max_words=100)
    assert len(chunks) >= 5
    assert all(len(c.split()) <= 100 for c in chunks)


def test_chunk_article_short_trailing_paragraph_not_dropped():
    """A short (<10-word) standalone trailing paragraph must survive --
    proving the old `>= 10` drop-filter silently lost content like this."""
    # sized close to max_words so it won't get pre-merged with the next paragraph
    para = ("word " * 28).strip()
    text = para + "\n\n" + "Short pull quote."  # 3 words

    chunks = _chunk_article(text, target_words=15, max_words=30)

    assert any("Short pull quote." in c for c in chunks)


def test_chunk_article_merges_tiny_trailing_fragment_when_it_fits():
    """Positive case: step 5's merge specifically fires and produces one
    combined chunk -- isolated from step 2's ordinary paragraph merge by
    using a single oversized, punctuation-free paragraph (so it can only be
    reduced by step 4's hard-split, never touched by step 2 or step 3).

    Before any splitting, big (57 words) + tiny (3 words) = 60 words, which
    is over max_words=30, so step 2 refuses to merge them at the paragraph
    level and keeps them as separate chunks. Step 4's hard-split then
    breaks the 57-word blob into a 30-word chunk and a 27-word remainder.
    Only then, in step 5, does the 27-word remainder have enough headroom
    (27 + 3 = 30 <= max_words) to absorb the tiny trailing fragment -- a
    merge that step 2 could never have performed on the unsplit paragraph.
    """
    big = ("word " * 57).strip()  # no punctuation -> forces hard-split, not sentence-split
    text = big + "\n\n" + "Short pull quote."  # 3 words

    chunks = _chunk_article(text, target_words=10, max_words=30)

    assert len(chunks) == 2
    assert chunks[0] == "word " * 29 + "word"  # first hard-split chunk, untouched by merge
    assert len(chunks[0].split()) == 30
    assert chunks[-1].endswith("Short pull quote.")
    assert chunks[-1] != "Short pull quote."  # proves it was merged, not left standalone
    assert len(chunks[-1].split()) == 30  # 27-word remainder + 3-word fragment


def test_chunk_context_brief_prepended_to_later_chunks(tmp_path, monkeypatch):
    """Chunk 1's article_summary becomes the context brief for chunk 2+.

    Paragraphs are sized at 600 words each (combined 1200 > max_words=1000,
    so _chunk_article's step 2 refuses to merge them; each is individually
    under max_words and over target_words=300, so step 2b's forward-merge
    doesn't touch them either) -- this reliably yields exactly 2 chunks
    under the default target_words=300, max_words=1000.
    """
    article = tmp_path / "article.md"
    article.write_text(
        ("First paragraph. " * 300).strip() + "\n\n" + ("Second paragraph. " * 300).strip()
    )
    output = str(tmp_path / "targets.json")

    pd = tmp_path / "playbooks"
    pd.mkdir()
    (pd / "test_check.yaml").write_text(
        "name: Test Check\n"
        "extraction:\n  prompt: \"Extract from: {{article_text}}\"\n"
        "verification:\n  prompt: Verify.\n"
    )

    calls = []

    def fake_create(self, **kw):
        calls.append(kw)
        n = len(calls)
        return type("r", (), {"content": [type("b", (), {
            "type": "tool_use",
            "input": {
                "targets": [{"target_text": f"t{n}", "anchor_text": f"a{n}"}],
                "article_title": "" if n == 1 else f"Title from chunk {n}",
                "article_summary": f"Summary from chunk {n}",
            },
        })()]})()

    fake = type("c", (), {"messages": type("m", (), {"create": fake_create})()})()
    monkeypatch.setattr("anthropic.Anthropic", lambda **kw: fake)

    from pipeline.stage_a_extract import run_stage_a
    run_stage_a(
        article_path=str(article), output_path=output, corpus_root="", project_name="",
        model="m", quality_gate=False, playbook_dir=str(pd), playbook_names=["test_check"],
    )

    assert len(calls) == 2
    # Chunk 2's user message contains chunk 1's summary as a brief.
    chunk2_user_msg = calls[1]["messages"][0]["content"]
    assert "This chunk is from an article about: Summary from chunk 1" in chunk2_user_msg
    # Chunk 1's user message has no brief prepended (nothing precedes it yet).
    chunk1_user_msg = calls[0]["messages"][0]["content"]
    assert "This chunk is from an article about" not in chunk1_user_msg

    data = json.loads(open(output).read())
    # First non-empty title wins: chunk 1 returned "", chunk 2 returned a title.
    assert data["article_title"] == "Title from chunk 2"
    # First non-empty summary wins: chunk 1's summary, not chunk 2's.
    assert data["article_summary"] == "Summary from chunk 1"


def test_chunk_article_tiny_first_chunk_has_nothing_to_merge_into():
    """A tiny fragment as the very first (and only) chunk has no previous
    chunk to merge into -- must survive standalone, not crash on
    final_chunks[-1] (the empty-guard branch of step 5)."""
    text = "Short pull quote."  # 3 words, alone

    chunks = _chunk_article(text, target_words=15, max_words=30)

    assert chunks == ["Short pull quote."]
