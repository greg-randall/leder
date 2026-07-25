"""Tests for Stage B: Claim verification via Claude Agent SDK."""
import pytest
import os
from pipeline.stage_b_verify import (
    AGENT_SYSTEM_PROMPT,
    agent_failure_result,
    parse_verdict,
    _populate_claim_from_dict,
    _summarize_web_cache,
    _write_web_cache_folder_summary,
    _check_corpus_ready,
    VerdictOutput,
)
from pipeline.models import Claim


def test_agent_system_prompt_contains_key_rules():
    assert "SEARCH STRATEGY" in AGENT_SYSTEM_PROMPT
    assert "Summaries are a map" in AGENT_SYSTEM_PROMPT
    assert "ORIGINALS (MANDATORY VERIFICATION STEP)" in AGENT_SYSTEM_PROMPT
    assert "OUTPUT" in AGENT_SYSTEM_PROMPT
    assert "structured data" in AGENT_SYSTEM_PROMPT


def test_structured_output_populates_claim():
    """Structured output from the SDK should populate all claim fields."""
    claim = Claim(
        claim_id="c0001",
        claim_text="Test claim.",
        source_quote="Test claim.",
        claim_type="numeric",
    )
    data = {
        "verdict": "supported",
        "source_proximity": "original",
        "source_path": "LA-0304/permits/2020.md",
        "source_url": None,
        "rationale": "The permit confirms 165 acres of irrigation.",
        "human_review": False,
        "confidence": 0.95,
    }
    result = _populate_claim_from_dict(claim, data)
    assert result.verdict == "supported"
    assert result.source_proximity == "original"
    assert result.source_path == "LA-0304/permits/2020.md"
    assert result.rationale == "The permit confirms 165 acres of irrigation."
    assert result.human_review is False
    assert result.confidence == 0.95


def test_structured_output_missing_field_falls_back():
    """Missing source_proximity defaults to 'original' (not a hard failure)."""
    claim = Claim(
        claim_id="c0002",
        claim_text="Test.",
        source_quote="Test.",
        claim_type="numeric",
    )
    result = _populate_claim_from_dict(
        claim,
        {"verdict": "supported", "rationale": "ok", "human_review": False, "confidence": 0.5},
    )
    assert result.verdict == "supported"
    assert result.source_proximity == "unverifiable"


def test_verdict_output_schema():
    """VerdictOutput schema should be valid JSON Schema."""
    schema = VerdictOutput.model_json_schema()
    assert schema["type"] == "object"
    assert "verdict" in schema["properties"]
    assert "source_proximity" in schema["properties"]
    # Enum values should be enforced
    assert "supported" in schema["properties"]["verdict"]["enum"]


def test_parse_verdict_valid_json():
    claim = Claim(
        claim_id="c0001",
        claim_text="Test claim.",
        source_quote="Test claim.",
        claim_type="numeric",
    )
    agent_output = (
        "I searched the corpus and found the relevant permit.\n\n"
        '{"verdict":"supported","source_proximity":"original",'
        '"source_path":"LA-0304/permits/2020.md","source_url":null,'
        '"rationale":"The permit confirms 165 acres of irrigation.",'
        '"human_review":false,"confidence":0.95}\n'
    )
    result = parse_verdict(claim, agent_output)
    assert result.verdict == "supported"
    assert result.source_proximity == "original"
    assert result.source_path == "LA-0304/permits/2020.md"
    assert result.rationale == "The permit confirms 165 acres of irrigation."
    assert result.human_review is False
    assert result.confidence == 0.95


def test_parse_verdict_multiline_json():
    claim = Claim(
        claim_id="c0002",
        claim_text="Another claim.",
        source_quote="Another claim.",
        claim_type="attribution",
    )
    agent_output = """Some preamble text.

{
  "verdict": "contradicted",
  "source_proximity": "derived",
  "source_path": "overview.md",
  "source_url": null,
  "rationale": "The overview says something different.",
  "human_review": true,
  "confidence": 0.5
}

Some trailing text."""
    result = parse_verdict(claim, agent_output)
    assert result.verdict == "contradicted"
    assert result.human_review is True
    assert result.confidence == 0.5


def test_parse_verdict_no_json():
    claim = Claim(
        claim_id="c0003",
        claim_text="No JSON here.",
        source_quote="No JSON here.",
        claim_type="generalization",
    )
    result = parse_verdict(claim, "Just some text without any JSON object.")
    assert result.verdict == "unsupported"
    assert result.source_proximity == "unverifiable"
    assert result.human_review is True


def test_parse_verdict_invalid_json():
    claim = Claim(
        claim_id="c0004",
        claim_text="Bad JSON.",
        source_quote="Bad JSON.",
        claim_type="legal",
    )
    result = parse_verdict(claim, '{"verdict": "supported", invalid}')
    assert result.verdict == "unsupported"
    assert result.human_review is True


def test_parse_verdict_missing_field():
    claim = Claim(
        claim_id="c0005",
        claim_text="Missing field.",
        source_quote="Missing field.",
        claim_type="numeric",
    )
    result = parse_verdict(claim, '{"verdict":"supported","rationale":"ok.","human_review":false,"confidence":0.9}')
    assert result.verdict == "unsupported"
    assert result.human_review is True


def test_parse_verdict_empty_text():
    claim = Claim(
        claim_id="c0006",
        claim_text="Empty.",
        source_quote="Empty.",
        claim_type="numeric",
    )
    result = parse_verdict(claim, "")
    assert result.verdict == "unsupported"
    assert result.human_review is True


def test_agent_failure_fallback():
    claim = Claim(
        claim_id="c9999",
        claim_text="Test claim for failure.",
        source_quote="Test claim for failure.",
        claim_type="generalization",
    )
    result = agent_failure_result(claim)
    assert result.verdict == "unsupported"
    assert result.source_proximity == "unverifiable"
    assert result.human_review is True
    assert result.rationale is not None


@pytest.mark.integration
def test_verify_claim_integration():
    """Integration test: requires Claude Code CLI and API key."""
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")):
        pytest.skip("No API key configured")
    corpus_root = "/mnt/e/Google Drive/rrc-permit-files/g-journalism-run/source-docs-and-summaries"
    if not os.path.exists(corpus_root):
        pytest.skip("Corpus not available")

    import asyncio
    from pipeline.stage_b_verify import _verify_claim_async

    claim = Claim(
        claim_id="c_test",
        claim_text="LA-0304 irrigates 165 acres in Karnes County via sprinkler.",
        source_quote="irrigates 165 acres in Karnes County via sprinkler",
        claim_type="numeric",
    )

    result = asyncio.run(_verify_claim_async(claim, corpus_root, AGENT_SYSTEM_PROMPT))
    assert result.verdict is not None
    assert result.rationale is not None
    assert len(result.rationale) > 5


def test_build_verification_prompt_injects_variables():
    from pipeline.playbook import Playbook
    from pipeline.stage_b_verify import _build_verification_prompt

    pb = Playbook(name="t", extraction_prompt="E",
                  verification_prompt="S:{{article_summary}} C:{{target_text}} X:{{context}}")
    r = _build_verification_prompt(pb, "Sum", "Claim", "Ctx")
    assert "Sum" in r and "Claim" in r and "Ctx" in r


def test_get_playbook_caches(tmp_path):
    pd = tmp_path / "p"
    pd.mkdir()
    (pd / "tc.yaml").write_text(
        "name: TC\nextraction:\n  prompt: E\n"
        "verification:\n  prompt: V\n  allowed_tools: [Read, WebSearch]"
    )
    from pipeline.stage_b_verify import _get_playbook, _PLAYBOOK_CACHE
    _PLAYBOOK_CACHE.clear()
    pb = _get_playbook("tc", str(pd))
    assert pb.name == "TC" and pb.allowed_tools == ["Read", "WebSearch"]
    assert _get_playbook("tc", str(pd)) is pb  # cached


# ── web_cache summarization ──────────────────────────────────────

def test_summarize_web_cache_writes_page_summary_md(tmp_path):
    """_summarize_web_cache writes page_summary.md, not _summary.md."""
    wc = tmp_path / "web_cache"
    claim_dir = wc / "claim-001"
    claim_dir.mkdir(parents=True)
    (claim_dir / "page.md").write_text(
        "First paragraph of fetched page.\n\nSecond paragraph with more detail.\n\nThird.",
        encoding="utf-8",
    )

    _summarize_web_cache(str(wc))

    # Should write page_summary.md (rollup-friendly name)
    ps = claim_dir / "page_summary.md"
    assert ps.exists(), f"Expected {ps} to exist"
    content = ps.read_text()
    assert "First paragraph" in content
    assert "Web-cached page" in content

    # Should NOT write the old _summary.md name
    old = claim_dir / "_summary.md"
    assert not old.exists(), f"{old} should NOT exist (old naming convention)"


def test_write_web_cache_folder_summary(tmp_path):
    """_write_web_cache_folder_summary generates _FOLDER_SUMMARY.md from page_summary.md files."""
    wc = tmp_path / "web_cache"
    for cid in ("claim-a", "claim-b"):
        d = wc / cid
        d.mkdir(parents=True)
        (d / "page_summary.md").write_text(
            f"**Summary:** Web-cached page. Preview for {cid}.\n\n"
            f"**Facts:** *(see page.md for full content)*\n",
            encoding="utf-8",
        )

    _write_web_cache_folder_summary(str(wc))

    fs = wc / "_FOLDER_SUMMARY.md"
    assert fs.exists(), f"Expected {fs} to exist"
    content = fs.read_text()
    assert "Web Cache" in content
    assert "2 cached page" in content
    assert "claim-a" in content
    assert "claim-b" in content
    assert "[open cached page](claim-a/page.md)" in content


# ── Corpus readiness check ───────────────────────────────────────

def test_check_corpus_ready_passes_when_prepared(tmp_path):
    """Corpus with _FOLDER_SUMMARY.md and web_cache with _FOLDER_SUMMARY.md → no issues."""
    cr = tmp_path / "corpus"
    cr.mkdir()
    (cr / "_FOLDER_SUMMARY.md").write_text("folder summary")
    # Also create a web_cache with folder summary
    wc = tmp_path / "web_cache"
    wc.mkdir()
    (wc / "_FOLDER_SUMMARY.md").write_text("wc summary")

    issues = _check_corpus_ready(str(cr), str(wc))
    assert issues == [], f"Expected no issues, got: {issues}"


def test_check_corpus_ready_passes_with_overview(tmp_path):
    """CORPUS_OVERVIEW.md alone satisfies the check (no _FOLDER_SUMMARY needed)."""
    cr = tmp_path / "corpus"
    cr.mkdir()
    (cr / "CORPUS_OVERVIEW.md").write_text("overview")

    issues = _check_corpus_ready(str(cr), "")
    assert issues == []


def test_check_corpus_ready_blocks_when_no_summaries(tmp_path):
    """Empty corpus without summaries → returns an issue."""
    cr = tmp_path / "corpus"
    cr.mkdir()

    issues = _check_corpus_ready(str(cr), "")
    assert len(issues) == 1
    assert "_folder_summary.md" in issues[0].lower()
    assert "prepare-2" in issues[0]
    assert "prepare-3" in issues[0]


def test_check_corpus_ready_blocks_when_web_cache_has_pages_but_no_folder_summary(tmp_path):
    """web_cache with cached pages but no _FOLDER_SUMMARY.md → returns an issue."""
    cr = tmp_path / "corpus"
    cr.mkdir()
    (cr / "_FOLDER_SUMMARY.md").write_text("folder summary")  # corpus is fine

    wc = tmp_path / "web_cache"
    wc.mkdir()
    (wc / "claim-x").mkdir()
    (wc / "claim-x" / "page.md").write_text("cached content")

    issues = _check_corpus_ready(str(cr), str(wc))
    assert len(issues) == 1
    assert "web cache" in issues[0].lower()
    assert "_FOLDER_SUMMARY.md" in issues[0]


def test_check_corpus_ready_web_cache_empty_ok(tmp_path):
    """Empty web_cache with no pages is fine — no issue."""
    cr = tmp_path / "corpus"
    cr.mkdir()
    (cr / "_FOLDER_SUMMARY.md").write_text("folder summary")
    wc = tmp_path / "web_cache"
    wc.mkdir()  # exists but empty — no page.md files

    issues = _check_corpus_ready(str(cr), str(wc))
    assert issues == []


# ── _summarize_web_cache edge cases ──────────────────────────────

def test_summarize_web_cache_noop_on_missing_dir(tmp_path):
    """Non-existent dir → no-op, no crash."""
    _summarize_web_cache(str(tmp_path / "nonexistent"))  # should not raise


def test_summarize_web_cache_noop_when_no_page_md(tmp_path):
    """web_cache exists but has no */page.md files → no-op."""
    wc = tmp_path / "web_cache"
    wc.mkdir()
    (wc / "empty-dir").mkdir()
    _summarize_web_cache(str(wc))
    assert not list(wc.rglob("page_summary.md"))
    assert not list(wc.rglob("_FOLDER_SUMMARY.md"))


def test_summarize_web_cache_handles_empty_page_md(tmp_path):
    """page.md with empty content → writes (empty) placeholder."""
    wc = tmp_path / "web_cache"
    d = wc / "claim-x"
    d.mkdir(parents=True)
    (d / "page.md").write_text("", encoding="utf-8")

    _summarize_web_cache(str(wc))

    ps = d / "page_summary.md"
    assert ps.exists()
    content = ps.read_text()
    assert "(empty)" in content


def test_summarize_web_cache_skips_existing_summary(tmp_path):
    """Already-existing page_summary.md > 10 bytes → skipped, not overwritten."""
    wc = tmp_path / "web_cache"
    d = wc / "claim-x"
    d.mkdir(parents=True)
    (d / "page.md").write_text("New first paragraph.\n\nMore text.", encoding="utf-8")
    (d / "page_summary.md").write_text("EXISTING SUMMARY THAT SHOULD SURVIVE", encoding="utf-8")

    _summarize_web_cache(str(wc))

    ps = d / "page_summary.md"
    assert "EXISTING SUMMARY THAT SHOULD SURVIVE" in ps.read_text()
    assert "New first paragraph" not in ps.read_text()


def test_summarize_web_cache_truncates_long_first_paragraph(tmp_path):
    """First paragraph > 300 chars → truncated."""
    wc = tmp_path / "web_cache"
    d = wc / "claim-x"
    d.mkdir(parents=True)
    long_para = "A" * 500
    (d / "page.md").write_text(long_para + "\n\nSecond paragraph.", encoding="utf-8")

    _summarize_web_cache(str(wc))

    ps = d / "page_summary.md"
    content = ps.read_text()
    assert "A" * 300 in content
    assert "A" * 301 not in content


def test_summarize_web_cache_many_claim_dirs(tmp_path):
    """All claim dirs with page.md get page_summary.md."""
    wc = tmp_path / "web_cache"
    for i in range(20):
        d = wc / f"claim-{i:03d}"
        d.mkdir(parents=True)
        (d / "page.md").write_text(f"Content for claim {i}.\n\nMore.", encoding="utf-8")

    _summarize_web_cache(str(wc))

    count = len(list(wc.glob("*/page_summary.md")))
    assert count == 20


def test_summarize_web_cache_single_paragraph_page(tmp_path):
    """page.md with no double-newline → entire text is the first paragraph."""
    wc = tmp_path / "web_cache"
    d = wc / "claim-x"
    d.mkdir(parents=True)
    (d / "page.md").write_text("Single line. Still single line. No paragraph break.", encoding="utf-8")

    _summarize_web_cache(str(wc))

    ps = d / "page_summary.md"
    content = ps.read_text()
    assert "Single line. Still single line." in content


# ── _write_web_cache_folder_summary edge cases ────────────────────

def test_write_web_cache_folder_summary_noop_on_missing_dir(tmp_path):
    """Non-existent dir → no-op, no crash."""
    _write_web_cache_folder_summary(str(tmp_path / "nonexistent"))


def test_write_web_cache_folder_summary_noop_when_no_summaries(tmp_path):
    """web_cache exists but no page_summary.md files → no-op."""
    wc = tmp_path / "web_cache"
    wc.mkdir()
    (wc / "empty-dir").mkdir()
    _write_web_cache_folder_summary(str(wc))
    assert not (wc / "_FOLDER_SUMMARY.md").exists()


# ── _check_corpus_ready edge cases ───────────────────────────────

def test_check_corpus_ready_both_broken(tmp_path):
    """Corpus missing summaries AND web_cache has orphaned pages → 2 issues."""
    cr = tmp_path / "corpus"
    cr.mkdir()  # no summaries
    wc = tmp_path / "web_cache"
    wc.mkdir()
    (wc / "claim-x").mkdir()
    (wc / "claim-x" / "page.md").write_text("cached")

    issues = _check_corpus_ready(str(cr), str(wc))
    assert len(issues) == 2
    assert any("corpus" in i.lower() for i in issues)
    assert any("web cache" in i.lower() for i in issues)


def test_check_corpus_ready_web_cache_dir_does_not_exist(tmp_path):
    """web_cache dir doesn't exist at all → no web_cache issue."""
    cr = tmp_path / "corpus"
    cr.mkdir()
    (cr / "_FOLDER_SUMMARY.md").write_text("ready")

    issues = _check_corpus_ready(str(cr), str(tmp_path / "nonexistent"))
    assert issues == []


def test_check_corpus_ready_empty_web_cache_dir_string(tmp_path):
    """Empty string for web_cache_dir → doesn't find pages, no issue."""
    cr = tmp_path / "corpus"
    cr.mkdir()
    (cr / "_FOLDER_SUMMARY.md").write_text("ready")

    issues = _check_corpus_ready(str(cr), "")
    assert issues == []


def test_check_corpus_ready_finds_deep_folder_summary(tmp_path):
    """_FOLDER_SUMMARY.md in a subdirectory is found by rglob."""
    cr = tmp_path / "corpus"
    cr.mkdir()
    (cr / "subdir").mkdir()
    (cr / "subdir" / "_FOLDER_SUMMARY.md").write_text("deep summary")

    issues = _check_corpus_ready(str(cr), "")
    assert issues == [], f"rglob should find nested _FOLDER_SUMMARY.md, got: {issues}"


# ── Agent prompt contains web_cache instructions ──────────────────

def test_agent_prompt_mentions_web_cache():
    """System prompt tells agents to check web_cache before fetching."""
    assert "web_cache/_FOLDER_SUMMARY.md" in AGENT_SYSTEM_PROMPT
    assert "--cache-dir web_cache" in AGENT_SYSTEM_PROMPT
    cache_hint = AGENT_SYSTEM_PROMPT.lower()
    assert "already cached" in cache_hint or "already been cached" in cache_hint


# ── Near-duplicate target dedup + finding fan-out ─────────────────

def test_run_stage_b_dedups_near_duplicate_targets_and_fans_out(tmp_path, monkeypatch):
    """Two near-duplicate targets get ONE agent call but TWO findings."""
    import json as _json
    from pipeline import stage_b_verify as sb

    targets_path = tmp_path / "targets.json"
    targets_path.write_text(_json.dumps({
        "article_file": "article.md",
        "article_summary": "S",
        "targets": [
            {"playbook": "fact_check", "target_text": "20000 barrels a day were injected.",
             "anchor_text": "20000 barrels a day (timeline)", "context": "ctx-a"},
            {"playbook": "fact_check", "target_text": "20000 barrels a day were injected!",  # near-dup
             "anchor_text": "20000 barrels a day (key facts)", "context": "ctx-b"},
        ],
    }))

    pd = tmp_path / "playbooks"
    pd.mkdir()
    (pd / "fact_check.yaml").write_text(
        "name: Fact Check\n"
        "extraction:\n  prompt: E\n"
        "verification:\n  prompt: V\n"
        "  allowed_tools: [Read]\n"
    )

    output_path = str(tmp_path / "findings.json")
    calls = []

    async def fake_verify_target_async(
        target, prompt, allowed_tools, corpus_root,
        timeout, max_turns, debug_dir, article_summary,
    ):
        from pipeline.finding import Finding, Severity
        calls.append(target["anchor_text"])
        return Finding(
            finding_id="fact_check-rep",
            check_type="fact_check",
            severity=Severity.PASS,
            target_text=target["target_text"],
            anchor_text=target["anchor_text"],
            context=target.get("context", ""),
            agent_summary="ok",
        )

    monkeypatch.setattr(sb, "_verify_target_async", fake_verify_target_async)
    monkeypatch.setattr(sb, "_check_corpus_ready", lambda *a, **k: [])

    sb.run_stage_b(
        output_path=output_path, corpus_root=str(tmp_path),
        targets_path=str(targets_path), playbook_dir=str(pd),
        concurrency=4, force_run=True,
    )

    # Only ONE agent call for the two near-duplicate targets.
    assert len(calls) == 1

    data = _json.loads(open(output_path).read())
    findings = data["findings"]
    assert len(findings) == 2
    anchors = {f["anchor_text"] for f in findings}
    assert anchors == {"20000 barrels a day (timeline)", "20000 barrels a day (key facts)"}
    finding_ids = {f["finding_id"] for f in findings}
    assert "fact_check-rep" in finding_ids
    assert any(fid.endswith("-a1") for fid in finding_ids)
