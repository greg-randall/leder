"""Tests for Stage B: Claim verification via Claude Agent SDK."""
import asyncio
import pytest
import os
from pipeline.stage_b_verify import (
    agent_failure_result,
    parse_verdict,
    _populate_claim_from_dict,
    _summarize_web_cache,
    _write_web_cache_folder_summary,
    _check_corpus_ready,
)
from pipeline.models import Claim


def test_structured_output_populates_claim():
    """Structured output from the SDK should populate all claim fields."""
    claim = Claim(
        claim_id="c0001",
        claim_text="Test claim.",
        source_quote="Test claim.",
        claim_type="numeric",
    )
    data = {
        "severity": "PASS",
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
        {"severity": "PASS", "rationale": "ok", "human_review": False, "confidence": 0.5},
    )
    assert result.verdict == "supported"
    assert result.source_proximity == "unverifiable"


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
    # 0.5 is equidistant between 0.4/0.6 bands; snaps to 0.6 per
    # _CONFIDENCE_BANDS iteration order (0.6 appears before 0.4, and
    # min() keeps the first minimum encountered on ties).
    assert result.confidence == 0.6


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
    from pipeline.prompts import build_verification_rules_block
    from pipeline.stage_b_verify import _verify_claim_async

    claim = Claim(
        claim_id="c_test",
        claim_text="LA-0304 irrigates 165 acres in Karnes County via sprinkler.",
        source_quote="irrigates 165 acres in Karnes County via sprinkler",
        claim_type="numeric",
    )

    result = asyncio.run(_verify_claim_async(
        claim, corpus_root, build_verification_rules_block(""),
    ))
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


def test_build_verification_prompt_prepends_shared_rules_block():
    from pipeline.playbook import Playbook
    from pipeline.stage_b_verify import _build_verification_prompt

    pb = Playbook(name="t", extraction_prompt="E",
                  verification_prompt="S:{{article_summary}} C:{{target_text}} X:{{context}}")
    r = _build_verification_prompt(pb, "Sum", "Claim", "Ctx", corpus_description="A test corpus.")
    assert "SANDBOX" in r  # from the shared rules block
    assert "A test corpus." in r
    assert "Sum" in r and "Claim" in r and "Ctx" in r
    # Shared rules must come before the check-specific prompt
    assert r.index("SANDBOX") < r.index("S:Sum")


def test_fact_check_yaml_verification_prompt_has_two_level_verdict():
    from pipeline.playbook import load_playbook
    pb = load_playbook("pipelines/fact_check.yaml")
    prompt = pb.verification_prompt
    assert "attribution accuracy" in prompt.lower()
    assert "independent corroboration" in prompt.lower()
    assert "attribution_status" in prompt


def test_fact_check_yaml_quality_gate_has_near_duplicate_rule():
    from pipeline.playbook import load_playbook
    pb = load_playbook("pipelines/fact_check.yaml")
    assert "semantic equivalent" in pb.quality_gate_prompt.lower()


def test_fact_check_yaml_has_severity_human_review_trigger():
    from pipeline.playbook import load_playbook
    pb = load_playbook("pipelines/fact_check.yaml")
    prompt = pb.verification_prompt
    assert "WARNING or CRITICAL" in prompt


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


# ── Near-duplicate target dedup + finding fan-out ─────────────────

def test_normalize_target_text_strips_only_one_trailing_punct():
    from pipeline.stage_b_verify import _normalize_target_text
    assert _normalize_target_text("Really?!") == "really?"
    assert _normalize_target_text("Wow!!!") == "wow!!"
    assert _normalize_target_text("Plain sentence.") == "plain sentence"
    assert _normalize_target_text("  Extra   spaces  ") == "extra spaces"


def test_run_stage_b_requires_targets_path(tmp_path, monkeypatch):
    monkeypatch.setattr("pipeline.stage_b_verify._check_corpus_ready", lambda *a, **k: [])
    from pipeline.stage_b_verify import run_stage_b
    with pytest.raises(ValueError, match="targets_path"):
        run_stage_b(targets_path="", output_path=str(tmp_path / "out.json"),
                    corpus_root=str(tmp_path), force_run=True)


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
        agent_log_dir=None, web_cache_dir="web_cache",
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


def test_verify_target_async_finding_id_no_collision_on_equal_length(monkeypatch):
    """Two different-text targets of EQUAL length must not produce the same finding_id."""
    from pipeline import stage_b_verify as sb

    async def fake_verify_claim_async(
        claim, corpus_root, system_prompt, timeout, max_turns,
        debug_dir, agent_log_dir=None, article_summary="", allowed_tools=None,
        output_schema=None, web_cache_dir="web_cache",
    ):
        claim.verdict = "supported"
        claim.rationale = "ok"
        claim.source_path = None
        claim.source_url = None
        claim.source_excerpt = ""
        claim.confidence = 0.8
        claim.human_review = False
        claim._raw_output = {}
        return claim

    monkeypatch.setattr(sb, "_verify_claim_async", fake_verify_claim_async)

    target_a = {"playbook": "fact_check", "target_text": "AAAAAAAAAA", "anchor_text": "a"}
    target_b = {"playbook": "fact_check", "target_text": "BBBBBBBBBB", "anchor_text": "b"}
    assert len(target_a["target_text"]) == len(target_b["target_text"])

    finding_a = asyncio.run(sb._verify_target_async(
        target_a, "sys", ["Read"], "/corpus", 60, 10, None, ""))
    finding_b = asyncio.run(sb._verify_target_async(
        target_b, "sys", ["Read"], "/corpus", 60, 10, None, ""))

    assert finding_a.finding_id != finding_b.finding_id


# ── Confidence snapping + forced human_review ─────────────────────

def test_snap_confidence_to_nearest_band():
    from pipeline.stage_b_verify import _snap_confidence
    assert _snap_confidence(0.95) == 0.95
    assert _snap_confidence(0.9) == 0.95  # closer to 0.95 than 0.8
    assert _snap_confidence(0.7) == 0.8   # closer to 0.8 than 0.6
    assert _snap_confidence(0.0) == 0.2
    assert _snap_confidence(1.0) == 0.95
    assert _snap_confidence(None) is None


def test_populate_claim_from_dict_snaps_confidence():
    from pipeline.stage_b_verify import _populate_claim_from_dict

    claim = Claim(claim_id="c1", claim_text="T", source_quote="T", claim_type="numeric")
    result = _populate_claim_from_dict(claim, {
        "verdict": "supported", "source_proximity": "original",
        "rationale": "ok", "human_review": False, "confidence": 0.91,
    })
    assert result.confidence == 0.95


def test_populate_claim_from_dict_forces_human_review_below_threshold():
    from pipeline.stage_b_verify import _populate_claim_from_dict

    claim = Claim(claim_id="c2", claim_text="T", source_quote="T", claim_type="numeric")
    result = _populate_claim_from_dict(claim, {
        "verdict": "supported", "source_proximity": "original",
        "rationale": "ok", "human_review": False, "confidence": 0.55,
    })
    assert result.confidence == 0.6
    assert result.human_review is True  # forced despite agent saying False


def test_populate_claim_from_dict_leaves_human_review_alone_above_threshold():
    from pipeline.stage_b_verify import _populate_claim_from_dict

    claim = Claim(claim_id="c3", claim_text="T", source_quote="T", claim_type="numeric")
    result = _populate_claim_from_dict(claim, {
        "verdict": "supported", "source_proximity": "original",
        "rationale": "ok", "human_review": False, "confidence": 0.85,
    })
    assert result.confidence == 0.8
    assert result.human_review is False


def test_verify_target_async_flags_summary_source(monkeypatch):
    from pipeline import stage_b_verify as sb

    async def fake_verify_claim_async(
        claim, corpus_root, system_prompt, timeout, max_turns,
        debug_dir, agent_log_dir=None, article_summary="", allowed_tools=None,
        output_schema=None, web_cache_dir="web_cache",
    ):
        claim.verdict = "supported"
        claim.rationale = "The overview mentions this."
        claim.source_path = "ALL_SUMMARIES.md"
        claim.source_url = None
        claim.source_excerpt = ""
        claim.confidence = 0.8
        claim.human_review = False
        claim._raw_output = {}
        return claim

    monkeypatch.setattr(sb, "_verify_claim_async", fake_verify_claim_async)

    target = {"playbook": "fact_check", "target_text": "Some claim.", "anchor_text": "a"}
    finding = asyncio.run(sb._verify_target_async(
        target, "sys", ["Read"], "/corpus", 60, 10, None, ""))

    assert finding.human_review is True
    assert "cites a summary" in finding.agent_summary


# ── Schema-aware boilerplate + absolute fetch_page.py invocation ──────────

def test_verify_claim_async_user_prompt_matches_output_schema(monkeypatch):
    """The boilerplate 'Output fields' line must match the schema actually in use,
    not a hardcoded field list from a different schema."""
    from pipeline import stage_b_verify as sb

    captured_prompts = []

    def _fake_query(prompt, options):
        async def _gen():
            async for msg in prompt:
                if isinstance(msg, dict) and "message" in msg:
                    captured_prompts.append(msg["message"]["content"])
            return
            yield  # pragma: no cover -- makes this an async generator
        return _gen()

    monkeypatch.setattr("claude_agent_sdk.query", _fake_query)

    claim = Claim(claim_id="c1", claim_text="T", source_quote="T", claim_type="numeric")
    asyncio.run(sb._verify_claim_async(
        claim, "/corpus", "sys", timeout=5, max_turns=1,
        output_schema=sb.FindingOutput.model_json_schema(),
    ))

    assert captured_prompts, "query() was never called"
    prompt = captured_prompts[0]
    for field in ("severity", "agent_summary", "recommended_action"):
        assert field in prompt
    # Old-schema-only fields must NOT appear as a stray "Output fields" list entry
    assert "Output fields: verdict, source_proximity" not in prompt


# ── FindingOutput source_excerpt_offset / source_excerpt_similarity ──

def test_finding_output_defaults_new_fields_to_none():
    """Both source_excerpt_offset and source_excerpt_similarity should default to None."""
    from pipeline.stage_b_verify import FindingOutput
    f = FindingOutput(severity="PASS", agent_summary="test")
    assert f.source_excerpt_offset is None
    assert f.source_excerpt_similarity is None


# ── excerpt gate ──────────────────────────────────────────────────────────

def _gate(tmp_path, raw, claim_id="c1", doc_text=None, doc_name="doc.md"):
    from pipeline.stage_b_verify import _apply_excerpt_gate
    corpus = tmp_path / "corpus"
    corpus.mkdir(exist_ok=True)
    wc = corpus / "web_cache"
    wc.mkdir(exist_ok=True)
    if doc_text is not None:
        (corpus / doc_name).write_text(doc_text)
    return _apply_excerpt_gate(raw, str(corpus), str(wc), claim_id)


def test_excerpt_gate_exact_records_offset(tmp_path):
    doc = "The road commission issued a 68-page notice of violation."
    out = _gate(tmp_path,
                {"source_path": "doc.md", "source_excerpt": "road commission issued"},
                doc_text=doc)
    assert out["excerpt_status"] == "exact"
    assert out["source_excerpt"] == "road commission issued"
    start, end = out["source_excerpt_offset"]
    assert doc[start:end] == "road commission issued"
    assert out["source_excerpt_similarity"] == 1.0
    assert "human_review" not in out


def test_excerpt_gate_repairs_paraphrase(tmp_path):
    doc = ("the road commission issued a 68 page notice of violation for the "
           "McBride Was facility on fe February 7th, 2025.")
    out = _gate(tmp_path,
                {"source_path": "doc.md",
                 "source_excerpt": "the Railroad Commission issued a 68-page notice of violation"},
                doc_text=doc)
    assert out["excerpt_status"] == "repaired"
    assert out["human_review"] is True
    # The excerpt is REPLACED with text that really is in the document
    assert out["source_excerpt"] in doc
    assert out["source_excerpt"] != "the Railroad Commission issued a 68-page notice of violation"
    start, end = out["source_excerpt_offset"]
    assert doc[start:end] == out["source_excerpt"]
    assert out["source_excerpt_similarity"] < 1.0


def test_excerpt_gate_not_found_drops_excerpt(tmp_path):
    out = _gate(tmp_path,
                {"source_path": "doc.md",
                 "source_excerpt": "a quote that is nowhere in this document at all"},
                doc_text="completely unrelated text about zoning permits")
    assert out["excerpt_status"] == "not_found"
    assert out["source_excerpt"] is None
    assert out["source_excerpt_offset"] is None
    assert out["human_review"] is True
    assert out.get("note")


def test_excerpt_gate_unchecked_when_no_source_file(tmp_path):
    out = _gate(tmp_path,
                {"source_path": "missing.md", "source_excerpt": "anything"})
    assert out == {"excerpt_status": "unchecked"}


def test_excerpt_gate_unchecked_when_no_excerpt(tmp_path):
    out = _gate(tmp_path, {"source_path": "doc.md", "source_excerpt": ""},
                doc_text="some text")
    assert out == {"excerpt_status": "unchecked"}


def test_excerpt_gate_unchecked_when_no_source_at_all(tmp_path):
    """Neither source_path nor source_url -- nothing to check against."""
    out = _gate(tmp_path, {"source_excerpt": "some quote"})
    assert out == {"excerpt_status": "unchecked"}


def test_excerpt_gate_strips_corpus_prefix(tmp_path):
    doc = "The road commission issued a notice."
    out = _gate(tmp_path,
                {"source_path": "corpus/doc.md", "source_excerpt": "road commission issued"},
                doc_text=doc)
    assert out["excerpt_status"] == "exact"


def test_excerpt_gate_strips_dot_slash_prefix(tmp_path):
    doc = "The road commission issued a notice."
    out = _gate(tmp_path,
                {"source_path": "./doc.md", "source_excerpt": "road commission issued"},
                doc_text=doc)
    assert out["excerpt_status"] == "exact"


def test_excerpt_gate_resolves_web_finding_to_cache(tmp_path):
    from pipeline.stage_b_verify import _apply_excerpt_gate
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    wc = corpus / "web_cache"
    (wc / "c9").mkdir(parents=True)
    (wc / "c9" / "page.md").write_text("The commission denied the renewal in February.")
    out = _apply_excerpt_gate(
        {"source_url": "https://x.test/a", "source_excerpt": "denied the renewal"},
        str(corpus), str(wc), "c9")
    assert out["excerpt_status"] == "exact"
    assert out["source_excerpt"] == "denied the renewal"


def test_excerpt_gate_rejects_path_escape(tmp_path):
    """An escaping source_path must be treated as unchecked, never read."""
    from pipeline.stage_b_verify import _apply_excerpt_gate
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "web_cache").mkdir()
    (tmp_path / "secret.md").write_text("road commission issued a notice")
    out = _apply_excerpt_gate(
        {"source_path": "../secret.md", "source_excerpt": "road commission issued"},
        str(corpus), str(corpus / "web_cache"), "c1")
    assert out == {"excerpt_status": "unchecked"}


def test_excerpt_gate_does_not_touch_severity_or_confidence(tmp_path):
    """The gate flags for human review; it must never rewrite the verdict."""
    out = _gate(tmp_path,
                {"source_path": "doc.md", "severity": "CRITICAL", "confidence": 0.95,
                 "source_excerpt": "nowhere in this text at all"},
                doc_text="completely unrelated content about zoning")
    assert "severity" not in out
    assert "confidence" not in out


def test_excerpt_gate_punctuation_only_diff_is_repaired_not_exact(tmp_path):
    """A candidate that differs from the source only by punctuation _clean()
    strips (apostrophe, comma) scores similarity 1.0 through validate_excerpt's
    FUZZY tier, not its literal tier -- excerpt_status must reflect that it
    was normalised, not quoted verbatim, and must flag for human review."""
    doc = "The commission said its a routine matter, nothing more."
    out = _gate(tmp_path,
                {"source_path": "doc.md",
                 "source_excerpt": "it's a routine matter nothing more"},
                doc_text=doc)
    assert out["excerpt_status"] == "repaired"
    assert out["human_review"] is True
    assert out["source_excerpt"] == "its a routine matter, nothing more"
    assert out["source_excerpt"] in doc


def test_excerpt_gate_genuine_literal_match_stays_exact(tmp_path):
    """Guard against the literal-vs-normalised rule drifting the other way:
    a true verbatim (case-insensitive) match must still be 'exact' with no
    human_review."""
    doc = "The road commission issued a 68-page notice of violation."
    out = _gate(tmp_path,
                {"source_path": "doc.md", "source_excerpt": "ROAD COMMISSION issued"},
                doc_text=doc)
    assert out["excerpt_status"] == "exact"
    assert out["source_excerpt"] == "road commission issued"
    assert "human_review" not in out


def test_excerpt_gate_strips_repeated_dot_slash_corpus_prefix(tmp_path):
    """'./corpus/doc.md' must resolve, not silently fall through to unchecked."""
    doc = "The road commission issued a notice."
    out = _gate(tmp_path,
                {"source_path": "./corpus/doc.md", "source_excerpt": "road commission issued"},
                doc_text=doc)
    assert out["excerpt_status"] == "exact"


def test_excerpt_gate_strips_corpus_dot_slash_prefix(tmp_path):
    """'corpus/./doc.md' must also resolve."""
    doc = "The road commission issued a notice."
    out = _gate(tmp_path,
                {"source_path": "corpus/./doc.md", "source_excerpt": "road commission issued"},
                doc_text=doc)
    assert out["excerpt_status"] == "exact"


def test_excerpt_gate_escape_logs_to_stderr(tmp_path, capsys):
    """An escaping source_path should still print a visible warning, even
    though excerpt_status stays 'unchecked' rather than a distinct status."""
    from pipeline.stage_b_verify import _apply_excerpt_gate
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "web_cache").mkdir()
    (tmp_path / "secret.md").write_text("road commission issued a notice")
    out = _apply_excerpt_gate(
        {"source_path": "../secret.md", "source_excerpt": "road commission issued"},
        str(corpus), str(corpus / "web_cache"), "c1")
    assert out == {"excerpt_status": "unchecked"}
    captured = capsys.readouterr()
    assert "escapes corpus" in captured.err


def test_excerpt_gate_non_string_excerpt_does_not_crash(tmp_path):
    """A non-string source_excerpt (int, list) must not raise AttributeError on
    .strip(). The JSON schema constrains this, but the gate should survive
    malformed input rather than crashing the entire pipeline."""
    for bogus in (42, ["a", "b"], None):
        out = _gate(tmp_path,
                    {"source_path": "doc.md", "source_excerpt": bogus},
                    doc_text="the road commission issued a notice")
        assert out["excerpt_status"] in ("unchecked", "not_found")


# ── Wired excerpt gate + permission callback tests ─────────────────────

def _fake_agent(monkeypatch, structured_output: dict):
    """Patch claude_agent_sdk.query to return a single ResultMessage with
    the given structured_output. Uses MagicMock(spec=ResultMessage) so that
    isinstance(msg, ResultMessage) works without constructing a real SDK object."""
    from unittest.mock import MagicMock
    from claude_agent_sdk import ResultMessage
    msg = MagicMock(spec=ResultMessage)
    msg.structured_output = structured_output

    def _fake_query(prompt, options):
        # Regular function (not async) that returns an async generator, matching
        # the real SDK's query() signature.
        async def _gen():
            yield msg
        return _gen()

    monkeypatch.setattr("claude_agent_sdk.query", _fake_query)


def test_offset_survives_into_finding(tmp_path, monkeypatch):
    """Regression test for item 3: offset must survive from agent -> Finding."""
    corpus = tmp_path / "corpus"; corpus.mkdir()
    (corpus / "web_cache").mkdir()
    doc = "The road commission issued a 68-page notice of violation."
    (corpus / "doc.md").write_text(doc)

    _fake_agent(monkeypatch, {
        "severity": "PASS",
        "agent_summary": "The transcript confirms the notice.",
        "source_path": "doc.md",
        "source_excerpt": "road commission issued a 68-page notice",
        "source_excerpt_offset": [4, 43],
        "source_excerpt_similarity": 1.0,
        "confidence": 0.95,
        "human_review": False,
    })

    from pipeline import stage_b_verify as sb
    target = {"playbook": "fact_check", "target_text": "T", "anchor_text": "A",
              "claim_type": "numeric", "context": ""}
    import asyncio
    finding = asyncio.run(sb._verify_target_async(
        target, "sys", ["Read"], str(corpus), 10, 5, None, "",
        web_cache_dir=str(corpus / "web_cache"),
    ))
    assert finding is not None
    d = finding.to_dict()
    assert d["source_excerpt_offset"], "offset was dropped between agent and Finding"
    start, end = d["source_excerpt_offset"]
    assert doc[start:end] == d["source_excerpt"]
    assert d["excerpt_status"] == "exact"
    assert d["source_excerpt_similarity"] == 1.0


def test_fabricated_excerpt_is_repaired_in_the_finding(tmp_path, monkeypatch):
    """A fabricated (paraphrased) excerpt should be repaired with human_review=True."""
    corpus = tmp_path / "corpus"; corpus.mkdir()
    (corpus / "web_cache").mkdir()
    doc = "The road commission issued a 68-page notice of violation."
    (corpus / "doc.md").write_text(doc)

    _fake_agent(monkeypatch, {
        "severity": "PASS",
        "agent_summary": "The transcript confirms the notice.",
        "source_path": "doc.md",
        "source_excerpt": "the railroad commission issued a 68 page notice of violation",
        "source_excerpt_offset": None,
        "source_excerpt_similarity": None,
        "confidence": 0.95,
        "human_review": False,
    })

    from pipeline import stage_b_verify as sb
    target = {"playbook": "fact_check", "target_text": "T", "anchor_text": "A",
              "claim_type": "numeric", "context": ""}
    import asyncio
    finding = asyncio.run(sb._verify_target_async(
        target, "sys", ["Read"], str(corpus), 10, 5, None, "",
        web_cache_dir=str(corpus / "web_cache"),
    ))
    assert finding is not None
    d = finding.to_dict()
    assert d["excerpt_status"] == "repaired"
    # The excerpt should have been REPLACED with real document text
    assert d["source_excerpt"] != "the railroad commission issued a 68 page notice of violation"
    assert d["source_excerpt"] in doc
    assert d["human_review"] is True
    assert d["source_excerpt_similarity"] is not None
    assert d["source_excerpt_similarity"] < 1.0


def test_agent_options_toolbox_and_permissions(monkeypatch):
    """Verify ClaudeAgentOptions has correct tool lists and permission config."""
    from pipeline import stage_b_verify as sb

    captured_options = []

    def _fake_query(prompt, options):
        captured_options.append(options)
        async def _gen():
            return
            yield  # pragma: no cover
        return _gen()

    monkeypatch.setattr("claude_agent_sdk.query", _fake_query)

    claim = Claim(claim_id="c1", claim_text="T", source_quote="T", claim_type="numeric")
    asyncio.run(sb._verify_claim_async(
        claim, "/corpus", "sys", timeout=5, max_turns=1,
        web_cache_dir="/corpus/web_cache",
    ))

    assert captured_options, "query() was never called"
    opts = captured_options[0]

    # Read/Grep/Glob in tools but NOT in allowed_tools
    assert "Read" in opts.tools
    assert "Grep" in opts.tools
    assert "Glob" in opts.tools
    assert "Read" not in opts.allowed_tools
    assert "Grep" not in opts.allowed_tools
    assert "Glob" not in opts.allowed_tools

    # Write/Edit/Bash in neither list
    assert "Write" not in opts.tools
    assert "Edit" not in opts.tools
    assert "Bash" not in opts.tools

    # Permission config
    assert opts.permission_mode == "default"
    assert opts.can_use_tool is not None
    assert opts.strict_mcp_config is True

    # MCP servers
    assert "leder" in opts.mcp_servers


def test_user_prompt_has_no_shell_command(monkeypatch):
    """The agent prompt must not contain any shell command or fetch_page.py reference."""
    from pipeline import stage_b_verify as sb

    captured_prompts = []

    def _fake_query(prompt, options):
        async def _gen():
            async for msg in prompt:
                if isinstance(msg, dict) and "message" in msg:
                    captured_prompts.append(msg["message"]["content"])
            return
            yield  # pragma: no cover
        return _gen()

    monkeypatch.setattr("claude_agent_sdk.query", _fake_query)

    claim = Claim(claim_id="c1", claim_text="T", source_quote="T", claim_type="numeric")
    asyncio.run(sb._verify_claim_async(
        claim, "/corpus", "sys", timeout=5, max_turns=1,
        web_cache_dir="/corpus/web_cache",
    ))

    assert captured_prompts, "query() was never called"
    prompt = captured_prompts[0]
    assert "fetch_page.py" not in prompt
    assert "pipeline/tools/" not in prompt
    assert "python3 " not in prompt
    assert "sys.executable" not in prompt
    # Claim ID should not appear as a label in the prompt text
    assert "Claim ID:" not in prompt


def test_user_prompt_is_async_iterable_not_str(monkeypatch):
    """The prompt argument to query() must be an async iterable, not a string."""
    from pipeline import stage_b_verify as sb

    captured_prompts = []

    def _fake_query(prompt, options):
        captured_prompts.append(prompt)
        async def _gen():
            return
            yield  # pragma: no cover
        return _gen()

    monkeypatch.setattr("claude_agent_sdk.query", _fake_query)

    claim = Claim(claim_id="c1", claim_text="T", source_quote="T", claim_type="numeric")
    asyncio.run(sb._verify_claim_async(
        claim, "/corpus", "sys", timeout=5, max_turns=1,
        web_cache_dir="/corpus/web_cache",
    ))

    assert captured_prompts, "query() was never called"
    prompt = captured_prompts[0]
    assert not isinstance(prompt, str)
    assert hasattr(prompt, "__aiter__")


def test_verify_claim_async_no_settings_tempfile(monkeypatch):
    """No tempfile.NamedTemporaryFile should be called -- settings block removed."""
    from pipeline import stage_b_verify as sb
    import tempfile

    def _raising_named_temp(*args, **kwargs):
        raise RuntimeError("NamedTemporaryFile should not be called")

    monkeypatch.setattr("tempfile.NamedTemporaryFile", _raising_named_temp)

    def _fake_query(prompt, options):
        async def _gen():
            return
            yield  # pragma: no cover
        return _gen()

    monkeypatch.setattr("claude_agent_sdk.query", _fake_query)

    claim = Claim(claim_id="c1", claim_text="T", source_quote="T", claim_type="numeric")
    result = asyncio.run(sb._verify_claim_async(
        claim, "/corpus", "sys", timeout=5, max_turns=1,
        web_cache_dir="/corpus/web_cache",
    ))
    assert result is not None  # Succeeded without tempfile


# ── agent log: always-on transcript logging ─────────────────────────────


def test_serialize_message_no_truncation():
    """_serialize_message must preserve full text, thinking, input, and result
    without any truncation — the user wants to see what the agent was thinking."""
    from pipeline.stage_b_verify import _serialize_message
    from unittest.mock import MagicMock
    from claude_agent_sdk import (
        AssistantMessage, ResultMessage, TextBlock, ThinkingBlock, ToolUseBlock,
    )

    # Build a message with content exceeding the OLD truncation limits
    long_text = "x" * 6000
    long_thinking = "y" * 3000
    long_input = {"key": "z" * 3000}

    text_block = MagicMock(spec=TextBlock)
    text_block.text = long_text
    think_block = MagicMock(spec=ThinkingBlock)
    think_block.thinking = long_thinking
    think_block.signature = "sig"
    tool_block = MagicMock(spec=ToolUseBlock)
    tool_block.id = "id1"
    tool_block.name = "grep"
    tool_block.input = long_input

    msg = MagicMock(spec=AssistantMessage)
    msg.content = [text_block, think_block, tool_block]

    d = _serialize_message(msg)
    blocks = d["blocks"]
    assert blocks[0]["text"] == long_text, "text was truncated"
    assert blocks[1]["thinking"] == long_thinking, "thinking was truncated"
    assert blocks[2]["input"] == long_input, "input was truncated / str()'d"

    # Result long text
    long_result = "r" * 600
    res_msg = MagicMock(spec=ResultMessage)
    res_msg.content = []
    res_msg.result = long_result
    res_msg.subtype = "success"
    res_msg.total_cost_usd = 0.0042
    res_msg.num_turns = 5

    d2 = _serialize_message(res_msg)
    assert d2["result"] == str(long_result), "result was truncated"
    assert d2["total_cost_usd"] == 0.0042


def test_verify_claim_async_always_collects_transcript(tmp_path, monkeypatch):
    """Transcript is collected even without a debug_dir — always-on."""
    from pipeline import stage_b_verify as sb

    def _fake_query(prompt, options):
        # Return one AssistantMessage with text, then a ResultMessage
        from unittest.mock import MagicMock
        from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

        text_block = MagicMock(spec=TextBlock)
        text_block.text = "hello"
        asst = MagicMock(spec=AssistantMessage)
        asst.content = [text_block]

        result = MagicMock(spec=ResultMessage)
        result.structured_output = {
            "severity": "PASS", "agent_summary": "ok",
            "source_path": "doc.md", "source_excerpt": "hello",
            "source_excerpt_offset": [0, 5], "source_excerpt_similarity": 1.0,
            "confidence": 0.95, "human_review": False,
        }

        async def _gen():
            yield asst
            yield result
        return _gen()

    monkeypatch.setattr("claude_agent_sdk.query", _fake_query)

    claim = Claim(claim_id="c1", claim_text="T", source_quote="T", claim_type="numeric")
    result = asyncio.run(sb._verify_claim_async(
        claim, str(tmp_path), "sys", timeout=5, max_turns=1,
        web_cache_dir=str(tmp_path / "web_cache"),
    ))
    assert result.verdict is not None  # Still works end-to-end


def test_verify_claim_async_writes_agent_log(tmp_path, monkeypatch):
    """Passing agent_log_dir must create .log and .jsonl files."""
    from pipeline import stage_b_verify as sb
    import json as _json

    def _fake_query(prompt, options):
        from unittest.mock import MagicMock
        from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

        text_block = MagicMock(spec=TextBlock)
        text_block.text = "found it"
        asst = MagicMock(spec=AssistantMessage)
        asst.content = [text_block]

        result = MagicMock(spec=ResultMessage)
        result.structured_output = {
            "severity": "PASS", "agent_summary": "ok",
            "source_excerpt": "found it", "source_excerpt_similarity": 1.0,
        }

        async def _gen():
            yield asst
            yield result
        return _gen()

    monkeypatch.setattr("claude_agent_sdk.query", _fake_query)

    log_dir = tmp_path / "agent-logs" / "20260101-120000"
    claim = Claim(claim_id="c1", claim_text="T", source_quote="T", claim_type="numeric")
    result = asyncio.run(sb._verify_claim_async(
        claim, str(tmp_path), "sys", timeout=5, max_turns=1,
        agent_log_dir=str(log_dir),
        web_cache_dir=str(tmp_path / "web_cache"),
    ))
    assert result.verdict is not None

    # Files must exist
    log_file = log_dir / "c1.log"
    jsonl_file = log_dir / "c1.jsonl"
    assert log_file.exists(), ".log not written"
    assert jsonl_file.exists(), ".jsonl not written"

    # JSONL must be valid JSON lines
    lines = jsonl_file.read_text().strip().split("\n")
    assert len(lines) >= 1
    assert _json.loads(lines[0])  # each line is valid JSON

    # .log must contain the text
    assert "found it" in log_file.read_text()


def test_agent_log_includes_thinking(tmp_path, monkeypatch):
    """The .log file must include thinking blocks, delimited by --- thinking ---."""
    from pipeline import stage_b_verify as sb

    def _fake_query(prompt, options):
        from unittest.mock import MagicMock
        from claude_agent_sdk import AssistantMessage, ResultMessage, ThinkingBlock

        think_block = MagicMock(spec=ThinkingBlock)
        think_block.thinking = "hmm, let me search for that"
        asst = MagicMock(spec=AssistantMessage)
        asst.content = [think_block]

        result = MagicMock(spec=ResultMessage)
        result.structured_output = {
            "severity": "PASS", "agent_summary": "ok",
            "source_excerpt": "x", "source_excerpt_similarity": 1.0,
        }

        async def _gen():
            yield asst
            yield result
        return _gen()

    monkeypatch.setattr("claude_agent_sdk.query", _fake_query)

    log_dir = tmp_path / "agent-logs" / "run01"
    claim = Claim(claim_id="c1", claim_text="T", source_quote="T", claim_type="numeric")
    asyncio.run(sb._verify_claim_async(
        claim, str(tmp_path), "sys", timeout=5, max_turns=1,
        agent_log_dir=str(log_dir),
        web_cache_dir=str(tmp_path / "web_cache"),
    ))

    log_text = (log_dir / "c1.log").read_text()
    assert "--- thinking ---" in log_text
    assert "hmm, let me search for that" in log_text


def test_agent_log_and_debug_can_coexist(tmp_path, monkeypatch):
    """Both agent_log_dir and debug_dir write independently."""
    from pipeline import stage_b_verify as sb
    def _fake_query(prompt, options):
        from unittest.mock import MagicMock
        from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

        text_block = MagicMock(spec=TextBlock)
        text_block.text = "x"
        asst = MagicMock(spec=AssistantMessage)
        asst.content = [text_block]
        result = MagicMock(spec=ResultMessage)
        result.structured_output = {
            "severity": "PASS", "agent_summary": "ok",
            "source_excerpt": "x", "source_excerpt_similarity": 1.0,
        }

        async def _gen():
            yield asst
            yield result
        return _gen()

    monkeypatch.setattr("claude_agent_sdk.query", _fake_query)

    agent_dir = tmp_path / "agent" / "ts"
    debug_dir = tmp_path / "debug"
    claim = Claim(claim_id="c1", claim_text="T", source_quote="T", claim_type="numeric")
    asyncio.run(sb._verify_claim_async(
        claim, str(tmp_path), "sys", timeout=5, max_turns=1,
        agent_log_dir=str(agent_dir), debug_dir=str(debug_dir),
        web_cache_dir=str(tmp_path / "web_cache"),
    ))

    assert (agent_dir / "c1.log").exists()
    assert (agent_dir / "c1.jsonl").exists()
    assert (debug_dir / "c1.log").exists()
    assert (debug_dir / "c1.jsonl").exists()


def test_run_stage_b_creates_agent_log_run_folder(tmp_path, monkeypatch):
    """run_stage_b with agent_log_dir creates a YYYYMMDD-HHMMSS subfolder."""
    from pipeline import stage_b_verify as sb

    # Write a minimal targets.json
    targets = tmp_path / "targets.json"
    import json as _json
    _json.dump({
        "targets": [{"playbook": "fact_check", "target_text": "T",
                      "anchor_text": "A", "claim_type": "numeric"}],
        "article_summary": "S",
        "article_file": "a.md",
    }, open(str(targets), "w"))

    # Stub _verify_target_async so we don't spawn real agents
    from pipeline.finding import Finding
    async def _fake_verify(*a, **kw):
        return Finding(
            finding_id="fc-abc", check_type="fact_check", severity="PASS",
            target_text="T", anchor_text="A", agent_summary="ok",
        )

    monkeypatch.setattr(sb, "_verify_target_async", _fake_verify)

    log_base = tmp_path / "agent-logs"
    output = tmp_path / "findings.json"
    sb.run_stage_b(
        targets_path=str(targets), output_path=str(output),
        corpus_root=str(tmp_path), force_run=True,
        agent_log_dir=str(log_base),
    )

    # A timestamped subfolder was created
    subdirs = list(log_base.iterdir())
    assert len(subdirs) == 1, f"expected 1 subdir, got {subdirs}"
    name = subdirs[0].name
    # Must match YYYYMMDD-HHMMSS
    import re
    assert re.match(r"^\d{8}-\d{6}$", name), f"bad run folder name: {name}"
