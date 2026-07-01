"""Stage B: Verify claims against the local corpus (with web fallback).

Each claim gets a real Claude Code agent session via the Claude Agent SDK.
Claims are verified in parallel with configurable concurrency.
Results are written incrementally — crash-resistant.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys

from typing import Literal

from pydantic import BaseModel

from pipeline.models import Claim, ClaimsDocument, Article, Corpus


class VerdictOutput(BaseModel):
    """Structured output schema for claim verification agents."""
    verdict: Literal["supported", "contradicted", "unsupported"]
    source_proximity: Literal["original", "derived", "unverifiable"]
    source_path: str | None = None
    source_url: str | None = None
    rationale: str
    human_review: bool
    confidence: float


AGENT_SYSTEM_PROMPT = """You are a fact-checker verifying claims against a corpus of documents.
For each claim, your job is to FIND the best available source and EVALUATE
whether it supports or contradicts the claim.

## SEARCH STRATEGY -- Tiered (Local First)

The corpus is organized with summaries at three levels. Search top-down.

**CRITICAL RULE: Summaries are a map, not the territory.** Summaries tell you WHERE to look, but you MUST verify the claim against the ORIGINAL converted .md file. Never cite a summary as your source -- the summary is a search index, not evidence. If a summary mentions a fact but the original file doesn't confirm it, the verdict is "unsupported." If the original file is unavailable or unreadable, flag `human_review: true` and cite the best available source.

**SANDBOX: Stay inside the corpus.** All files you need are within the current working directory. Use relative paths only (e.g. `rg "term" .` or `rg "term" LA-0304/`). Never use absolute paths like `/home/...` -- you are searching a specific document corpus, not the entire filesystem. The Read tool also takes paths relative to this directory.

1. PROJECT LEVEL -- CORPUS_OVERVIEW.md / CORPUS_ROLLUP.md
   Cross-cutting patterns, claims about "all permits" or multi-case comparisons.
   Use Bash with ripgrep: rg "keywords" .

2. CASE LEVEL -- _CASE_OVERVIEW.md in each permit folder
   Permit-specific claims about a single operator or facility.
   Use Bash with ripgrep: rg "keywords" <case-folder>/

3. FILE LEVEL -- _summary.md files within case subfolders
   Specific data points, numbers, dates, names.
   Use Bash with ripgrep: rg "keywords" <case-folder>/<category>/

4. ORIGINALS (MANDATORY VERIFICATION STEP) -- converted .md files
   Once the summaries have pointed you to a specific file, you MUST read the
   original .md file (the one WITHOUT "_summary" in its name) and verify the
   claim against its content. This is not optional -- this is where the actual
   evidence lives. Cite this file as `source_path`.
   Use the Read tool to open the file.

5. WEB -- when the claim involves information not in the local corpus
   Some claims are about statutes, company statements, news events, or external
   context that simply doesn't exist in the local files. That's expected.
   Use WebSearch to find relevant pages, then WebFetch to pull and evaluate
   the best match.

## EVALUATION

For each claim, determine:

- verdict:
  "supported" -- source confirms the claim
  "contradicted" -- source directly contradicts the claim
  "unsupported" -- no source found, or source exists but is silent on this claim

- source_proximity:
  "original" -- the actual permit, report, statute, email, or official webpage
  "derived" -- a summary, overview, or secondary source. ONLY use this when the
    original file cannot be read. If you cite a derived source, you MUST flag
    `human_review: true` and explain why the original was unavailable.
  "unverifiable" -- no source could be found at all

- human_review: set TRUE when:
  - The claim is central/important AND the best source is a converted file
  - The verdict is "unsupported" or "contradicted"
  - You are below 80% confidence in your assessment

- rationale: One sentence explaining what the source says and why it supports,
  contradicts, or fails to address the claim. Be specific -- mention the actual
  source content, not just "a document was found."

## PACE YOURSELF

Aim to complete verification in 30 turns or fewer. Most claims can be verified
in 3-4 searches + 1-2 file reads. If you find yourself searching repeatedly
without narrowing, stop and work with what you have — "unsupported" with a
clear rationale is better than exhausting your turn budget.

## OUTPUT

Your verdict will be collected as structured data — no need to format JSON manually.
Explain your reasoning in the rationale field: mention the specific source document
and what it says. If no source was found, explain what you searched for.
source_path should be a relative path within the corpus, source_url a full URL.
confidence is between 0.0 (complete guess) and 1.0 (verbatim match in primary source)."""

# ---- helpers ----

def agent_failure_result(claim: Claim) -> Claim:
    claim.verdict = "unsupported"
    claim.source_proximity = "unverifiable"
    claim.source_path = None
    claim.source_url = None
    claim.rationale = "Agent failed after retry -- unable to verify."
    claim.human_review = True
    claim.confidence = 0.0
    return claim


def parse_verdict(claim: Claim, text: str) -> Claim:
    """Extract JSON verdict from agent output text."""
    if not text:
        return agent_failure_result(claim)

    match = re.search(r'\{[^{}]*"verdict"\s*:\s*"[^"]*"[^{}]*\}', text, re.DOTALL)
    if not match:
        print(f"  [{claim.claim_id}] No verdict JSON found in output", file=sys.stderr)
        return agent_failure_result(claim)

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        print(f"  [{claim.claim_id}] Invalid JSON in output", file=sys.stderr)
        return agent_failure_result(claim)

    try:
        claim.verdict = data["verdict"]
        claim.source_proximity = data["source_proximity"]
        claim.source_path = data.get("source_path")
        claim.source_url = data.get("source_url")
        claim.rationale = data.get("rationale", "No rationale provided.")
        claim.human_review = data.get("human_review", True)
        claim.confidence = data.get("confidence")
    except KeyError as e:
        print(f"  [{claim.claim_id}] Missing field: {e}", file=sys.stderr)
        return agent_failure_result(claim)

    return claim


def _write_incremental(claims: list[Claim], results_by_id: dict[str, Claim], output_path: str) -> None:
    """Write partial results so progress is never lost."""
    merged = []
    for c in claims:
        merged.append(results_by_id.get(c.claim_id, c))
    # Build a lightweight doc
    data = {
        "article": {"path": "", "title": "", "generated_at": ""},
        "corpus": {"root": "", "project": ""},
        "claims": [
            {
                "claim_id": c.claim_id,
                "claim_text": c.claim_text,
                "source_quote": c.source_quote,
                "claim_type": c.claim_type,
                "verdict": c.verdict,
                "source_proximity": c.source_proximity,
                "source_path": c.source_path,
                "source_url": c.source_url,
                "rationale": c.rationale,
                "human_review": c.human_review,
                "confidence": c.confidence,
                "reconciled": c.reconciled,
            }
            for c in merged
        ],
    }
    tmp = output_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, output_path)


# ---- async agent logic ----

async def _verify_claim_async(
    claim: Claim,
    corpus_root: str,
    system_prompt: str,
    timeout: int = 600,
    max_turns: int = 30,
    debug_dir: str | None = None,
) -> Claim:
    from claude_agent_sdk import (
        query, ClaudeAgentOptions,
        AssistantMessage, TextBlock, ResultMessage,
        ProcessError, CLINotFoundError,
    )

    prompt = (
        f"Verify this claim:\n\n"
        f"{claim.claim_text}\n\n"
        f"Claim type: {claim.claim_type}"
    )

    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        allowed_tools=["Read", "Bash", "WebSearch", "WebFetch"],
        permission_mode="acceptEdits",
        cwd=corpus_root,
        max_turns=max_turns,
        output_format={"type": "json_schema", "schema": VerdictOutput.model_json_schema()},
    )

    full_text = ""
    structured_output = None
    transcript: list[dict] = []  # Full message stream for debug

    async def _run():
        nonlocal full_text, structured_output
        async for message in query(prompt=prompt, options=options):
            if debug_dir:
                try:
                    transcript.append(_serialize_message(message))
                except Exception:
                    pass

            if isinstance(message, ResultMessage):
                structured_output = message.structured_output
            elif isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        full_text += block.text

    try:
        await asyncio.wait_for(_run(), timeout=timeout)
    except asyncio.TimeoutError:
        print(f"  [{claim.claim_id}] Timed out after {timeout}s", file=sys.stderr)
        return agent_failure_result(claim)
    except CLINotFoundError:
        return agent_failure_result(claim)
    except ProcessError as e:
        print(f"  [{claim.claim_id}] Process error (exit {e.exit_code})", file=sys.stderr)
        return agent_failure_result(claim)
    except Exception as e:
        print(f"  [{claim.claim_id}] Error: {e}", file=sys.stderr)
        return agent_failure_result(claim)

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        with open(os.path.join(debug_dir, f"{claim.claim_id}.log"), "w") as f:
            f.write(full_text)
        with open(os.path.join(debug_dir, f"{claim.claim_id}.jsonl"), "w") as f:
            for entry in transcript:
                f.write(json.dumps(entry, default=str) + "\n")

    # Primary path: structured output from the SDK
    if structured_output and isinstance(structured_output, dict):
        return _populate_claim_from_dict(claim, structured_output)

    # Fallback: regex parse from text (older CLI, no structured output support)
    return parse_verdict(claim, full_text)


def _populate_claim_from_dict(claim: Claim, data: dict) -> Claim:
    """Populate claim fields from a structured output dict. Falls back to
    agent_failure_result on missing required fields."""
    try:
        claim.verdict = data["verdict"]
        claim.source_proximity = data["source_proximity"]
        claim.source_path = data.get("source_path")
        claim.source_url = data.get("source_url")
        claim.rationale = data.get("rationale", "No rationale provided.")
        claim.human_review = data.get("human_review", True)
        claim.confidence = data.get("confidence")
    except KeyError as e:
        print(f"  [{claim.claim_id}] Missing field in structured output: {e}", file=sys.stderr)
        return agent_failure_result(claim)
    return claim


def _serialize_message(msg) -> dict:
    """Convert an SDK message to a plain dict for JSONL debug output."""
    result: dict = {"type": type(msg).__name__}
    if hasattr(msg, "content") and msg.content:
        blocks = []
        for block in msg.content:
            b: dict = {"type": type(block).__name__}
            if hasattr(block, "text"):
                b["text"] = block.text[:5000]
            if hasattr(block, "name"):
                b["name"] = block.name
            if hasattr(block, "input"):
                b["input"] = str(block.input)[:2000]
            if hasattr(block, "tool_use_id"):
                b["tool_use_id"] = block.tool_use_id
            if hasattr(block, "thinking"):
                b["thinking"] = block.thinking[:2000]
            blocks.append(b)
        result["blocks"] = blocks
    if hasattr(msg, "result"):
        result["result"] = str(msg.result)[:500]
    return result


async def _verify_claim_with_retry(
    claim: Claim,
    corpus_root: str,
    system_prompt: str,
    timeout: int = 600,
    max_turns: int = 30,
    debug_dir: str | None = None,
) -> Claim:
    """Verify a claim with one retry if the agent produced no verdict."""
    result = await _verify_claim_async(claim, corpus_root, system_prompt, timeout, max_turns, debug_dir)
    if result.verdict is not None:
        return result
    # Agent produced no verdict at all — retry once
    print(f"  Retrying {claim.claim_id}...", file=sys.stderr)
    return await _verify_claim_async(claim, corpus_root, system_prompt, timeout, max_turns, debug_dir)


async def _verify_all(
    claims: list[Claim],
    corpus_root: str,
    system_prompt: str,
    concurrency: int,
    timeout: int = 600,
    max_turns: int = 30,
    output_path: str | None = None,
    debug_dir: str | None = None,
) -> list[Claim]:
    import time as time_mod

    sem = asyncio.Semaphore(concurrency)
    total = len(claims)
    completed = 0
    agent_times: list[float] = []
    results_by_id: dict[str, Claim] = {}

    async def verify_one(claim: Claim) -> Claim:
        nonlocal completed
        t0 = time_mod.time()
        async with sem:
            result = await _verify_claim_with_retry(
                claim, corpus_root, system_prompt, timeout, max_turns, debug_dir,
            )
        elapsed = time_mod.time() - t0
        agent_times.append(elapsed)
        completed += 1
        results_by_id[result.claim_id] = result

        status = (
            "✓" if result.verdict == "supported" else
            "✗" if result.verdict == "contradicted" else "?"
        )

        avg = sum(agent_times) / len(agent_times) if agent_times else 0
        remaining = total - completed
        eta_sec = int(avg * remaining / concurrency) if concurrency else 0
        eta_str = f"{eta_sec // 60}m{eta_sec % 60}s" if eta_sec > 0 else "—"

        print(
            f"  [{completed}/{total}] {status} {claim.claim_id} "
            f"({elapsed:.0f}s, avg {avg:.0f}s, ETA {eta_str}) "
            f"{claim.claim_text[:60]}..."
        )

        if output_path:
            _write_incremental(claims, results_by_id, output_path)

        return result

    return await asyncio.gather(*[verify_one(c) for c in claims])


# ---- public entry point ----

def run_stage_b(
    claims_path: str,
    output_path: str,
    corpus_root: str,
    web_cache_dir: str = "",
    model: str = "",
    concurrency: int = 32,
    timeout: int = 600,
    max_turns: int = 30,
    debug_count: int = 0,
) -> ClaimsDocument:
    """Load claims.json, verify each claim, write enriched claims.json.

    Args:
        claims_path: Path to claims.json from Stage A.
        output_path: Where to write enriched claims.json.
        corpus_root: Root of the local document corpus.
        concurrency: Max concurrent agent sessions (default 32).
        timeout: Per-agent timeout in seconds (default 600).
        max_turns: Max tool-calling turns per agent (default 30).
        debug_count: If >0, run only the first N claims and save agent
                     output to debug/ directory alongside output_path.
    """
    import time as time_mod

    with open(claims_path, encoding="utf-8") as f:
        doc = ClaimsDocument.from_json(f.read())

    claims = doc.claims
    total = len(claims)

    debug_dir = None
    if debug_count > 0:
        claims = claims[:debug_count]
        total = len(claims)
        debug_dir = os.path.join(os.path.dirname(output_path) or ".", "debug")
        print(f"Debug mode: {total} claims, logs → {debug_dir}/", file=sys.stderr)

    model_name = os.environ.get("ANTHROPIC_MODEL", "unknown")

    print(f"Dispatching {total} agents", file=sys.stderr)
    print(f"  Concurrency: {concurrency}  |  Model: {model_name}", file=sys.stderr)
    print(f"  Timeout: {timeout}s/agent  |  Max turns: {max_turns}", file=sys.stderr)
    print(f"  Corpus: {corpus_root}", file=sys.stderr)

    t0 = time_mod.time()
    results = asyncio.run(_verify_all(
        claims,
        corpus_root,
        AGENT_SYSTEM_PROMPT,
        concurrency,
        timeout=timeout,
        max_turns=max_turns,
        output_path=output_path,
        debug_dir=debug_dir,
    ))
    elapsed = time_mod.time() - t0

    # Merge results back into full claim set
    result_map = {r.claim_id: r for r in results}
    doc.claims = [result_map.get(c.claim_id, c) for c in doc.claims]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(doc.to_json())

    supported = sum(1 for c in doc.claims if c.verdict == "supported")
    contradicted = sum(1 for c in doc.claims if c.verdict == "contradicted")
    unsupported = sum(1 for c in doc.claims if c.verdict == "unsupported")
    review = sum(1 for c in doc.claims if c.human_review)

    avg = elapsed / total if total > 0 else 0
    print(f"\nDone: {supported} ✓ / {contradicted} ✗ / {unsupported} ?  ({review} flagged for review)")
    print(f"  {total} claims in {elapsed:.0f}s ({avg:.1f}s avg, ~{total / elapsed * 60:.0f}/min)")

    return doc
