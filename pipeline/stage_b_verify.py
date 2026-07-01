"""Stage B: Verify claims against the local corpus (with web fallback).

Each claim gets a real Claude Code agent session via the Claude Agent SDK.
Claims are verified in parallel with configurable concurrency.
"""
from __future__ import annotations

import asyncio
import json
import re
import sys

from pipeline.models import Claim, ClaimsDocument


AGENT_SYSTEM_PROMPT = """You are a fact-checker verifying claims against a corpus of documents.
For each claim, your job is to FIND the best available source and EVALUATE
whether it supports or contradicts the claim.

## SEARCH STRATEGY -- Tiered (Local First)

The corpus is organized with summaries at three levels. Search top-down.

**CRITICAL RULE: Summaries are a map, not the territory.** Summaries tell you WHERE to look, but you MUST verify the claim against the ORIGINAL converted .md file. Never cite a summary as your source -- the summary is a search index, not evidence. If a summary mentions a fact but the original file doesn't confirm it, the verdict is "unsupported." If the original file is unavailable or unreadable, flag `human_review: true` and cite the best available source.

1. PROJECT LEVEL -- CORPUS_OVERVIEW.md / CORPUS_ROLLUP.md
   Cross-cutting patterns, claims about "all permits" or multi-case comparisons.
   Use these to IDENTIFY which cases are relevant, then drill into those cases.
   Use Bash with ripgrep: rg "keywords" .

2. CASE LEVEL -- _CASE_OVERVIEW.md in each permit folder
   Permit-specific claims about a single operator or facility.
   Use these to IDENTIFY which document categories contain the relevant data.
   Use Bash with ripgrep: rg "keywords" <case-folder>/

3. FILE LEVEL -- _summary.md files within case subfolders
   Specific data points, numbers, dates, names.
   Use these to IDENTIFY the exact original file to read.
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
    (OCR/conversion risk -- the original PDF should be spot-checked)
  - The verdict is "unsupported" or "contradicted"
  - You are below 80% confidence in your assessment

- rationale: One sentence explaining what the source says and why it supports,
  contradicts, or fails to address the claim. Be specific -- mention the actual
  source content, not just "a document was found."

## OUTPUT

When you have finished your research, output ONLY this JSON object on a single line.
No other text before or after:
{"verdict":"supported|contradicted|unsupported","source_proximity":"original|derived|unverifiable","source_path":null,"source_url":null,"rationale":"One sentence citing the specific source and what it says.","human_review":false,"confidence":0.0}

- source_path: relative path to the local file within the corpus, or null
- source_url: URL of the web source, or null
- confidence: number between 0.0 and 1.0"""


def agent_failure_result(claim: Claim) -> Claim:
    """Return a fallback result when the agent fails entirely."""
    claim.verdict = "unsupported"
    claim.source_proximity = "unverifiable"
    claim.source_path = None
    claim.source_url = None
    claim.rationale = "Agent failed after retry -- unable to verify."
    claim.human_review = True
    claim.confidence = 0.0
    return claim


def parse_verdict(claim: Claim, text: str) -> Claim:
    """Extract JSON verdict from agent output. Fall back to failure result on any error."""
    if not text:
        return agent_failure_result(claim)

    # Find JSON object containing "verdict" key
    match = re.search(r'\{[^{}]*"verdict"\s*:\s*"[^"]*"[^{}]*\}', text, re.DOTALL)
    if not match:
        print(f"  [{claim.claim_id}] No verdict JSON found in agent output", file=sys.stderr)
        return agent_failure_result(claim)

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        print(f"  [{claim.claim_id}] Invalid JSON in agent output", file=sys.stderr)
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
        print(f"  [{claim.claim_id}] Missing field in verdict: {e}", file=sys.stderr)
        return agent_failure_result(claim)

    return claim


async def _verify_claim_async(
    claim: Claim,
    corpus_root: str,
    system_prompt: str,
) -> Claim:
    """One claim = one Claude Code agent session via the Claude Agent SDK."""
    from claude_agent_sdk import (
        query, ClaudeAgentOptions,
        AssistantMessage, TextBlock,
        ProcessError, CLINotFoundError,
    )

    prompt = (
        f"Verify this claim:\n\n"
        f"{claim.claim_text}\n\n"
        f"Claim type: {claim.claim_type}\n\n"
        f"When you have finished your research, output ONLY a JSON object "
        f"with your verdict. No other text before or after the JSON."
    )

    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        allowed_tools=["Read", "Bash", "WebSearch", "WebFetch"],
        permission_mode="acceptEdits",
        cwd=corpus_root,
        max_turns=15,
    )

    full_text = ""
    try:
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        full_text += block.text
    except CLINotFoundError:
        return agent_failure_result(claim)
    except ProcessError as e:
        print(f"  [{claim.claim_id}] Agent process error (exit {e.exit_code})", file=sys.stderr)
        return agent_failure_result(claim)
    except Exception as e:
        print(f"  [{claim.claim_id}] Agent error: {e}", file=sys.stderr)
        return agent_failure_result(claim)

    return parse_verdict(claim, full_text)


async def _verify_claim_with_retry(
    claim: Claim,
    corpus_root: str,
    system_prompt: str,
) -> Claim:
    """Verify a claim with one retry on failure."""
    try:
        return await _verify_claim_async(claim, corpus_root, system_prompt)
    except Exception as e:
        print(f"  Agent failed for {claim.claim_id}: {e}. Retrying once...")
        try:
            return await _verify_claim_async(claim, corpus_root, system_prompt)
        except Exception as e2:
            print(f"  Agent failed again for {claim.claim_id}: {e2}. Marking as unsupported.")
            return agent_failure_result(claim)


async def _verify_all(
    claims: list[Claim],
    corpus_root: str,
    system_prompt: str,
    concurrency: int,
) -> list[Claim]:
    """Verify all claims with a concurrency semaphore."""
    sem = asyncio.Semaphore(concurrency)
    total = len(claims)
    completed = 0

    async def verify_one(claim: Claim) -> Claim:
        nonlocal completed
        async with sem:
            result = await _verify_claim_with_retry(claim, corpus_root, system_prompt)
            completed += 1
            status = "?" if result.verdict is None else (
                "✓" if result.verdict == "supported" else (
                    "✗" if result.verdict == "contradicted" else "?"
                )
            )
            print(f"  [{completed}/{total}] {status} {claim.claim_id}: {claim.claim_text[:80]}...")
            return result

    return await asyncio.gather(*[verify_one(c) for c in claims])


def run_stage_b(
    claims_path: str,
    output_path: str,
    corpus_root: str,
    web_cache_dir: str = "",
    model: str = "",
    concurrency: int = 3,
) -> ClaimsDocument:
    """Load claims.json, verify each claim via Claude Agent SDK, write enriched claims.json.

    Args:
        claims_path: Path to claims.json from Stage A.
        output_path: Where to write the enriched claims.json.
        corpus_root: Root directory of the local document corpus.
        web_cache_dir: Ignored (kept for backward compat). Agent uses its own fetch.
        model: Ignored (kept for backward compat). Agent uses env-configured model.
        concurrency: Max concurrent agent sessions (default 3).
    """
    with open(claims_path, encoding="utf-8") as f:
        doc = ClaimsDocument.from_json(f.read())

    total = len(doc.claims)
    print(f"Verifying {total} claims (concurrency={concurrency})...")

    results = asyncio.run(_verify_all(
        doc.claims,
        corpus_root,
        AGENT_SYSTEM_PROMPT,
        concurrency,
    ))

    # Merge results back, preserving original claim order
    result_map = {r.claim_id: r for r in results}
    doc.claims = [result_map.get(c.claim_id, agent_failure_result(c)) for c in doc.claims]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(doc.to_json())

    supported = sum(1 for c in doc.claims if c.verdict == "supported")
    contradicted = sum(1 for c in doc.claims if c.verdict == "contradicted")
    unsupported = sum(1 for c in doc.claims if c.verdict == "unsupported")
    review = sum(1 for c in doc.claims if c.human_review)

    print(f"\nDone: {supported} supported, {contradicted} contradicted, {unsupported} unsupported")
    print(f"  {review} claim(s) flagged for human review")

    return doc
