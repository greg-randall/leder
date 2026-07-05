"""Stage B: Verify claims against the local corpus (with web fallback).

Each claim gets a real Claude Code agent session via the Claude Agent SDK.
Claims are verified in parallel with configurable concurrency.
Results are written incrementally and are crash-resistant.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys

from typing import Literal, Optional

from pydantic import BaseModel, Field

from pathlib import Path as _Path

from pipeline.models import Claim, ClaimsDocument


class VerdictOutput(BaseModel):
    """Structured output schema for claim verification agents."""
    verdict: Literal["supported", "contradicted", "unsupported"]
    source_proximity: Literal["original", "derived", "unverifiable"]
    source_path: str | None = None
    source_url: str | None = None
    rationale: str
    source_excerpt: str = ""  # Verbatim text from the source document that supports/contradicts
    human_review: bool
    confidence: float


class FindingOutput(BaseModel):
    """Structured output schema for playbook-driven verification agents."""
    severity: str = Field(description="PASS, WARNING, or CRITICAL")
    agent_summary: str = Field(description="1-2 sentences explaining the finding")
    recommended_action: Optional[str] = Field(default=None)
    source_path: Optional[str] = Field(default=None)
    source_url: Optional[str] = Field(default=None)
    source_excerpt: Optional[str] = Field(default=None)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    human_review: Optional[bool] = Field(default=None)
    metadata: dict = Field(default_factory=dict)


AGENT_SYSTEM_PROMPT = """You are a fact-checker verifying claims against a corpus of documents.
For each claim, your job is to FIND the best available source and EVALUATE
whether it supports, contradicts, or is silent on the claim.

## SEARCH STRATEGY -- Tiered (Local First)

The corpus is organized with summaries at three levels. Search top-down.

**CRITICAL RULE: Summaries are a map, not the territory.** Summaries tell you
WHERE to look, but you MUST verify the claim against the ORIGINAL converted .md
file. Never cite a summary as your source -- the summary is a search index, not
evidence. If a summary mentions a fact but the original file doesn't confirm it,
the verdict is "unsupported." If the original file is unavailable or unreadable,
flag `human_review: true` and cite the best available source.

**SANDBOX: Stay inside the corpus.** Your working directory IS the document
corpus. All files you need are here. Use relative paths only: `rg "term" .` or
`rg "term" LA-0304/`. NEVER use absolute paths (`/home/...`, `/mnt/...`). The
Read tool also takes paths relative to this directory. If you can't find a
source within the corpus, the verdict is "unsupported" -- do not go looking
elsewhere on the filesystem.

1. PROJECT LEVEL -- scan the top-level overview files.
   CORPUS_OVERVIEW.md: cross-cutting patterns across ALL cases, per-operator
     notes, data highlights, and outlier cases. Start here for claims about
     "all permits," multi-case comparisons, or general trends.
   CORPUS_ROLLUP.md: statistical summary (counts, dates, volumes). Use for
     claims with aggregate numbers.

2. GROUP LEVEL -- _FOLDER_SUMMARY.md inside each case folder.
   One overview per permit/pilot. Use for claims about a specific operator,
   facility, or permit number. To discover case folder names, start by
   listing the top-level directory or searching CORPUS_OVERVIEW.md for the
   permit/operator name mentioned in the claim.
   Use: `rg "term" <case-folder>/`

3. FILE LEVEL -- _summary.md files within case subfolders.
   One summary per source document. Use these to find which specific file
   contains the number or detail you need. Subfolder names (e.g. "Compl",
   "Permits", "RADs") correspond to document categories -- explore with
   `ls <case-folder>/` to see what categories exist.
   Use: `rg "term" <case-folder>/<category>/`

4. ORIGINALS (MANDATORY VERIFICATION STEP) -- converted .md files.
   Once the summaries have pointed you to a specific file, you MUST read the
   original .md file (the one WITHOUT "_summary" in its name) and verify the
   claim against its content. This is not optional -- this is where the actual
   evidence lives. Cite this file as `source_path`.
   Use the Read tool to open the file.

5. WEB -- when the claim involves information not in the local corpus.
   Use WebSearch to find relevant pages, then fetch the best match.
   SAVE what you fetch: write the content to `web_cache/<claim_id>/page.md`
   (create the directory first). This is the audit trail for human review.
   Fetch methods to try:
   - `curl -sL <url> | head -200` for plain text / PDF links
   - WebFetch for JavaScript-heavy pages
   - `curl -sL https://r.jina.ai/<url>` for clean markdown

## EVALUATION

For each claim, determine:

- verdict -- pick ONE:
  "supported"     -- source explicitly states the claim (e.g., a permit says
                     "irrigates 165 acres" and the claim is "irrigates 165 acres")
  "contradicted"  -- source says the opposite (e.g., the claim says "no
                     exceedances" but the quarterly report shows values above
                     the limit). Use this for direct contradiction only; a
                     source being silent or ambiguous is "unsupported."
  "unsupported"   -- no source found, or source exists but doesn't address
                     this specific claim

- source_proximity:
  "original"      -- the primary document itself (permit, report, statute,
                     email, official webpage, lab result, inspection report)
  "derived"       -- a summary, overview, or secondary analysis. ONLY use
                     when the original cannot be read (corrupted, missing).
                     MUST flag `human_review: true` and explain why.
  "unverifiable"  -- no source could be found at all

- human_review: set TRUE if ANY of the following are true:
  - The verdict is "unsupported" or "contradicted"
  - source_proximity is "derived"
  - Your confidence is below 80%

- rationale: One to two sentences explaining what the source says and why it
  supports, contradicts, or fails to address the claim. Be specific -- mention
  the actual source content. BAD: "A document was found that supports this."
  GOOD: "The 2020 permit renewal (p. 3) states the facility 'shall irrigate
  165 acres via sprinkler,' matching the claim."

- source_excerpt: The verbatim text from the source document that confirms
  or contradicts the claim. Copy-paste the exact sentence(s) from the source.
  This is the evidence the human reviewer will check against.

## PACE YOURSELF

Aim to complete verification in 30 turns or fewer. Most claims can be verified
in 3-4 searches + 1-2 file reads. If you find yourself searching repeatedly
without narrowing, stop and work with what you have -- "unsupported" with a
clear rationale is better than exhausting your turn budget.

## OUTPUT

Your verdict will be collected as structured data -- no need to format JSON
manually. Explain your reasoning in the rationale field: mention the specific
source document and what it says. If no source was found, explain what you
searched for.
- source_path: relative path within the corpus, or null
- source_url: URL of the web source, or null
- confidence: 0.9+ = source states this directly and unambiguously.
  0.5-0.7 = source implies it or requires connecting multiple documents.
  Below 0.5 = your best read without solid grounding."""

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
        claim.source_excerpt = data.get("source_excerpt", "")
        claim.human_review = data.get("human_review", True)
        claim.confidence = data.get("confidence")
    except KeyError as e:
        print(f"  [{claim.claim_id}] Missing field: {e}", file=sys.stderr)
        return agent_failure_result(claim)

    if claim.source_proximity == "original" and _is_summary_path(claim.source_path):
        claim.source_proximity = "derived"
        claim.human_review = True

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
                "context": c.context,
                "verdict": c.verdict,
                "source_proximity": c.source_proximity,
                "source_path": c.source_path,
                "source_url": c.source_url,
                "rationale": c.rationale,
                "source_excerpt": c.source_excerpt,
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


# ---- playbook helpers ----

_PLAYBOOK_CACHE = {}


def _get_playbook(name: str, playbook_dir: str):
    if name not in _PLAYBOOK_CACHE:
        from pipeline.playbook import load_playbook
        _PLAYBOOK_CACHE[name] = load_playbook(str(_Path(playbook_dir) / f"{name}.yaml"))
    return _PLAYBOOK_CACHE[name]


def _build_verification_prompt(playbook, article_summary: str, target_text: str, context: str) -> str:
    return (playbook.verification_prompt
            .replace("{{article_summary}}", article_summary)
            .replace("{{target_text}}", target_text)
            .replace("{{context}}", context))


# ---- async agent logic ----

async def _verify_claim_async(
    claim: Claim,
    corpus_root: str,
    system_prompt: str,
    timeout: int = 600,
    max_turns: int = 30,
    debug_dir: str | None = None,
    article_summary: str = "",
    allowed_tools: list[str] | None = None,
    output_schema: dict | None = None,
) -> Claim:
    from claude_agent_sdk import (
        query, ClaudeAgentOptions,
        AssistantMessage, TextBlock, ResultMessage,
        ProcessError, CLINotFoundError,
    )

    article_context = ""
    if article_summary:
        article_context = f"This claim is from an article about: {article_summary}\n\n"

    para_context = ""
    if claim.context:
        para_context = f"Surrounding paragraph from the article:\n> {claim.context}\n\n"

    from datetime import datetime as _dt
    today = _dt.now().strftime("%B %d, %Y")

    prompt = (
        f"Verify this claim:\n\n"
        f"{article_context}"
        f"{para_context}"
        f"Today's date: {today}\n\n"
        f"Claim: {claim.claim_text}\n\n"
        f"Claim type: {claim.claim_type}\n\n"
        f"Claim ID: {claim.claim_id}\n\n"
        f"If you fetch any web pages, save them to "
        f"web_cache/{claim.claim_id}/page.md for the audit trail.\n\n"
        f"Output fields: verdict, source_proximity, source_path, "
        f"source_url, rationale, human_review, confidence."
    )

    # Dynamic sandbox: deny access outside the corpus and /tmp.
    # The agent's cwd is corpus_root, so relative paths work for corpus access.
    # Explicit deny blocks the agent from wandering into /home, /etc, etc.
    import tempfile as _tempfile
    _settings = {
        "permissions": {
            "deny": [
                "Bash(/home/**)",
                "Bash(/etc/**)",
                "Bash(/usr/**)",
                "Bash(/var/**)",
                "Bash(/root/**)",
                "Bash(/opt/**)",
                "Bash(/sys/**)",
                "Bash(/proc/**)",
                "Read(/home/**)",
                "Read(/etc/**)",
                "Read(/usr/**)",
                "Read(/var/**)",
                "Read(/root/**)",
                "Read(/opt/**)",
                "Read(/sys/**)",
                "Read(/proc/**)",
            ]
        }
    }
    _sf = _tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(_settings, _sf)
    _sf.close()
    settings_path = _sf.name

    tools = allowed_tools if allowed_tools is not None else ["Read", "Bash", "WebSearch", "WebFetch"]
    schema = output_schema if output_schema is not None else VerdictOutput.model_json_schema()
    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        allowed_tools=tools,
        permission_mode="acceptEdits",
        cwd=corpus_root,
        max_turns=max_turns,
        output_format={"type": "json_schema", "schema": schema},
        settings=settings_path,
    )

    full_text = ""
    structured_output = None
    transcript: list[dict] = []  # Full message stream for debug

    result = agent_failure_result(claim)
    try:
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

        await asyncio.wait_for(_run(), timeout=timeout)

        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            with open(os.path.join(debug_dir, f"{claim.claim_id}.log"), "w") as f:
                f.write(full_text)
            with open(os.path.join(debug_dir, f"{claim.claim_id}.jsonl"), "w") as f:
                for entry in transcript:
                    f.write(json.dumps(entry, default=str) + "\n")

        if structured_output and isinstance(structured_output, dict):
            result = _populate_claim_from_dict(claim, structured_output)
            # Stash the raw output so callers using FindingOutput can extract
            # fields the old Claim model doesn't carry (recommended_action, metadata).
            result._raw_output = structured_output
        else:
            result = parse_verdict(claim, full_text)
    except asyncio.TimeoutError:
        print(f"  [{claim.claim_id}] Timed out after {timeout}s", file=sys.stderr)
    except CLINotFoundError:
        pass
    except ProcessError as e:
        print(f"  [{claim.claim_id}] Process error (exit {e.exit_code})", file=sys.stderr)
    except Exception as e:
        print(f"  [{claim.claim_id}] Error: {e}", file=sys.stderr)
    finally:
        os.unlink(settings_path)

    return result


async def _verify_target_async(
    target, system_prompt, allowed_tools, corpus_root,
    timeout, max_turns, debug_dir, article_summary,
):
    """Verify one target dict via a playbook's prompt+tools, returning a Finding or None."""
    from pipeline.models import Claim
    from pipeline.finding import Finding, Severity

    claim = Claim(
        claim_id=f"{target['playbook']}-{hash(target.get('target_text',''))}",
        claim_text=target["target_text"],
        source_quote=target["anchor_text"],
        claim_type=target.get("claim_type", "generalization"),
        context=target.get("context", ""),
    )

    result = await _verify_claim_async(
        claim, corpus_root, system_prompt, timeout, max_turns,
        debug_dir, article_summary,
        allowed_tools=allowed_tools,
        output_schema=FindingOutput.model_json_schema(),
    )

    if result.verdict is None:
        return None

    # Map old verdict to severity
    sev = Severity.PASS
    if result.verdict == "contradicted":
        sev = Severity.CRITICAL
    elif result.verdict == "unsupported":
        sev = Severity.WARNING

    # Fields from old Claim model
    rec_action = None
    meta = {}
    # New FindingOutput fields flow through _raw_output (stashed during struct output processing)
    raw = getattr(result, "_raw_output", None) or {}
    rec_action = raw.get("recommended_action")
    meta = raw.get("metadata", {})

    return Finding(
        finding_id=f"{target['playbook']}-{len(target.get('target_text',''))}",
        check_type=target["playbook"],
        severity=sev,
        target_text=target["target_text"],
        anchor_text=target["anchor_text"],
        context=target.get("context", ""),
        agent_summary=result.rationale or "",
        recommended_action=rec_action,
        source_path=result.source_path,
        source_url=result.source_url,
        source_excerpt=result.source_excerpt,
        confidence=result.confidence,
        human_review=result.human_review,
        metadata=meta,
    )


def _is_summary_path(path: str | None) -> bool:
    """Check if a source path appears to be a summary/overview rather than an original document."""
    if not path:
        return False
    summary_markers = [
        "_summary", "_FOLDER_SUMMARY", "CORPUS_OVERVIEW", "CORPUS_ROLLUP",
        "ALL_SUMMARIES", "INDEX.md",
    ]
    return any(m in path for m in summary_markers)


def _backfill_web_cache(claims: list[Claim], web_cache_dir: str) -> None:
    """For any claim with a source_url but no cached page, try to fetch it.

    Uses obscura if available, falls back to Jina, then curl.
    """
    import subprocess as _sp

    missing = [
        c for c in claims
        if c.source_url and c.source_url != "null"
        and not (os.path.isdir(os.path.join(web_cache_dir, c.claim_id)))
    ]
    if not missing:
        return

    print(f"\nBackfilling web cache for {len(missing)} claims...", file=sys.stderr)
    for c in missing:
        cid = c.claim_id
        url = c.source_url
        cache_dir = os.path.join(web_cache_dir, cid)
        os.makedirs(cache_dir, exist_ok=True)

        content = None
        method = None

        # Try obscura first
        if _sp.run(["which", "obscura"], capture_output=True).returncode == 0:
            try:
                r = _sp.run(
                    ["obscura", "fetch", url, "--dump", "markdown", "--timeout", "30"],
                    capture_output=True, text=True, timeout=60,
                )
                if r.returncode == 0 and len(r.stdout) > 100:
                    content = r.stdout
                    method = "obscura"
            except Exception:
                pass

        # Fall back to Jina
        if not content:
            try:
                import httpx
                r = httpx.get(f"https://r.jina.ai/{url}", timeout=30)
                if r.status_code == 200 and len(r.text) > 100:
                    content = r.text
                    method = "jina"
            except Exception:
                pass

        # Last resort: curl
        if not content:
            try:
                r = _sp.run(
                    ["curl", "-sL", url], capture_output=True, text=True, timeout=30,
                )
                if r.returncode == 0 and len(r.stdout) > 100:
                    content = r.stdout
                    method = "curl"
            except Exception:
                pass

        if content:
            with open(os.path.join(cache_dir, "page.md"), "w") as f:
                f.write(content)
            with open(os.path.join(cache_dir, "source.txt"), "w") as f:
                f.write(f"source_url: {url}\nfetch_method: {method}\n")
            print(f"  {cid}: cached via {method} ({len(content)} bytes)", file=sys.stderr)
        else:
            print(f"  {cid}: FAILED to cache {url}", file=sys.stderr)


def _populate_claim_from_dict(claim: Claim, data: dict) -> Claim:
    """Populate claim fields from a structured output dict. Falls back to
    agent_failure_result on missing required fields.

    Accepts both legacy (verdict) and new FindingOutput (severity) schemas.
    """
    # severity -> verdict mapping (new FindingOutput schema)
    if "verdict" not in data and "severity" in data:
        sev = data["severity"]
        data["verdict"] = {"PASS": "supported", "CRITICAL": "contradicted",
                           "WARNING": "unsupported"}.get(sev, "unsupported")
    if "source_proximity" not in data:
        data["source_proximity"] = "original" if data.get("source_path") else "unverifiable"
    try:
        claim.verdict = data["verdict"]
        claim.source_proximity = data["source_proximity"]
        claim.source_path = data.get("source_path")
        claim.source_url = data.get("source_url")
        claim.rationale = data.get("rationale") or data.get("agent_summary", "No rationale provided.")
        claim.source_excerpt = data.get("source_excerpt", "")
        claim.human_review = data.get("human_review", True)
        claim.confidence = data.get("confidence")
    except KeyError as e:
        print(f"  [{claim.claim_id}] Missing field in structured output: {e}", file=sys.stderr)
        return agent_failure_result(claim)

    # Mechanical safety net: if the agent cited a summary as "original",
    # downgrade to "derived" and flag for human review.
    if claim.source_proximity == "original" and _is_summary_path(claim.source_path):
        claim.source_proximity = "derived"
        claim.human_review = True
        print(
            f"  [{claim.claim_id}] Downgraded source_proximity original→derived "
            f"(summary path: {claim.source_path})",
            file=sys.stderr,
        )

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
    article_summary: str = "",
) -> Claim:
    """Verify a claim with one retry if the agent produced no verdict."""
    result = await _verify_claim_async(
        claim, corpus_root, system_prompt, timeout, max_turns, debug_dir, article_summary,
    )
    if result.verdict is not None:
        return result
    print(f"  Retrying {claim.claim_id}...", file=sys.stderr)
    return await _verify_claim_async(claim, corpus_root, system_prompt, timeout, max_turns, debug_dir, article_summary)


async def _verify_all(
    claims: list[Claim],
    corpus_root: str,
    system_prompt: str,
    concurrency: int,
    timeout: int = 600,
    max_turns: int = 30,
    output_path: str | None = None,
    debug_dir: str | None = None,
    article_summary: str = "",
) -> list[Claim]:
    import time as time_mod
    from tqdm import tqdm

    sem = asyncio.Semaphore(concurrency)
    total = len(claims)
    agent_times: list[float] = []
    results_by_id: dict[str, Claim] = {}

    pbar = tqdm(total=total, desc="  Agents", unit="claim")

    async def verify_one(claim: Claim) -> Claim:
        t0 = time_mod.time()
        async with sem:
            result = await _verify_claim_with_retry(
                claim, corpus_root, system_prompt, timeout, max_turns, debug_dir, article_summary,
            )
        elapsed = time_mod.time() - t0
        agent_times.append(elapsed)
        results_by_id[result.claim_id] = result

        status = (
            "✓" if result.verdict == "supported" else
            "✗" if result.verdict == "contradicted" else "?"
        )

        avg = sum(agent_times) / len(agent_times) if agent_times else 0
        remaining = total - pbar.n - 1
        eta_sec = int(avg * remaining / concurrency) if concurrency else 0
        eta_str = f"{eta_sec // 60}m{eta_sec % 60}s" if eta_sec > 0 else "—"

        pbar.set_postfix_str(
            f"{status} {claim.claim_id} {elapsed:.0f}s avg:{avg:.0f}s ETA:{eta_str}"
        )
        pbar.update(1)

        if output_path:
            _write_incremental(claims, results_by_id, output_path)

        return result

    results = await asyncio.gather(*[verify_one(c) for c in claims])
    pbar.close()
    return results


# ---- public entry point ----

def run_stage_b(
    claims_path: str = "",
    output_path: str = "",
    corpus_root: str = "",
    web_cache_dir: str = "",
    model: str = "",
    concurrency: int = 32,
    timeout: int = 600,
    max_turns: int = 30,
    debug_count: int = 0,
    targets_path: str = "",
    playbook_dir: str = "pipelines/",
) -> ClaimsDocument | FindingsDocument:
    """Load claims.json, verify each claim, write enriched claims.json.

    Args:
        claims_path: Path to claims.json from Stage A.
        output_path: Where to write enriched claims.json.
        corpus_root: Root of the local document corpus.
        concurrency: Max concurrent agent sessions (default 32).
        timeout: Per-agent timeout in seconds (default 600).
        max_turns: Max tool-calling turns per agent (default 30).
        debug_count: If >0, randomly sample N claims and save agent
                     output to debug/ directory alongside output_path.
        targets_path: Path to targets.json from new Stage A (playbook-driven).
        playbook_dir: Directory containing playbook YAML files.
    """
    if targets_path:
        import json as _json
        from pipeline.finding import FindingsDocument

        data = _json.loads(open(targets_path, encoding="utf-8").read())
        targets_list = data["targets"]
        summary = data.get("article_summary", "")
        article_file = data.get("article_file", "")

        debug_dir = None
        if debug_count > 0:
            debug_dir = os.path.join(os.path.dirname(output_path) or ".", "debug")
            print(f"Debug mode: logs -> {debug_dir}/", file=sys.stderr)

        print(f"Stage B: {len(targets_list)} targets. Model: {os.environ.get('ANTHROPIC_MODEL', '?')}", file=sys.stderr)

        async def _do():
            sem = asyncio.Semaphore(concurrency)

            async def _one(i, t):
                async with sem:
                    pb = _get_playbook(t["playbook"], playbook_dir)
                    prompt = _build_verification_prompt(pb, summary, t["target_text"], t.get("context", ""))
                    return await _verify_target_async(
                        t, prompt, pb.allowed_tools, corpus_root,
                        timeout, max_turns, debug_dir, summary,
                    )

            return await asyncio.gather(*[_one(i, t) for i, t in enumerate(targets_list)])

        results = asyncio.run(_do())
        findings_list = [r for r in results if r is not None]
        doc = FindingsDocument(article_file=article_file, article_summary=summary, findings=findings_list)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(doc.to_json())
        print(f"Stage B done: {len(findings_list)} findings -> {output_path}", file=sys.stderr)
        return doc

    # --- old path continues below unchanged ---
    import time as time_mod

    with open(claims_path, encoding="utf-8") as f:
        doc = ClaimsDocument.from_json(f.read())

    claims = doc.claims

    # Deduplicate by claim_text BEFORE verification — don't pay twice
    from collections import defaultdict
    text_groups: dict[str, list[Claim]] = defaultdict(list)
    for c in claims:
        text_groups[c.claim_text].append(c)
    dup_map: dict[str, str] = {}  # claim_id → representative claim_id
    unique_claims = []
    for text, group in text_groups.items():
        rep = group[0]
        unique_claims.append(rep)
        for c in group[1:]:
            dup_map[c.claim_id] = rep.claim_id
    if dup_map:
        print(f"Dedup: {len(claims)} claims → {len(unique_claims)} unique "
              f"({len(dup_map)} duplicates skipped)", file=sys.stderr)
    claims = unique_claims

    # Resume support: skip supported/contradicted, retry unsupported/untouched
    already_done = [c for c in claims if c.verdict in ("supported", "contradicted")]
    pending = [c for c in claims if c.verdict not in ("supported", "contradicted")]
    if already_done and pending:
        print(f"Resuming: {len(already_done)} done, {len(pending)} to retry "
              f"({sum(1 for c in pending if c.verdict == 'unsupported')} unsupported, "
              f"{sum(1 for c in pending if c.verdict is None)} untouched)", file=sys.stderr)
    claims = pending
    total = len(claims)

    if total == 0:
        print("All claims already verified — nothing to do.", file=sys.stderr)
        return doc

    debug_dir = None
    if debug_count > 0:
        import random
        claims = random.sample(claims, min(debug_count, len(claims)))
        total = len(claims)
        debug_dir = os.path.join(os.path.dirname(output_path) or ".", "debug")
        print(f"Debug mode: {total} claims, logs → {debug_dir}/", file=sys.stderr)

    # Ensure web_cache exists for agents to save fetched pages
    web_cache_dir = os.path.join(os.path.dirname(output_path) or ".", "web_cache")
    os.makedirs(web_cache_dir, exist_ok=True)

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
        article_summary=doc.article_summary,
    ))
    elapsed = time_mod.time() - t0

    # Merge results back into full claim set (including already-done claims)
    result_map = {r.claim_id: r for r in results}
    result_map.update({c.claim_id: c for c in already_done})

    # Propagate verdicts from verified representatives to skipped duplicates
    for dup_id, rep_id in dup_map.items():
        if rep_id in result_map:
            rep = result_map[rep_id]
            for c in doc.claims:
                if c.claim_id == dup_id:
                    c.verdict = rep.verdict
                    c.source_proximity = rep.source_proximity
                    c.source_path = rep.source_path
                    c.source_url = rep.source_url
                    c.rationale = rep.rationale
                    c.source_excerpt = rep.source_excerpt
                    c.human_review = rep.human_review
                    c.confidence = rep.confidence

    # Deduplicate by claim_text, merging verdicts intelligently.
    # Group by claim_text, then:
    #   - If all dupes agree on verdict: keep median-length rationale
    #   - If verdicts disagree: keep one representative per verdict
    from collections import defaultdict
    groups: dict[str, list[Claim]] = defaultdict(list)
    for c in doc.claims:
        if c.verdict is not None:
            groups[c.claim_text].append(c)

    deduped = []
    for text, dupes in groups.items():
        # Group by verdict within this claim_text
        by_verdict: dict[str, list[Claim]] = defaultdict(list)
        for c in dupes:
            by_verdict[c.verdict or "unsupported"].append(c)

        for verdict, vclaims in by_verdict.items():
            # Pick the one with median rationale length
            vclaims.sort(key=lambda c: len(c.rationale or ""))
            deduped.append(vclaims[len(vclaims) // 2])

    # Add untouched claims (verdict is None, not in any group)
    seen_texts = set(groups.keys())
    for c in doc.claims:
        if c.verdict is None and c.claim_text not in seen_texts:
            deduped.append(c)
            seen_texts.add(c.claim_text)

    # Re-number sequentially
    for i, c in enumerate(deduped):
        c.claim_id = f"c{i+1:04d}"

    doc.claims = deduped

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(doc.to_json())

    verified = [c for c in doc.claims if c.claim_id in result_map]
    unverified = [c for c in doc.claims if c.claim_id not in result_map]
    # Backfill missing web cache entries using obscura
    _backfill_web_cache(verified, web_cache_dir)

    supported = sum(1 for c in verified if c.verdict == "supported")
    contradicted = sum(1 for c in verified if c.verdict == "contradicted")
    unsupported = sum(1 for c in verified if c.verdict == "unsupported")
    review = sum(1 for c in verified if c.human_review)

    avg = elapsed / len(verified) if verified else 0
    print(f"\nDone: {supported} ✓ / {contradicted} ✗ / {unsupported} ?  ({review} flagged for review)")
    print(f"  {len(verified)} claims in {elapsed:.0f}s ({avg:.1f}s avg, ~{len(verified) / elapsed * 60:.0f}/min)")
    if unverified:
        print(f"  {len(unverified)} claims untouched (not in this run)")

    return doc
