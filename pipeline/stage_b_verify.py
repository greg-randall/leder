"""Stage B: Verify claims against the local corpus (with web fallback).

Each claim gets a real Claude Code agent session via the Claude Agent SDK.
Claims are verified in parallel with configurable concurrency.
Results are written incrementally and are crash-resistant.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sys

from typing import Optional

from pydantic import BaseModel, Field

from pathlib import Path as _Path

from pipeline.models import Claim

# The claude CLI prints this once per spawned subprocess whenever an API-key
# auth source (e.g. DeepSeek via ANTHROPIC_AUTH_TOKEN) takes precedence over
# a claude.ai login -- expected and harmless here since every agent uses
# API-key auth by design, but noisy at 40+ spawns per run.
_CLI_NOISE_SUBSTRINGS = ("claude.ai connectors are disabled",)


def _filter_agent_stderr(line: str) -> None:
    if any(s in line for s in _CLI_NOISE_SUBSTRINGS):
        return
    print(line, file=sys.stderr)


class FindingOutput(BaseModel):
    """Structured output schema for playbook-driven verification agents."""
    severity: str = Field(description="PASS, WARNING, or CRITICAL")
    agent_summary: str = Field(description="1-2 sentences explaining the finding")
    recommended_action: Optional[str] = Field(
        default=None,
        description=(
            "A concrete edit instruction for the article's editor, e.g. "
            "\"change 'per week' to 'per day'\" or \"add attribution: "
            "'according to testimony from...'\". Null when severity is PASS "
            "and no edit is needed."
        ),
    )
    source_path: Optional[str] = Field(
        default=None,
        description=(
            "Path to the source document, relative to the corpus root, exactly "
            "as it exists on disk (the converted .md file, never the raw source "
            "format). Null if no local source was found."
        ),
    )
    source_url: Optional[str] = Field(
        default=None, description="URL of the web source used, or null.",
    )
    source_excerpt: Optional[str] = Field(
        default=None,
        description=(
            "Actual text from the source file as returned by validate_excerpt. "
            "Guaranteed to be a verbatim substring of the source document. "
            "Populated from the tool's 'actual_text' field, never from the agent's own wording."
        ),
    )
    source_excerpt_offset: Optional[list[int]] = Field(
        default=None,
        description="[start_char, end_char] character positions in the source document, from validate_excerpt.",
    )
    source_excerpt_similarity: Optional[float] = Field(
        default=None,
        description=(
            "1.0 if exact match from validate_excerpt; lower if fuzzy-matched via Levenshtein. "
            "Null if validate_excerpt was not called or returned found=false. "
            "Informs confidence — a 0.62 match should lower confidence vs. a 1.0 match."
        ),
    )
    confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description=(
            "One of exactly 0.95, 0.8, 0.6, 0.4, or 0.2 per the confidence "
            "rubric -- how likely this severity judgment would survive human "
            "review, not whether the claim is true."
        ),
    )
    human_review: Optional[bool] = Field(
        default=None,
        description=(
            "True if this finding needs a human to double-check it before "
            "the article ships."
        ),
    )
    metadata: dict = Field(
        default_factory=dict,
        description=(
            "Structured extras: attribution_status "
            "(attributed_and_confirmed | stated_as_fact_testimony_only | "
            "corroborated | contradicted | not_found) and/or "
            "corpus_contradicted_by_external (bool)."
        ),
    )


_CONFIDENCE_BANDS = (0.95, 0.8, 0.6, 0.4, 0.2)


def _snap_confidence(value: float | None) -> float | None:
    """Snap a confidence value to the nearest of the five allowed bands.

    Belt-and-suspenders for the confidence rubric -- the DeepSeek endpoint
    isn't guaranteed to honor JSON-schema float enum constraints reliably,
    so this runs in code, not schema.
    """
    if value is None:
        return None
    # Round the delta before comparing -- raw float subtraction introduces
    # representation noise (e.g. abs(0.6-0.7) < abs(0.8-0.7) in raw floats
    # even though both are mathematically 0.1 away), which would silently
    # break the "first band wins on a tie" rule below. Rounding restores the
    # true tie so min() picks by _CONFIDENCE_BANDS iteration order.
    return min(_CONFIDENCE_BANDS, key=lambda band: round(abs(band - value), 9))


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
        claim.confidence = _snap_confidence(data.get("confidence"))
    except KeyError as e:
        print(f"  [{claim.claim_id}] Missing field: {e}", file=sys.stderr)
        return agent_failure_result(claim)

    if claim.source_proximity == "original" and _is_summary_path(claim.source_path):
        claim.source_proximity = "derived"
        claim.human_review = True

    if claim.confidence is not None and claim.confidence <= 0.6:
        claim.human_review = True

    return claim


# ---- playbook helpers ----

_PLAYBOOK_CACHE = {}


def _get_playbook(name: str, playbook_dir: str):
    if name not in _PLAYBOOK_CACHE:
        from pipeline.playbook import load_playbook
        _PLAYBOOK_CACHE[name] = load_playbook(str(_Path(playbook_dir) / f"{name}.yaml"))
    return _PLAYBOOK_CACHE[name]


def _build_verification_prompt(
    playbook, article_summary: str, target_text: str, context: str,
    corpus_description: str = "",
) -> str:
    from pipeline.prompts import build_verification_rules_block

    task_prompt = (playbook.verification_prompt
                   .replace("{{article_summary}}", article_summary)
                   .replace("{{target_text}}", target_text)
                   .replace("{{context}}", context))
    return build_verification_rules_block(corpus_description) + "\n\n" + task_prompt


# ---- async agent logic ----

async def _verify_claim_async(
    claim: Claim,
    corpus_root: str,
    system_prompt: str,
    timeout: int = 600,
    max_turns: int = 30,
    debug_dir: str | None = None,
    agent_log_dir: str | None = None,
    article_summary: str = "",
    allowed_tools: list[str] | None = None,
    output_schema: dict | None = None,
    web_cache_dir: str = "web_cache",
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

    schema = output_schema if output_schema is not None else FindingOutput.model_json_schema()
    schema_fields = ", ".join(schema.get("properties", {}).keys())

    prompt = (
        f"Verify this claim:\n\n"
        f"{article_context}"
        f"{para_context}"
        f"Today's date: {today}\n\n"
        f"Claim: {claim.claim_text}\n\n"
        f"Claim type: {claim.claim_type}\n\n"
        f"Output fields: {schema_fields}."
    )

    from pipeline.agent_tools import build_verification_tools, corpus_only_permission

    base_tools = ["Read", "Grep", "Glob", "WebSearch", "WebFetch"]
    mcp_tools = ["mcp__leder__validate_excerpt", "mcp__leder__fetch_page"]
    auto_allowed = ["WebSearch", "WebFetch"] + mcp_tools

    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        tools=base_tools + mcp_tools,
        allowed_tools=auto_allowed,
        permission_mode="default",
        can_use_tool=corpus_only_permission(corpus_root, web_cache_dir),
        mcp_servers={
            "leder": build_verification_tools(corpus_root, web_cache_dir, claim.claim_id),
        },
        strict_mcp_config=True,
        cwd=corpus_root,
        max_turns=max_turns,
        output_format={"type": "json_schema", "schema": schema},
        stderr=_filter_agent_stderr,
    )

    full_text = ""
    structured_output = None
    transcript: list[dict] = []  # Full message stream for debug

    result = agent_failure_result(claim)

    async def _prompt_stream():
        yield {
            "type": "user",
            "session_id": "",
            "message": {"role": "user", "content": prompt},
            "parent_tool_use_id": None,
        }

    try:
        async def _run():
            nonlocal full_text, structured_output
            async for message in query(prompt=_prompt_stream(), options=options):
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

        # Always-on agent log (timestamped run folder, created by run_stage_b)
        if agent_log_dir:
            os.makedirs(agent_log_dir, exist_ok=True)
            with open(os.path.join(agent_log_dir, f"{claim.claim_id}.log"), "w") as f:
                f.write(_build_log_text(transcript))
            with open(os.path.join(agent_log_dir, f"{claim.claim_id}.jsonl"), "w") as f:
                for entry in transcript:
                    f.write(json.dumps(entry, default=str) + "\n")
            with open(os.path.join(agent_log_dir, f"{claim.claim_id}.prompt.md"), "w") as f:
                f.write(f"# System Prompt\n\n{system_prompt}\n\n"
                        f"# User Prompt\n\n{prompt}\n")

        # Targeted debug log (--debug / --debug-ids, separate from always-on)
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            with open(os.path.join(debug_dir, f"{claim.claim_id}.log"), "w") as f:
                f.write(_build_log_text(transcript))
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

    return result


async def _verify_target_async(
    target, system_prompt, allowed_tools, corpus_root,
    timeout, max_turns, debug_dir, article_summary,
    agent_log_dir=None,
    web_cache_dir="web_cache",
):
    """Verify one target dict via a playbook's prompt+tools, returning a Finding or None."""
    from pipeline.models import Claim
    from pipeline.finding import Finding, Severity

    # sha1, not hash() -- hash() is randomized per-process (PYTHONHASHSEED),
    # which would defeat web_cache/{claim_id}/ reuse across separate runs.
    _text_hash = hashlib.sha1(target.get("target_text", "").encode("utf-8")).hexdigest()[:10]
    claim = Claim(
        claim_id=f"{target['playbook']}-{_text_hash}",
        claim_text=target["target_text"],
        source_quote=target["anchor_text"],
        claim_type=target.get("claim_type", "generalization"),
        context=target.get("context", ""),
    )

    result = await _verify_claim_async(
        claim, corpus_root, system_prompt, timeout, max_turns,
        debug_dir, agent_log_dir, article_summary,
        allowed_tools=allowed_tools,
        output_schema=FindingOutput.model_json_schema(),
        web_cache_dir=web_cache_dir,
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

    # Run the excerpt gate: re-verify the agent's quoted excerpt against the
    # actual document.
    gate = _apply_excerpt_gate(raw, corpus_root, web_cache_dir, claim.claim_id)

    agent_summary = result.rationale or ""
    if gate.get("note"):
        agent_summary = f"{agent_summary} {gate['note']}".strip()
    human_review = result.human_review
    if _is_summary_path(result.source_path):
        human_review = True
        agent_summary = (
            agent_summary + " ⚠ cites a summary, not an original "
            "— verify against the source document"
        ).strip()

    _text_hash = hashlib.sha1(target.get("target_text", "").encode("utf-8")).hexdigest()[:10]
    return Finding(
        finding_id=f"{target['playbook']}-{_text_hash}",
        check_type=target["playbook"],
        severity=sev,
        target_text=target["target_text"],
        anchor_text=target["anchor_text"],
        context=target.get("context", ""),
        agent_summary=agent_summary,
        recommended_action=rec_action,
        source_path=result.source_path,
        source_url=result.source_url,
        source_excerpt=gate.get("source_excerpt", result.source_excerpt),
        source_excerpt_offset=gate.get("source_excerpt_offset"),
        source_excerpt_similarity=gate.get("source_excerpt_similarity"),
        excerpt_status=gate.get("excerpt_status"),
        confidence=result.confidence,
        human_review=bool(human_review) or bool(gate.get("human_review")),
        metadata=meta,
    )


def _normalize_target_text(text: str) -> str:
    """Normalize target_text for near-duplicate grouping before stage-b dispatch."""
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    if normalized and normalized[-1] in ".?!":
        normalized = normalized[:-1]
    return normalized


def _is_summary_path(path: str | None) -> bool:
    """Check if a source path appears to be a summary/overview rather than an original document."""
    if not path:
        return False
    summary_markers = [
        "_summary", "_FOLDER_SUMMARY", "CORPUS_OVERVIEW", "CORPUS_ROLLUP",
        "ALL_SUMMARIES", "INDEX.md",
    ]
    return any(m in path for m in summary_markers)


def _apply_excerpt_gate(raw: dict, corpus_root: str, web_cache_dir: str,
                        claim_id: str) -> dict:
    """Re-verify the agent's reported source_excerpt against the document it cites.

    Runs whether or not the agent called the validate_excerpt tool: the tool is
    a convenience for the agent, this is the guarantee. Returns ONLY the fields
    to apply to the Finding, plus an optional "note" for agent_summary.

    Four outcomes, recorded in excerpt_status:
      exact      -- found verbatim; offset recorded
      repaired   -- found fuzzily; source_excerpt REPLACED with the document's
                    real wording, flagged for human review
      not_found  -- no match; excerpt dropped, flagged
      unchecked  -- no readable source to check against; nothing changed

    "unchecked" exists to separate "we checked and found nothing" from "we had
    nothing to check against". Collapsing both to a null offset is what let the
    original dropped-offset bug stay invisible.

    Deliberately does NOT touch severity or confidence. A paraphrased supporting
    quote does not by itself mean the verdict is wrong, and rewriting an
    editorial judgment from a string-match score would be a worse defect than
    the one this fixes.
    """
    from pipeline.agent_tools import resolve_within
    from pipeline.tools.validate_excerpt import validate_excerpt

    excerpt = str(raw.get("source_excerpt") or "").strip()
    source_path = raw.get("source_path")
    source_url = raw.get("source_url")

    target = None
    if source_path:
        # Agents sometimes prefix the path with the corpus folder name and/or
        # "./", occasionally both ("./corpus/doc.md", "corpus/./doc.md");
        # stage_d_sources.py strips the same prefixes when resolving sources.
        # Keep stripping while a known prefix remains at the front, stopping
        # the instant a pass strips nothing so a pathological input can't spin.
        while True:
            stripped = source_path
            for prefix in ("corpus/", "./"):
                if stripped.startswith(prefix):
                    stripped = stripped[len(prefix):]
                    break
            if stripped == source_path:
                break
            source_path = stripped
        target = resolve_within(_Path(corpus_root), source_path)
        if target is None:
            # A source_path that resolves outside the corpus is treated the
            # same as a missing file (see module docstring / task discussion):
            # excerpt_status stays "unchecked", never surfaced as a distinct
            # security signal on the Finding. Still worth a line in the run
            # output so an escape attempt isn't completely invisible.
            print(f"  ⚠ source_path escapes corpus, not read: {source_path!r}",
                  file=sys.stderr)
    elif source_url:
        target = _Path(web_cache_dir) / claim_id / "page.md"

    if not excerpt or target is None or not target.is_file():
        return {"excerpt_status": "unchecked"}

    result = validate_excerpt(str(target), excerpt)

    if not result.get("found"):
        return {
            "excerpt_status": "not_found",
            "source_excerpt": None,
            "source_excerpt_offset": None,
            "source_excerpt_similarity": None,
            "human_review": True,
            "note": "⚠ the quoted excerpt could not be located in the cited source",
        }

    similarity = result.get("similarity", 1.0)
    fields = {
        "source_excerpt": result["actual_text"],
        "source_excerpt_offset": result["offset"],
        "source_excerpt_similarity": similarity,
    }
    # similarity alone cannot distinguish the tiers: validate_excerpt's _clean()
    # strips punctuation before scoring, so a candidate differing only by an
    # apostrophe or comma scores 1.0 through the FUZZY tier. Compare the
    # agent's wording against what the document actually says to tell a
    # literal match from a normalised one -- tier 1's own match rule is
    # text.lower().find(candidate.lower()), so a genuine literal match always
    # satisfies excerpt.lower() == actual_text.lower().
    is_literal = excerpt.lower() == result["actual_text"].lower()
    if similarity >= 1.0 and is_literal:
        fields["excerpt_status"] = "exact"
    else:
        fields["excerpt_status"] = "repaired"
        fields["human_review"] = True
        fields["note"] = ("⚠ excerpt replaced with the source's actual wording "
                          f"(match {similarity:.2f})")
    return fields


def _summarize_web_cache(web_cache_dir: str) -> None:
    """Create minimal page_summary.md for each web_cache page so tiered search
    finds them on re-runs. Free — takes the first paragraph of each page
    as its summary, no LLM calls."""
    wc = _Path(web_cache_dir)
    if not wc.is_dir():
        return
    pages = sorted(wc.glob("*/page.md"))
    if not pages:
        return
    count = 0
    for page in pages:
        summary = page.parent / "page_summary.md"
        if summary.exists() and summary.stat().st_size > 10:
            continue
        text = page.read_text(encoding="utf-8", errors="replace").strip()
        # Grab first paragraph as the summary — cheap, effective for search
        first_para = text.split("\n\n")[0] if text else "(empty)"
        summary.write_text(
            f"**Summary:** Web-cached page. {first_para[:300]}\n\n"
            f"**Facts:** *(see page.md for full content)*\n",
            encoding="utf-8")
        count += 1
    if count:
        print(f"  Web cache: wrote {count} page_summary.md files for tiered search",
              file=sys.stderr)
    _write_web_cache_folder_summary(web_cache_dir)


def _write_web_cache_folder_summary(web_cache_dir: str) -> None:
    """Generate web_cache/_FOLDER_SUMMARY.md from all page_summary.md files.

    This puts web_cache on the tiered search map so agents can discover
    previously cached pages when verifying new claims.
    """
    wc = _Path(web_cache_dir)
    if not wc.is_dir():
        return
    summaries = sorted(wc.glob("*/page_summary.md"))
    if not summaries:
        return
    lines = [
        "# Web Cache — Previously Fetched Pages\n",
        f"**{len(summaries)} cached page(s)** from prior verification runs.\n",
        "Each entry below is a page fetched by a fact-checking agent. "
        "Check here first before re-fetching a URL.\n",
    ]
    for s in summaries:
        claim_dir = s.parent.name
        text = s.read_text(encoding="utf-8", errors="replace").strip()
        # Extract just the first paragraph as a preview
        preview = text.split("\n\n")[0] if text else "(empty)"
        lines.append(f"\n### {claim_dir}\n")
        lines.append(f"{preview}\n")
        lines.append(f"[open cached page]({claim_dir}/page.md)\n")
    (wc / "_FOLDER_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  Web cache: wrote _FOLDER_SUMMARY.md ({len(summaries)} entries)",
          file=sys.stderr)


def _populate_claim_from_dict(claim: Claim, data: dict) -> Claim:
    """Populate claim fields from a FindingOutput-shaped structured output
    dict. Falls back to agent_failure_result on missing required fields.

    Synthesizes claim.verdict from data["severity"] -- the shared Claim
    model (and stage-c's rendering) still key off verdict internally, even
    though only FindingOutput (severity-keyed) data ever arrives now.

    Fields the Claim model doesn't carry (recommended_action, metadata,
    source_excerpt_offset, source_excerpt_similarity) pass through via
    _raw_output, which the caller stashes from the full structured output dict.
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
        claim.confidence = _snap_confidence(data.get("confidence"))
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

    if claim.confidence is not None and claim.confidence <= 0.6:
        claim.human_review = True

    return claim


def _serialize_message(msg) -> dict:
    """Convert an SDK message to a plain dict for JSONL output.

    No truncation: the user wants to see what the agent was thinking, and
    a 2000-char thinking block truncated mid-sentence is useless. Every
    field that existed at the time of writing is captured in full.
    """
    result: dict = {"type": type(msg).__name__}
    _capture_optional(result, msg,
                      "subtype", "is_error", "num_turns", "session_id",
                      "model", "stop_reason", "message_id",
                      "parent_tool_use_id",
                      "total_cost_usd", "duration_ms",
                      "duration_api_ms", "usage")
    if hasattr(msg, "content") and msg.content:
        blocks = []
        for block in msg.content:
            b: dict = {"type": type(block).__name__}
            if hasattr(block, "text"):
                b["text"] = block.text
            if hasattr(block, "name"):
                b["name"] = block.name
            if hasattr(block, "input"):
                b["input"] = block.input
            if hasattr(block, "tool_use_id"):
                b["tool_use_id"] = block.tool_use_id
            if hasattr(block, "thinking"):
                b["thinking"] = block.thinking
            if hasattr(block, "signature"):
                b["signature"] = block.signature
            blocks.append(b)
        result["blocks"] = blocks
    if hasattr(msg, "result"):
        result["result"] = str(msg.result)
    if hasattr(msg, "structured_output"):
        result["structured_output"] = msg.structured_output
    return result


def _capture_optional(result: dict, msg, *fields: str) -> None:
    """Copy every *fields attr from `msg` into `result` when present and non-None."""
    for name in fields:
        if hasattr(msg, name):
            v = getattr(msg, name)
            if v is not None:
                result[name] = v


def _build_log_text(transcript: list[dict]) -> str:
    """Build a human-readable .log from the transcript.

    Includes text responses and thinking blocks in message order. For the
    full structured record (tool calls, results, metadata), see the .jsonl.
    """
    parts: list[str] = []
    for entry in transcript:
        for block in entry.get("blocks", []):
            if "text" in block:
                parts.append(block["text"])
            if "thinking" in block:
                parts.append(f"\n\n--- thinking ---\n{block['thinking']}\n")
    return "\n".join(parts)


# ---- public entry point ----

def _check_corpus_ready(corpus_root: str, web_cache_dir: str) -> list[str]:
    """Verify the corpus and web_cache have summaries before Stage B runs.

    Agents rely on the tiered search hierarchy (_FOLDER_SUMMARY.md →
    _summary.md → original). Without summaries, they waste tokens blindly
    grepping or miss documents entirely.

    Returns a list of human-readable issue strings. Empty list = ready.
    """
    issues: list[str] = []
    cr = _Path(corpus_root)

    # Check 1: Main corpus has at least one folder summary or overview
    has_overview = (cr / "CORPUS_OVERVIEW.md").exists()
    has_folder_summaries = any(cr.rglob("_FOLDER_SUMMARY.md"))
    if not has_overview and not has_folder_summaries:
        issues.append(
            "Corpus has no _FOLDER_SUMMARY.md files and no CORPUS_OVERVIEW.md. "
            "Run:  python3 -m pipeline.cli prepare-2 && python3 -m pipeline.cli prepare-3"
        )

    # Check 2: web_cache has pages but no folder summary → agents can't find them
    wc = _Path(web_cache_dir)
    if wc.is_dir():
        has_cached_pages = any(wc.glob("*/page.md"))
        has_wc_folder = (wc / "_FOLDER_SUMMARY.md").exists()
        if has_cached_pages and not has_wc_folder:
            issues.append(
                "Web cache has cached pages but no _FOLDER_SUMMARY.md. "
                "Re-run stage-b (which generates it automatically) or run prepare-3."
            )

    return issues


def run_stage_b(
    targets_path: str,
    output_path: str = "",
    corpus_root: str = "",
    web_cache_dir: str = "",
    model: str = "",
    concurrency: int = 32,
    timeout: int = 600,
    max_turns: int = 30,
    debug_count: int = 0,
    playbook_dir: str = "pipelines/",
    pricing: dict | None = None,
    debug_ids: list[int] | None = None,
    force_run: bool = False,
    corpus_description: str = "",
    agent_log_dir: str | None = None,
) -> FindingsDocument:
    """Load targets.json, verify each target, write findings.json.

    Args:
        targets_path: Path to targets.json from Stage A. Required.
        output_path: Where to write findings.json.
        corpus_root: Root of the local document corpus.
        concurrency: Max concurrent agent sessions (default 32).
        timeout: Per-agent timeout in seconds (default 600).
        max_turns: Max tool-calling turns per agent (default 30).
        debug_count: If >0, randomly sample N targets and save agent
                     output to debug/ directory alongside output_path.
        playbook_dir: Directory containing playbook YAML files.
        force_run: Skip the corpus readiness check.
        corpus_description: Domain description injected into the shared
                             verification rules block (see pipeline.prompts).
    """
    if not targets_path:
        raise ValueError("targets_path is required")

    # Compute web_cache_dir once, before it's needed below.
    if not web_cache_dir:
        web_cache_dir = os.path.join(corpus_root, "web_cache")

    # Pre-flight: corpus must have summaries unless --force-run
    if not force_run:
        issues = _check_corpus_ready(corpus_root, web_cache_dir)
        if issues:
            print("\nERROR: Corpus not ready for Stage B verification.\n",
                  file=sys.stderr)
            for issue in issues:
                print(f"  • {issue}", file=sys.stderr)
            print("\nRe-run with --force-run to skip this check.\n",
                  file=sys.stderr)
            sys.exit(1)

    if not os.path.exists(targets_path):
        print(f"ERROR: {targets_path} not found. Run stage-a first to generate it.",
              file=sys.stderr)
        sys.exit(1)

    import json as _json
    from pipeline.finding import Finding, FindingsDocument

    data = _json.loads(open(targets_path, encoding="utf-8").read())
    targets_list = data["targets"]
    if debug_ids:
        targets_list = [t for i, t in enumerate(targets_list) if i in debug_ids]
        print(f"Stage B: {len(targets_list)} targets (debug-ids filter)",
              file=sys.stderr)

    from collections import OrderedDict as _OrderedDict
    _groups: "_OrderedDict[tuple, list]" = _OrderedDict()
    for t in targets_list:
        key = (t.get("playbook", ""), _normalize_target_text(t.get("target_text", "")))
        _groups.setdefault(key, []).append(t)
    representatives = [members[0] for members in _groups.values()]
    _dup_count = len(targets_list) - len(representatives)
    if _dup_count:
        print(f"Stage B: dedup {len(targets_list)} targets -> {len(representatives)} "
              f"unique ({_dup_count} near-duplicate(s) fanned out from verified "
              f"representatives)", file=sys.stderr)

    summary = data.get("article_summary", "")
    article_file = data.get("article_file", "")

    base_debug_dir = os.path.join(os.path.dirname(output_path) or ".", "debug")
    if debug_count > 0 or debug_ids:
        os.makedirs(base_debug_dir, exist_ok=True)
        if debug_ids:
            print(f"Debug mode: logging targets {debug_ids} -> {base_debug_dir}/",
                  file=sys.stderr)
        else:
            print(f"Debug mode: logging {debug_count} random targets -> "
                  f"{base_debug_dir}/", file=sys.stderr)

    # Always-on agent logging: timestamped subfolder so logs don't pile up
    # in one flat directory across runs.
    actual_agent_log_dir = None
    if agent_log_dir:
        from datetime import datetime as _dt2
        run_ts = _dt2.now().strftime("%Y%m%d-%H%M%S")
        actual_agent_log_dir = os.path.join(agent_log_dir, run_ts)
        os.makedirs(actual_agent_log_dir, exist_ok=True)
        print(f"Agent log: {actual_agent_log_dir}/", file=sys.stderr)

    print(f"Stage B: {len(representatives)} targets to verify. "
          f"Model: {os.environ.get('ANTHROPIC_MODEL', '?')}", file=sys.stderr)

    from tqdm import tqdm
    import threading as _thr
    pbar = tqdm(total=len(representatives), desc="  Stage B agents", unit="target")
    findings_lock = _thr.Lock()
    findings_by_id: dict[str, Finding] = {}

    tmp_path = output_path + ".tmp"

    def _save_findings():
        with findings_lock:
            doc = FindingsDocument(
                article_file=article_file, article_summary=summary,
                findings=list(findings_by_id.values()),
            )
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(doc.to_json())

    async def _do():
        from dataclasses import replace as _replace
        sem = asyncio.Semaphore(concurrency)

        async def _one(i, t):
            # NOTE: i/t iterate over `representatives` (post-dedup), not the
            # original targets_list -- --debug-ids/--debug N indices no longer
            # correspond 1:1 to targets.json's original target order once
            # near-duplicate targets have been collapsed to one representative.
            async with sem:
                # Per-target debug: debug_ids (explicit indices) or random sample
                t_debug_dir = None
                if debug_ids:
                    if i in debug_ids:
                        t_debug_dir = base_debug_dir
                elif debug_count > 0 and i < debug_count:
                    t_debug_dir = base_debug_dir
                pb = _get_playbook(t["playbook"], playbook_dir)
                prompt = _build_verification_prompt(
                    pb, summary, t["target_text"], t.get("context", ""),
                    corpus_description=corpus_description)
                finding = await _verify_target_async(
                    t, prompt, pb.allowed_tools, corpus_root,
                    timeout, max_turns, t_debug_dir, summary,
                    agent_log_dir=actual_agent_log_dir,
                    web_cache_dir=web_cache_dir,
                )
                if finding is not None:
                    with findings_lock:
                        findings_by_id[finding.finding_id] = finding
                        key = (t.get("playbook", ""), _normalize_target_text(t.get("target_text", "")))
                        for n, member in enumerate(_groups[key][1:], start=1):
                            clone = _replace(
                                finding,
                                finding_id=f"{finding.finding_id}-a{n}",
                                anchor_text=member.get("anchor_text", finding.anchor_text),
                                context=member.get("context", finding.context),
                            )
                            findings_by_id[clone.finding_id] = clone
                    _save_findings()
                pbar.update(1)
                return finding

        return await asyncio.gather(*[_one(i, t) for i, t in enumerate(representatives)])

    asyncio.run(_do())
    pbar.close()
    findings_list = list(findings_by_id.values())
    doc = FindingsDocument(article_file=article_file, article_summary=summary, findings=findings_list)
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(doc.to_json())
    os.replace(tmp_path, output_path)  # atomic rename
    print(f"Stage B done: {len(findings_list)} findings -> {output_path}", file=sys.stderr)
    _summarize_web_cache(web_cache_dir)
    # Cost estimate (pricing from config.yaml, rates per 1M tokens)
    rates = (pricing or {}).get(
        os.environ.get("ANTHROPIC_MODEL", "deepseek-v4-pro"),
        {"input": 0.435, "output": 0.87},  # fallback to pro pricing
    )
    try:
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")
        sys_tokens = 3700  # playbook prompt + injected article_summary
        prompt_tokens = sum(
            sys_tokens +
            len(enc.encode(t.get("target_text", ""))) +
            len(enc.encode(t.get("context", "")))
            for t in targets_list
        )
        out_tokens = len(findings_list) * 2500  # agent response + tool-call log
        cost = (prompt_tokens / 1_000_000 * rates["input"] +
                out_tokens / 1_000_000 * rates["output"])
        print(f"  Cost estimate: ${cost:.3f}  "
              f"({prompt_tokens:,} in / {out_tokens:,} out tokens)", file=sys.stderr)
    except ImportError:
        print(f"  Cost estimate: unavailable (tiktoken not installed)", file=sys.stderr)
    return doc
