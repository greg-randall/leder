"""The verification agent's in-process tools: validate_excerpt and fetch_page.

Stage-B agents run with cwd=corpus_root and must not reach outside it. Rather
than express that as CLI paths and permission-rule globs (which is what failed
before -- Bash(...) rules match command patterns, not paths), the tools and
their path containment live here, in Python we can unit-test without an agent.
A permission callback that reuses resolve_within() to police Read/Grep/Glob
arrives in a later task; _PATH_ARG below is the map it will key off.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

from claude_agent_sdk import McpSdkServerConfig, create_sdk_mcp_server, tool

from pipeline.tools.fetch_page import fetch_page
from pipeline.tools.validate_excerpt import validate_excerpt

# Tool name -> the input key naming the path it will touch. A tool absent from
# this map is one we have not reasoned about, so the permission callback
# (added in the next task) denies it.
_PATH_ARG = {"Read": "file_path", "Grep": "path", "Glob": "path"}


def _json_result(payload: dict, is_error: bool = False) -> dict:
    result = {"content": [{"type": "text", "text": json.dumps(payload)}]}
    if is_error:
        result["is_error"] = True
    return result


def resolve_within(root: Path, candidate: str | None) -> Path | None:
    """Resolve `candidate` against `root`; return None if it lands outside.

    Relative paths resolve against `root`; absolute paths are taken as-is and
    then checked. `.resolve()` collapses `..` and follows symlinks BEFORE the
    containment check, so neither can be used to escape.

    A missing candidate (None or "") means "no path given" -- e.g. a Grep/Glob
    call with no `path` argument, which searches the whole corpus -- so it
    resolves to the corpus root itself rather than being rejected.
    """
    if not candidate:
        return root.resolve()
    try:
        raw = Path(candidate)
        target = (raw if raw.is_absolute() else root / raw).resolve()
        root_resolved = root.resolve()
    except (OSError, RuntimeError, ValueError, TypeError):
        return None
    if target.is_relative_to(root_resolved):
        return target
    return None


def build_verification_tools(
    corpus_root: str, web_cache_dir: str, claim_id: str
) -> McpSdkServerConfig:
    """Build the per-claim in-process MCP server for a verification agent.

    corpus_root / web_cache_dir / claim_id are captured in the closure, so the
    agent supplies none of them -- it cannot get them wrong, and it cannot point
    the cache somewhere else.
    """
    root = Path(corpus_root)

    @tool(
        "validate_excerpt",
        "Confirm that a candidate excerpt really appears in a corpus document. "
        "Returns the document's ACTUAL text and its character offsets. Always "
        "report the returned actual_text as your source_excerpt -- never your "
        "own wording.",
        {
            "source_path": Annotated[
                str, "Path to the source .md file, relative to the corpus root."],
            "candidate_text": Annotated[
                str, "The text you believe supports the claim."],
        },
    )
    async def _validate_excerpt_tool(args: dict) -> dict:
        target = resolve_within(root, args["source_path"])
        if target is None:
            return _json_result(
                {"found": False,
                 "error": "source_path resolves outside the corpus"},
                is_error=True)
        try:
            result = await asyncio.to_thread(
                validate_excerpt, str(target), args["candidate_text"])
        except Exception as e:
            return _json_result(
                {"found": False, "error": f"{type(e).__name__}: {e}"},
                is_error=True)
        return _json_result(result)

    @tool(
        "fetch_page",
        "Fetch a web page and cache it for the audit trail, returning its text. "
        "The cache location is handled for you -- pass only the URL.",
        {"url": Annotated[str, "The URL to fetch."]},
    )
    async def _fetch_page_tool(args: dict) -> dict:
        # fetch_page is synchronous throughout (httpx, subprocess, Playwright
        # sync API). All agents share one event loop, so a blocking call here
        # would stall every other concurrent verification.
        #
        # fetch_page degrades rather than raising for fetch failures, but its
        # own mkdir/atomic-write can still raise on a full or unwritable disk --
        # an exception must become a tool error, not escape into the SDK.
        try:
            result = await asyncio.to_thread(
                fetch_page, args["url"], claim_id, web_cache_dir)
        except Exception as e:
            return {"content": [{"type": "text",
                                 "text": f"fetch failed: {type(e).__name__}: {e}"}],
                    "is_error": True}
        # fetch_page's contract (see its docstring) guarantees "content" is
        # always present; index directly rather than defaulting so a broken
        # contract fails loudly instead of handing the agent silent emptiness.
        blocks = [{"type": "text", "text": result["content"]}]
        method = result.get("method")
        if method:
            # Audit-relevant: e.g. "archive.is-raw" means the agent got raw
            # HTML rather than clean extracted markdown.
            blocks.append({"type": "text", "text": f"[fetched via {method}]"})
        warning = result.get("warning")
        if warning:
            # The paywall detector -- an agent citing this page needs to know
            # it may be reading a subscription teaser, not the full article.
            blocks.append({"type": "text", "text": f"WARNING: {warning}"})
        payload = {"content": blocks}
        if not result.get("ok"):
            payload["is_error"] = True
        return payload

    return create_sdk_mcp_server(
        name="leder", version="1.0.0",
        tools=[_validate_excerpt_tool, _fetch_page_tool],
    )
