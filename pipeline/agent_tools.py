"""The verification agent's in-process tools: validate_excerpt and fetch_page.

Stage-B agents run with cwd=corpus_root and must not reach outside it. Rather
than express that as CLI paths and permission-rule globs (which is what failed
before -- Bash(...) rules match command patterns, not paths), the tools and
their path containment live here, in Python we can unit-test without an agent.
corpus_only_permission() below is the can_use_tool callback that polices
Read/Grep/Glob using resolve_within(); _TOOL_SPEC is the map it keys off.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

from claude_agent_sdk import (
    McpSdkServerConfig,
    PermissionResultAllow,
    PermissionResultDeny,
    create_sdk_mcp_server,
    tool,
)

from pipeline.tools.fetch_page import fetch_page
from pipeline.tools.validate_excerpt import validate_excerpt

# Tool name -> how it can reach the filesystem:
#   "path"     the single input key naming a location, checked via
#              resolve_within (None if the tool has no such key).
#   "patterns" input keys holding glob-style strings that are a SECOND route
#              to the filesystem, independent of "path" -- Glob's `pattern`
#              and Grep's `glob` are matched against files on disk in their
#              own right (e.g. Glob(pattern="../*.md") escapes corpus_root
#              even though Glob's `path` argument was never touched). Grep's
#              `pattern` is a search regex, not a filesystem pattern, so it is
#              deliberately absent here.
# A tool absent from this map is one we have not reasoned about, so the
# permission callback denies it.
_TOOL_SPEC = {
    "Read": {"path": "file_path", "patterns": ()},
    "Grep": {"path": "path", "patterns": ("glob",)},
    "Glob": {"path": "path", "patterns": ("pattern",)},
}


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
    try:
        if candidate is None or candidate == "":
            return root.resolve()
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
        # An "error" key (missing file, empty candidate_text) means validate_excerpt
        # couldn't even attempt the check -- a usage problem the agent must see AS
        # an error. A bare {"found": False} with no "error" key is a legitimate
        # negative result (the text just isn't there) and stays a normal reply.
        return _json_result(result, is_error=bool(result.get("error")))

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


def _pattern_escapes(pattern: str) -> bool:
    """True if a glob-style pattern reaches outside the corpus on its own,
    independent of whatever the tool's `path` argument says.

    A legitimate corpus search never needs an absolute prefix, a leading `~`,
    or a `..` path segment -- Glob/Grep's own `path` argument (already run
    through resolve_within) is how you scope a search to a subdirectory. This
    is a conservative, string-level check rather than a resolve_within-style
    filesystem check: we cannot verify on this machine whether the real
    Glob/Grep implementations interpret `..`/absolute segments inside a
    pattern as filesystem traversal, so we deny the shapes that WOULD be a
    traversal if they are interpreted that way, rather than assume they are
    safe.
    """
    if pattern.startswith("/") or pattern.startswith("~"):
        return True
    return ".." in pattern.split("/")


def corpus_only_permission(corpus_root: str, web_cache_dir: str):
    """Build a can_use_tool callback that confines filesystem reads to the corpus.

    Only reached for tools NOT in allowed_tools -- the SDK does not invoke
    can_use_tool for auto-approved tools, which is exactly why Read/Grep/Glob
    are deliberately left out of allowed_tools in stage B.

    Default-deny: a tool absent from _TOOL_SPEC is one whose filesystem reach
    we have not reasoned about, so it is refused rather than waved through.
    """
    corpus_resolved = Path(corpus_root).resolve()
    roots = [corpus_resolved]
    web_cache = Path(web_cache_dir).resolve()
    if web_cache not in roots:
        roots.append(web_cache)

    async def _can_use_tool(tool_name: str, tool_input: dict, context):
        spec = _TOOL_SPEC.get(tool_name)
        if spec is None:
            return PermissionResultDeny(
                message=f"{tool_name} is not available to verification agents.")

        # Second route to the filesystem: Glob's `pattern` and Grep's `glob`
        # are matched against files on disk independently of the `path`
        # argument below, so a clean `path` doesn't clear them.
        for pattern_arg in spec["patterns"]:
            pattern_val = tool_input.get(pattern_arg)
            if pattern_val is None:
                continue
            if not isinstance(pattern_val, str):
                return PermissionResultDeny(
                    message=f"{pattern_arg} must be a string.")
            if _pattern_escapes(pattern_val):
                return PermissionResultDeny(
                    message=(f"{pattern_arg} must not be absolute, start "
                             f"with '~', or contain a '..' segment."))

        path_arg = spec["path"]
        raw = tool_input.get(path_arg) if path_arg else None
        if raw is not None and not isinstance(raw, str):
            return PermissionResultDeny(
                message=f"{path_arg} must be a string.")
        if isinstance(raw, str) and raw.startswith("~"):
            # resolve_within would happily contain "<corpus_root>/~/..." --
            # correct IF Read does no tilde expansion of its own, which we
            # cannot verify here. Nothing in the corpus is legitimately
            # addressed with a leading '~', so deny it outright rather than
            # trust that assumption.
            return PermissionResultDeny(
                message=f"{path_arg} must not begin with '~'.")

        # A relative (or missing) path resolves against the *process* cwd,
        # which the SDK sets to corpus_root -- so only corpus_root's
        # containment applies. Checking it against web_cache_dir too would let
        # a corpus-root symlink escape slip through on a coincidental
        # non-existent match under web_cache_dir (resolve() does not raise for
        # path components that don't exist, so a bogus nested path still
        # counts as "inside" that root). Absolute paths carry their own
        # location and may legitimately land in any allowed root.
        candidate_roots = roots if (raw and Path(raw).is_absolute()) else [corpus_resolved]
        for root in candidate_roots:
            if resolve_within(root, raw) is not None:
                return PermissionResultAllow()

        return PermissionResultDeny(
            message=(f"{raw} is outside the corpus. Use paths relative to the "
                     f"corpus root."))

    return _can_use_tool
