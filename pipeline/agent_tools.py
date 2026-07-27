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
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class _ToolSpec:
    """How one tool can reach the filesystem.

    path_arg       The input key naming a location, checked via
                    resolve_within. Always a real key name -- there is no
                    "this tool has no path" shape, because that used to be
                    spelled path_arg=None, and a spec of
                    (path_arg=None, pattern_args=()) matches neither the path
                    branch nor any pattern, so it fell through to Allow no
                    matter what tool_input held. A tool with no path concept
                    simply cannot be added here; it stays outside _TOOL_SPEC
                    and is denied by the "unrecognised tool" branch.
    path_required   True: a missing path_arg denies outright (Read.file_path
                    is mandatory). False: a missing path_arg means "the
                    corpus root" (Grep/Glob's path is genuinely optional).
    pattern_args    Input keys holding glob-style strings -- a SECOND route to
                    the filesystem, independent of path_arg (Glob(pattern=
                    "../*.md") escapes corpus_root even though Glob's `path`
                    argument was never touched). Defaults to () via a
                    dataclass field, not a dict .get(..., ()) -- a spec that
                    forgets this field now uses the explicit default rather
                    than raising KeyError deep inside an async callback.
                    Grep's `pattern` is a search regex, not a filesystem
                    pattern, so it is deliberately absent for Grep.
    """
    path_arg: str
    path_required: bool
    pattern_args: tuple[str, ...] = field(default_factory=tuple)


# Tool name -> _ToolSpec. A tool absent from this map is one we have not
# reasoned about, so the permission callback denies it.
_TOOL_SPEC: dict[str, _ToolSpec] = {
    "Read": _ToolSpec(path_arg="file_path", path_required=True),
    "Grep": _ToolSpec(path_arg="path", path_required=False, pattern_args=("glob",)),
    "Glob": _ToolSpec(path_arg="path", path_required=False, pattern_args=("pattern",)),
}

# Fail fast at import time rather than per-request: an empty path_arg is
# exactly the "no path route" shape that used to fall through to a blanket
# allow, so a future entry that omits it must never reach _can_use_tool.
for _tool_name, _spec in _TOOL_SPEC.items():
    assert _spec.path_arg, f"_TOOL_SPEC[{_tool_name!r}].path_arg must be a non-empty key name"
del _tool_name, _spec


def _json_result(payload: dict, is_error: bool = False) -> dict:
    result = {"content": [{"type": "text", "text": json.dumps(payload)}]}
    if is_error:
        result["is_error"] = True
    return result


def resolve_within(
    root: Path, candidate: str | None, *, root_resolved: Path | None = None
) -> Path | None:
    """Resolve `candidate` against `root`; return None if it lands outside.

    Relative paths resolve against `root`; absolute paths are taken as-is and
    then checked. `.resolve()` collapses `..` and follows symlinks BEFORE the
    containment check, so neither can be used to escape.

    A missing candidate (None or "") means "no path given" -- e.g. a Grep/Glob
    call with no `path` argument, which searches the whole corpus -- so it
    resolves to the corpus root itself rather than being rejected.

    `root.resolve()` is a real filesystem syscall (measured ~2-4ms on a DrvFs
    mount) -- expensive to repeat on every call when a caller such as
    corpus_only_permission already resolved `root` once per agent and holds
    onto it. Pass that value as `root_resolved` to skip resolving it again;
    omit it (the default) and this function resolves `root` itself, which is
    what every existing caller/test relies on.
    """
    try:
        resolved_root = root_resolved if root_resolved is not None else root.resolve()
        if candidate is None or candidate == "":
            return resolved_root
        raw = Path(candidate)
        target = (raw if raw.is_absolute() else root / raw).resolve()
    except (OSError, RuntimeError, ValueError, TypeError):
        return None
    if target.is_relative_to(resolved_root):
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


def _has_tilde_prefix(value: str) -> bool:
    """True if a path/pattern begins with '~'.

    Shared by both the path route and the pattern route: Path() never expands
    '~', so resolve_within's containment check treats "~/.ssh/id_rsa" as a
    literal (and contained) subdirectory name -- correct only if the real
    tool does no tilde expansion of its own, which we cannot verify here.
    Nothing in a document corpus is legitimately addressed with a leading
    '~', so both routes deny it outright rather than trust that assumption.
    """
    return value.startswith("~")


def _pattern_escapes(pattern: str) -> bool:
    """True if a glob-style pattern reaches outside the corpus on its own,
    independent of whatever the tool's `path` argument says.

    A legitimate corpus search never needs an absolute prefix, a leading `~`,
    a `..` path segment, or brace/bracket/backslash syntax -- Glob/Grep's own
    `path` argument (already run through resolve_within) is how you scope a
    search to a subdirectory. This is a conservative, string-level check
    rather than a resolve_within-style filesystem check: we cannot verify on
    this machine whether the real Glob/Grep implementations interpret
    `..`/absolute segments inside a pattern as filesystem traversal, so we
    deny the shapes that WOULD be a traversal if they are interpreted that
    way, rather than assume they are safe.

    The brace/bracket/backslash block exists because a plain substring check
    for ".." is not the standard being held to above: minimatch (which npm
    `glob`, and therefore almost certainly the Glob tool, is built on) does
    brace expansion by default, and `[.]` is a character class matching a
    literal dot -- so "{..,.}/x", "[.][.]/ x", "[.]./x", and ".[.]/x" are the
    same traversal spelled around a bare ".." check. Rather than reimplement
    brace/bracket expansion here to see whether a given instance actually
    decodes to "..", any pattern that could carry that meaning is refused.
    """
    if pattern.startswith("/") or _has_tilde_prefix(pattern):
        return True
    if any(c in pattern for c in "{[\\"):
        return True
    return ".." in pattern.split("/")


def _resolves_within_any(candidate_roots: list[Path], raw: str | None) -> bool:
    """The filesystem-touching part of a permission decision: does `raw`
    resolve inside any of `candidate_roots` (all already-resolved Path
    objects)? Runs entirely off the event loop via asyncio.to_thread --
    Path.resolve() is a real syscall (measured several ms on a DrvFs mount),
    and this callback is on the loop shared by every concurrent agent.

    An absolute `raw` resolves to the same target regardless of which root
    it's being checked against, so it is resolved once here and then checked
    against every candidate root for free (is_relative_to is pure Python, no
    syscall) -- rather than resolve_within's one-resolve-per-call cost
    repeated once per root in a loop. Relative/missing candidates only ever
    have a single candidate root (corpus_root -- see the comment where
    candidate_roots is built), so that duplication can't arise for them.
    """
    if raw and Path(raw).is_absolute():
        try:
            target = Path(raw).resolve()
        except (OSError, RuntimeError, ValueError, TypeError):
            return False
        return any(target.is_relative_to(root) for root in candidate_roots)
    root = candidate_roots[0]
    return resolve_within(root, raw, root_resolved=root) is not None


def corpus_only_permission(corpus_root: str, web_cache_dir: str):
    """Build a can_use_tool callback that confines filesystem reads to the corpus.

    Only reached for tools NOT in allowed_tools -- the SDK does not invoke
    can_use_tool for auto-approved tools, which is exactly why Read/Grep/Glob
    are deliberately left out of allowed_tools in stage B.

    Default-deny: a tool absent from _TOOL_SPEC is one whose filesystem reach
    we have not reasoned about, so it is refused rather than waved through.

    Accepted, not fixed: the symlink asymmetry between the path route (which
    resolves through resolve_within/Path.resolve(), so a symlink pointing
    outside the corpus is denied) and the pattern route (a string-level check
    only, per _pattern_escapes -- it does not resolve what a matched symlink
    would point to). Closing that would mean resolving every glob match, and
    it requires a symlink already present in the corpus, which the agent has
    no way to create -- it has no write tool.
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
        for pattern_arg in spec.pattern_args:
            pattern_val = tool_input.get(pattern_arg)
            if pattern_val is None:
                continue
            if not isinstance(pattern_val, str):
                return PermissionResultDeny(
                    message=f"{pattern_arg} must be a string.")
            if _pattern_escapes(pattern_val):
                return PermissionResultDeny(
                    message=(f"{pattern_arg} must not be absolute, start "
                             f"with '~', or contain '..', '{{', '[', or '\\\\'."))

        raw = tool_input.get(spec.path_arg)
        if spec.path_required and raw is None:
            return PermissionResultDeny(
                message=f"{spec.path_arg} is required.")
        if raw is not None and not isinstance(raw, str):
            return PermissionResultDeny(
                message=f"{spec.path_arg} must be a string.")
        if isinstance(raw, str) and _has_tilde_prefix(raw):
            return PermissionResultDeny(
                message=f"{spec.path_arg} must not begin with '~'.")

        # A relative (or missing) path resolves against the *process* cwd,
        # which the SDK sets to corpus_root -- so only corpus_root's
        # containment applies. Checking it against web_cache_dir too would let
        # a corpus-root symlink escape slip through on a coincidental
        # non-existent match under web_cache_dir (resolve() does not raise for
        # path components that don't exist, so a bogus nested path still
        # counts as "inside" that root). Absolute paths carry their own
        # location and may legitimately land in any allowed root.
        candidate_roots = roots if (raw and Path(raw).is_absolute()) else [corpus_resolved]
        if await asyncio.to_thread(_resolves_within_any, candidate_roots, raw):
            return PermissionResultAllow()

        return PermissionResultDeny(
            message=(f"{raw} is outside the corpus. Use paths relative to the "
                     f"corpus root."))

    return _can_use_tool
