"""Tests for pipeline/agent_tools.py -- the verification agent's reachable surface."""
import asyncio
import inspect
import json

import mcp.types as types

from pipeline.tools.fetch_page import fetch_page as real_fetch_page


async def _call(server, name: str, args: dict) -> types.CallToolResult:
    """Invoke a tool the same way the SDK does: through the server's own
    CallToolRequest handler, not by reaching for the underlying SdkMcpTool
    object.

    `create_sdk_mcp_server` (claude_agent_sdk 0.2.127) does not expose the
    original SdkMcpTool list anywhere public -- it lives only as a closure
    variable of this handler. Going through the handler instead of digging
    out that closure is strictly better: `request_handlers` is a genuine
    public attribute of mcp.server.lowlevel.Server, it cannot silently
    retarget to the wrong object, and it exercises the jsonschema validation
    and the is_error -> isError conversion our handlers rely on -- both of
    which a direct `.handler(args)` call would bypass.
    """
    handler = server["instance"].request_handlers[types.CallToolRequest]
    request = types.CallToolRequest(
        method="tools/call",
        params=types.CallToolRequestParams(name=name, arguments=args))
    result = await handler(request)
    return result.root


def _call_sync(server, name: str, args: dict) -> types.CallToolResult:
    return asyncio.run(_call(server, name, args))


def _json(result: types.CallToolResult) -> dict:
    return json.loads(result.content[0].text)


def _build(tmp_path, claim_id="c1"):
    from pipeline.agent_tools import build_verification_tools
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    wc = corpus / "web_cache"
    wc.mkdir()
    server = build_verification_tools(str(corpus), str(wc), claim_id)
    return corpus, wc, server


# ---------------------------------------------------------------- resolve_within


def test_resolve_within_symlinked_file_escape(tmp_path):
    from pipeline.agent_tools import resolve_within
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    secret = tmp_path / "secret.md"
    secret.write_text("secret")
    (corpus / "link.md").symlink_to(secret)
    assert resolve_within(corpus, "link.md") is None


def test_resolve_within_symlinked_dir_escape(tmp_path):
    from pipeline.agent_tools import resolve_within
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "o.md").write_text("secret")
    (corpus / "linkdir").symlink_to(outside, target_is_directory=True)
    assert resolve_within(corpus, "linkdir/o.md") is None


def test_resolve_within_missing_candidate_returns_corpus_root(tmp_path):
    from pipeline.agent_tools import resolve_within
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    assert resolve_within(corpus, None) == corpus.resolve()
    assert resolve_within(corpus, "") == corpus.resolve()


def test_resolve_within_non_string_candidate_does_not_raise(tmp_path):
    from pipeline.agent_tools import resolve_within
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    assert resolve_within(corpus, 123) is None
    assert resolve_within(corpus, ["a", "b"]) is None


# --------------------------------------------------------------- validate_excerpt


def test_validate_excerpt_tool_resolves_relative_to_corpus(tmp_path):
    corpus, _, server = _build(tmp_path)
    (corpus / "doc.md").write_text("The road commission issued a 68-page notice.")
    args = {"source_path": "doc.md", "candidate_text": "road commission issued"}
    out = _json(_call_sync(server, "validate_excerpt", args))
    assert out["found"] is True
    assert out["similarity"] == 1.0
    assert out["actual_text"] == "road commission issued"


def test_validate_excerpt_tool_rejects_path_escape(tmp_path):
    corpus, _, server = _build(tmp_path)
    (tmp_path / "secret.md").write_text("the road commission issued a 68-page notice")
    args = {"source_path": "../secret.md", "candidate_text": "road commission issued"}
    out = _json(_call_sync(server, "validate_excerpt", args))
    assert out["found"] is False
    assert "outside the corpus" in out["error"]


def test_validate_excerpt_tool_rejects_absolute_path_outside_corpus(tmp_path):
    corpus, _, server = _build(tmp_path)
    secret = tmp_path / "secret.md"
    secret.write_text("the road commission issued a 68-page notice")
    args = {"source_path": str(secret), "candidate_text": "road commission issued"}
    out = _json(_call_sync(server, "validate_excerpt", args))
    assert out["found"] is False
    assert "outside the corpus" in out["error"]


def test_validate_excerpt_tool_missing_file_is_not_an_escape(tmp_path):
    """A missing file INSIDE the corpus reports 'file not found', not an escape."""
    corpus, _, server = _build(tmp_path)
    args = {"source_path": "nope.md", "candidate_text": "anything"}
    out = _json(_call_sync(server, "validate_excerpt", args))
    assert out["found"] is False
    assert "file not found" in out["error"]


def test_validate_excerpt_tool_survives_an_exception(tmp_path, monkeypatch):
    def _boom(*a, **k):
        raise OSError("kaboom")

    corpus, _, server = _build(tmp_path)
    (corpus / "doc.md").write_text("x")
    monkeypatch.setattr("pipeline.agent_tools.validate_excerpt", _boom)
    args = {"source_path": "doc.md", "candidate_text": "x"}
    result = _call_sync(server, "validate_excerpt", args)
    assert result.isError is True


def test_validate_excerpt_tool_missing_required_arg_is_validation_error(tmp_path):
    corpus, _, server = _build(tmp_path)
    (corpus / "doc.md").write_text("x")
    result = _call_sync(server, "validate_excerpt", {"source_path": "doc.md"})
    assert result.isError is True
    assert "candidate_text" in result.content[0].text


# -------------------------------------------------------------------- fetch_page


def test_fetch_page_tool_takes_only_a_url(tmp_path, monkeypatch):
    corpus, wc, server = _build(tmp_path, claim_id="fact_check-abc123")
    captured = {}

    def _fake_fetch(url, target_id, cache_dir, **kwargs):
        captured.update(url=url, target_id=target_id, cache_dir=cache_dir, **kwargs)
        return {"ok": True, "path": "p", "method": "jina.ai",
                "content": "page text", "warning": None}

    monkeypatch.setattr("pipeline.agent_tools.fetch_page", _fake_fetch)
    result = _call_sync(server, "fetch_page", {"url": "https://x.test/a"})

    assert result.content[0].text == "page text"
    assert captured["url"] == "https://x.test/a"
    # claim_id and cache dir come from the closure, not from the agent
    assert captured["target_id"] == "fact_check-abc123"
    assert captured["cache_dir"] == str(wc)
    # Production never passes use_archive -- the real default governs whether
    # the slow, env-mutating archive.is tier ever runs.
    assert "use_archive" not in captured
    assert inspect.signature(real_fetch_page).parameters["use_archive"].default is False


def test_fetch_page_tool_forwards_method(tmp_path, monkeypatch):
    _, _, server = _build(tmp_path)
    monkeypatch.setattr(
        "pipeline.agent_tools.fetch_page",
        lambda *a, **k: {"ok": True, "path": "p", "method": "archive.is-raw",
                         "content": "raw html", "warning": None},
    )
    result = _call_sync(server, "fetch_page", {"url": "https://x.test/a"})
    texts = [block.text for block in result.content]
    assert any("archive.is-raw" in t for t in texts)


def test_fetch_page_tool_forwards_paywall_warning(tmp_path, monkeypatch):
    warning = ("content may be a paywall preview (11 chars). "
               "Consider finding an alternate source.")
    _, _, server = _build(tmp_path)
    monkeypatch.setattr(
        "pipeline.agent_tools.fetch_page",
        lambda *a, **k: {"ok": True, "path": "p", "method": "jina.ai",
                         "content": "teaser text", "warning": warning},
    )
    result = _call_sync(server, "fetch_page", {"url": "https://x.test/a"})
    texts = [block.text for block in result.content]
    assert any(warning in t for t in texts), texts
    # not itself an error -- ok=True, just flagged for the agent's judgement
    assert result.isError is False


def test_fetch_page_tool_marks_failure_as_error(tmp_path, monkeypatch):
    _, _, server = _build(tmp_path)
    monkeypatch.setattr(
        "pipeline.agent_tools.fetch_page",
        lambda *a, **k: {"ok": False, "path": "p", "method": None,
                         "content": "(failed to fetch x)", "warning": None},
    )
    result = _call_sync(server, "fetch_page", {"url": "https://x.test/a"})
    assert result.isError is True


def test_fetch_page_tool_survives_an_exception(tmp_path, monkeypatch):
    """fetch_page's docstring says it never raises, but mkdir/disk errors can
    still escape. An exception must not propagate into the agent's tool call."""
    def _boom(*a, **k):
        raise OSError("disk full")

    _, _, server = _build(tmp_path)
    monkeypatch.setattr("pipeline.agent_tools.fetch_page", _boom)
    result = _call_sync(server, "fetch_page", {"url": "https://x.test/a"})
    assert result.isError is True
    assert "disk full" in result.content[0].text


def test_tool_names_are_the_two_expected(tmp_path):
    _, _, server = _build(tmp_path)
    result = asyncio.run(
        server["instance"].request_handlers[types.ListToolsRequest](
            types.ListToolsRequest(method="tools/list")))
    names = {t.name for t in result.root.tools}
    assert names == {"validate_excerpt", "fetch_page"}
