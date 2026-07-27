"""Tests for pipeline/agent_tools.py -- the verification agent's reachable surface."""
import asyncio
import json


def _call(handler_tool, args: dict) -> dict:
    """Invoke an SdkMcpTool's handler and parse its JSON text payload."""
    result = asyncio.run(handler_tool.handler(args))
    return json.loads(result["content"][0]["text"])


def _find_tool_map(func, seen=None):
    """Depth-first search a closure chain for the {name: SdkMcpTool} dict.

    `create_sdk_mcp_server` (claude_agent_sdk 0.2.127) does not expose the
    original SdkMcpTool list anywhere on the returned McpSdkServerConfig or
    its `instance` (an mcp.server.lowlevel.server.Server) -- it is only kept
    as a closure variable of the nested handler registered for
    mcp.types.CallToolRequest. There is no public accessor for it, so this
    walks the closure chain looking for a dict whose values are all
    SdkMcpTool instances. Confirmed empirically (see task report) that no
    other route reaches it.
    """
    from claude_agent_sdk import SdkMcpTool

    if seen is None:
        seen = set()
    if id(func) in seen or not hasattr(func, "__closure__") or func.__closure__ is None:
        return None
    seen.add(id(func))
    for cell in func.__closure__:
        try:
            val = cell.cell_contents
        except ValueError:
            continue
        if isinstance(val, dict) and val and all(isinstance(v, SdkMcpTool) for v in val.values()):
            return val
        if callable(val):
            found = _find_tool_map(val, seen)
            if found is not None:
                return found
    return None


def _tools(server):
    """Map tool name -> SdkMcpTool for a server built by build_verification_tools."""
    import mcp.types as types
    handler = server["instance"].request_handlers[types.CallToolRequest]
    tool_map = _find_tool_map(handler)
    assert tool_map is not None, "could not locate tool map in server closures"
    return tool_map


def _build(tmp_path, claim_id="c1"):
    from pipeline.agent_tools import build_verification_tools
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    wc = corpus / "web_cache"
    wc.mkdir()
    server = build_verification_tools(str(corpus), str(wc), claim_id)
    return corpus, wc, server


def test_validate_excerpt_tool_resolves_relative_to_corpus(tmp_path):
    corpus, _, server = _build(tmp_path)
    (corpus / "doc.md").write_text("The road commission issued a 68-page notice.")
    out = _call(_tools(server)["validate_excerpt"],
                {"source_path": "doc.md", "candidate_text": "road commission issued"})
    assert out["found"] is True
    assert out["similarity"] == 1.0
    assert out["actual_text"] == "road commission issued"


def test_validate_excerpt_tool_rejects_path_escape(tmp_path):
    corpus, _, server = _build(tmp_path)
    (tmp_path / "secret.md").write_text("the road commission issued a 68-page notice")
    out = _call(_tools(server)["validate_excerpt"],
                {"source_path": "../secret.md",
                 "candidate_text": "road commission issued"})
    assert out["found"] is False
    assert "outside the corpus" in out["error"]


def test_validate_excerpt_tool_rejects_absolute_path_outside_corpus(tmp_path):
    corpus, _, server = _build(tmp_path)
    secret = tmp_path / "secret.md"
    secret.write_text("the road commission issued a 68-page notice")
    out = _call(_tools(server)["validate_excerpt"],
                {"source_path": str(secret),
                 "candidate_text": "road commission issued"})
    assert out["found"] is False
    assert "outside the corpus" in out["error"]


def test_validate_excerpt_tool_missing_file_is_not_an_escape(tmp_path):
    """A missing file INSIDE the corpus reports 'file not found', not an escape."""
    corpus, _, server = _build(tmp_path)
    out = _call(_tools(server)["validate_excerpt"],
                {"source_path": "nope.md", "candidate_text": "anything"})
    assert out["found"] is False
    assert "file not found" in out["error"]


def test_fetch_page_tool_takes_only_a_url(tmp_path, monkeypatch):
    corpus, wc, server = _build(tmp_path, claim_id="fact_check-abc123")
    captured = {}

    def _fake_fetch(url, target_id, cache_dir, debug=False, use_archive=False):
        captured.update(url=url, target_id=target_id, cache_dir=cache_dir,
                        use_archive=use_archive)
        return {"ok": True, "path": "p", "method": "jina.ai",
                "content": "page text", "warning": None}

    monkeypatch.setattr("pipeline.agent_tools.fetch_page", _fake_fetch)
    result = asyncio.run(_tools(server)["fetch_page"].handler({"url": "https://x.test/a"}))

    assert result["content"][0]["text"] == "page text"
    assert captured["url"] == "https://x.test/a"
    # claim_id and cache dir come from the closure, not from the agent
    assert captured["target_id"] == "fact_check-abc123"
    assert captured["cache_dir"] == str(wc)
    # archive.is is slow and mutates process env -- must stay off
    assert captured["use_archive"] is False


def test_fetch_page_tool_marks_failure_as_error(tmp_path, monkeypatch):
    _, _, server = _build(tmp_path)
    monkeypatch.setattr(
        "pipeline.agent_tools.fetch_page",
        lambda *a, **k: {"ok": False, "path": "p", "method": None,
                         "content": "(failed to fetch x)", "warning": None},
    )
    result = asyncio.run(_tools(server)["fetch_page"].handler({"url": "https://x.test/a"}))
    assert result.get("is_error") is True


def test_fetch_page_tool_survives_an_exception(tmp_path, monkeypatch):
    """fetch_page's docstring says it never raises, but mkdir/disk errors can
    still escape. An exception must not propagate into the agent's tool call."""
    def _boom(*a, **k):
        raise OSError("disk full")

    _, _, server = _build(tmp_path)
    monkeypatch.setattr("pipeline.agent_tools.fetch_page", _boom)
    result = asyncio.run(_tools(server)["fetch_page"].handler({"url": "https://x.test/a"}))
    assert result.get("is_error") is True
    assert "disk full" in result["content"][0]["text"]


def test_validate_excerpt_tool_survives_an_exception(tmp_path, monkeypatch):
    def _boom(*a, **k):
        raise OSError("kaboom")

    corpus, _, server = _build(tmp_path)
    (corpus / "doc.md").write_text("x")
    monkeypatch.setattr("pipeline.agent_tools.validate_excerpt", _boom)
    result = asyncio.run(_tools(server)["validate_excerpt"].handler(
        {"source_path": "doc.md", "candidate_text": "x"}))
    assert result.get("is_error") is True


def test_tool_names_are_the_two_expected(tmp_path):
    _, _, server = _build(tmp_path)
    assert set(_tools(server)) == {"validate_excerpt", "fetch_page"}
