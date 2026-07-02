"""Tests for prepare-1 converter (MarkItDown + gap-fillers)."""
from __future__ import annotations

import os

import pipeline.prepare_1_convert as p1


# ── Gap-filler tests ──────────────────────────────────────────

def test_convert_eml(tmp_path):
    src = tmp_path / "m.eml"
    src.write_text(
        "From: alice@example.com\r\nTo: bob@example.com\r\n"
        "Subject: Test Subject\r\nDate: Mon, 1 Jan 2024 00:00:00 +0000\r\n\r\n"
        "This is the body."
    )
    out = tmp_path / "m"
    ok, size, method = p1.convert_eml(src, out)
    md = (tmp_path / "m.md").read_text()
    assert ok
    assert "Test Subject" in md
    assert "alice@example.com" in md
    assert "This is the body." in md
    assert method == "eml-stdlib"


def test_convert_tsv(tmp_path):
    src = tmp_path / "d.tsv"
    src.write_text("h1\th2\nv1\tv2\nv3\tv4")
    out = tmp_path / "d"
    ok, size, method = p1.convert_tsv(src, out)
    md = (tmp_path / "d.md").read_text()
    assert ok
    assert "| h1 | h2 |" in md
    assert "| v3 | v4 |" in md
    assert method == "tsv-table"


def test_convert_xml(tmp_path):
    src = tmp_path / "d.xml"
    src.write_text("<root><a>1</a></root>")
    out = tmp_path / "d"
    ok, size, method = p1.convert_xml(src, out)
    md = (tmp_path / "d.md").read_text()
    assert ok
    assert "```xml" in md
    assert "<root>" in md
    assert method == "xml-fenced"


# ── MarkItDown adapter test ───────────────────────────────────

class _FakeMarkItDown:
    """Stand-in for the real MarkItDown class for testing."""
    def __init__(self, llm_client=None, llm_model=None):
        self.llm_client = llm_client
        self.llm_model = llm_model


class _FakeOpenAI:
    """Stand-in for OpenAI class."""
    def __init__(self, api_key=None):
        self.api_key = api_key


def test_get_markitdown_wiring(monkeypatch):
    """Verify _get_markitdown returns a client when MarkItDown is present."""
    monkeypatch.setattr(p1, "_HAS_MARKITDOWN", True)
    monkeypatch.setattr(p1, "_MarkItDown", _FakeMarkItDown)
    monkeypatch.setattr(p1, "_OpenAI", _FakeOpenAI)
    monkeypatch.setattr(os, "environ", {"OPENAI_API_KEY": "sk-test"})
    client = p1._get_markitdown("gpt-4o-mini")
    assert client is not None
    assert client.llm_model == "gpt-4o-mini"
    assert client.llm_client is not None
    assert client.llm_client.api_key == "sk-test"


# ── Dispatch + gap-filler fallback ────────────────────────────

def test_process_file_falls_back_to_gap_filler(tmp_path, monkeypatch):
    """When MarkItDown fails, the gap-filler handles .eml."""
    src_root = tmp_path / "src"
    corpus = tmp_path / "corpus"
    src_root.mkdir()
    (src_root / "msg.eml").write_text(
        "From: x@y.com\r\nSubject: Hello\r\n\r\nBody text."
    )
    monkeypatch.setattr(p1, "_HAS_MARKITDOWN", False)

    relpath, status, detail, note = p1.process_file(
        src_root / "msg.eml", src_root, corpus, "gpt-4o-mini",
        force=True,
    )
    assert status == "ok"
    assert "Hello" in (corpus / "msg.md").read_text()
    assert note is not None
    assert "fallback" in note


def test_process_file_unconvertible(tmp_path, monkeypatch):
    src_root = tmp_path / "src"
    corpus = tmp_path / "corpus"
    src_root.mkdir()
    (src_root / "mystery.xyz").write_text("???")

    monkeypatch.setattr(p1, "_HAS_MARKITDOWN", False)
    relpath, status, detail, note = p1.process_file(
        src_root / "mystery.xyz", src_root, corpus, "gpt-4o-mini",
        force=True,
    )
    assert status == "fail"
    assert "no converter" in detail
    assert note is None


# ── Reporting ─────────────────────────────────────────────────

def test_write_unconverted(tmp_path):
    p1._write_unconverted(tmp_path, [("bad.xyz", "no converter for .xyz")])
    report = tmp_path / "UNCONVERTED.md"
    assert report.exists()
    assert "bad.xyz" in report.read_text()


def test_clean_run_removes_unconverted(tmp_path):
    stale = tmp_path / "UNCONVERTED.md"
    stale.write_text("stale")
    p1._write_unconverted(tmp_path, [])
    assert not stale.exists()


def test_write_needs_review(tmp_path):
    p1._write_needs_review(tmp_path, [("scan.png", "used .eml fallback")])
    report = tmp_path / "NEEDS_REVIEW.md"
    assert report.exists()
    assert "scan.png" in report.read_text()


# ── run_prepare_1 end-to-end ─────────────────────────────────

def test_run_prepare_1_end_to_end(tmp_path, monkeypatch):
    src = tmp_path / "src"
    corpus = tmp_path / "corpus"
    src.mkdir()
    xml_content = (
        '<?xml version="1.0"?><doc>'
        'Hello world this is enough text to pass the meaningful check.'
        '</doc>'
    )
    (src / "a.xml").write_text(xml_content)
    (src / "b.eml").write_text(
        "From: x@y.com\r\nSubject: S\r\n\r\nBody text."
    )
    monkeypatch.setattr(p1, "_HAS_MARKITDOWN", False)
    report = p1.run_prepare_1(
        source_root=str(src), corpus_root=str(corpus),
        workers=2, vision_model="gpt-4o-mini", force=True,
    )
    assert report["failure_count"] == 0
    assert (corpus / "a.md").exists()
    assert (corpus / "b.md").exists()
    assert not (corpus / "UNCONVERTED.md").exists()


def test_run_prepare_1_reports_unconvertible(tmp_path, monkeypatch):
    src = tmp_path / "src"
    corpus = tmp_path / "corpus"
    src.mkdir()
    (src / "mystery.xyz").write_text("???")
    monkeypatch.setattr(p1, "_HAS_MARKITDOWN", False)
    report = p1.run_prepare_1(
        source_root=str(src), corpus_root=str(corpus),
        workers=1, vision_model="gpt-4o-mini", force=True,
    )
    assert report["failure_count"] == 1
    assert (corpus / "UNCONVERTED.md").exists()
