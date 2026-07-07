"""Tests for prepare-1 converter: routing (images/audio/pdf), MarkItDown, gap-fillers."""
from __future__ import annotations

import pipeline.prepare_1_convert as p1

VISION_CFG = {"enabled": False, "model": "gpt-4o-mini", "min_words": 20,
              "max_pages_per_doc": 30, "ocr_images": True}
AUDIO_CFG = {"enabled": False, "model": "medium", "device": "auto"}


# ── Gap-filler tests ──────────────────────────────────────────

def test_convert_eml(tmp_path):
    src = tmp_path / "m.eml"
    src.write_text(
        "From: alice@example.com\r\nTo: bob@example.com\r\n"
        "Subject: Test Subject\r\nDate: Mon, 1 Jan 2024 00:00:00 +0000\r\n\r\n"
        "This is the body."
    )
    ok, size, method = p1.convert_eml(src, tmp_path / "m.md")
    md = (tmp_path / "m.md").read_text()
    assert ok
    assert "Test Subject" in md and "alice@example.com" in md and "This is the body." in md
    assert method == "eml-stdlib"


def test_convert_tsv(tmp_path):
    src = tmp_path / "d.tsv"
    src.write_text("h1\th2\nv1\tv2\nv3\tv4")
    ok, size, method = p1.convert_tsv(src, tmp_path / "d.md")
    md = (tmp_path / "d.md").read_text()
    assert ok and "| h1 | h2 |" in md and "| v3 | v4 |" in md and method == "tsv-table"


def test_convert_xml(tmp_path):
    src = tmp_path / "d.xml"
    src.write_text("<root><a>1</a></root>")
    ok, size, method = p1.convert_xml(src, tmp_path / "d.md")
    md = (tmp_path / "d.md").read_text()
    assert ok and "```xml" in md and "<root>" in md and method == "xml-fenced"


def test_libreoffice_uses_isolated_profile(tmp_path, monkeypatch):
    """Concurrent-safe: each LibreOffice call gets its own -env:UserInstallation profile."""
    captured = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        return None  # no output file -> conversion "fails", but we inspect the cmd

    monkeypatch.setattr(p1.subprocess, "run", fake_run)
    p1._convert_via_libreoffice(tmp_path / "a.doc", tmp_path / "a")
    assert any(str(a).startswith("-env:UserInstallation=file://") for a in captured["cmd"])


# ── MarkItDown wiring ─────────────────────────────────────────

def test_get_markitdown_present(monkeypatch):
    monkeypatch.setattr(p1, "_HAS_MARKITDOWN", True)
    monkeypatch.setattr(p1, "_MarkItDown", lambda: "CLIENT")
    assert p1._get_markitdown() == "CLIENT"


def test_get_markitdown_construction_failure(monkeypatch):
    """A construction failure returns None (the 'unavailable' warning is reachable)."""
    monkeypatch.setattr(p1, "_HAS_MARKITDOWN", True)

    def boom():
        raise RuntimeError("missing optional deps")

    monkeypatch.setattr(p1, "_MarkItDown", boom)
    assert p1._get_markitdown() is None


# ── Routing ───────────────────────────────────────────────────

def test_image_routes_to_ocr(tmp_path, monkeypatch):
    src_root = tmp_path / "src"
    corpus = tmp_path / "corpus"
    src_root.mkdir()
    (src_root / "x.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    monkeypatch.setattr(p1, "ocr_image",
                        lambda inp, outp, vc: (True, 5, "tesseract", "image transcribed via OCR"))
    rel, status, method, note = p1.process_file(
        src_root / "x.png", src_root, corpus, VISION_CFG, None, True)
    assert status == "ok" and method == "tesseract" and note == "image transcribed via OCR"


def test_audio_routes_to_whisper(tmp_path, monkeypatch):
    src_root = tmp_path / "src"
    corpus = tmp_path / "corpus"
    src_root.mkdir()
    (src_root / "a.mp3").write_bytes(b"ID3")
    monkeypatch.setattr(p1, "convert_audio",
                        lambda inp, outp, wm: (True, 5, "whisper", "audio transcribed via whisper"))
    rel, status, method, note = p1.process_file(
        src_root / "a.mp3", src_root, corpus, VISION_CFG, "MODEL", True)
    assert status == "ok" and method == "whisper" and "whisper" in note


def test_audio_fail_when_whisper_unavailable(tmp_path, monkeypatch):
    src_root = tmp_path / "src"
    corpus = tmp_path / "corpus"
    src_root.mkdir()
    (src_root / "a.mp3").write_bytes(b"ID3")
    monkeypatch.setattr(p1, "convert_audio",
                        lambda inp, outp, wm: (False, 0, "whisper-unavailable", None))
    rel, status, detail, note = p1.process_file(
        src_root / "a.mp3", src_root, corpus, VISION_CFG, None, True)
    assert status == "fail" and "audio not transcribed" in detail


def test_pdf_scanned_falls_back_to_ocr(tmp_path, monkeypatch):
    """MarkItDown returns empty on a scanned PDF -> ocr_pdf is invoked."""
    src_root = tmp_path / "src"
    corpus = tmp_path / "corpus"
    src_root.mkdir()
    (src_root / "scan.pdf").write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr(p1, "_convert_with_markitdown",
                        lambda fp, op, mc: (False, 0, "markitdown-empty"))
    calls = []

    def fake_ocr_pdf(inp, outp, vc):
        calls.append(inp.name)
        outp.with_suffix(".md").write_text("OCR TEXT")
        return True, 8, "ocr", "scanned PDF; OCR only"

    monkeypatch.setattr(p1, "ocr_pdf", fake_ocr_pdf)
    rel, status, method, note = p1.process_file(
        src_root / "scan.pdf", src_root, corpus, VISION_CFG, None, True, md_client=object())
    assert status == "ok" and method == "ocr" and calls == ["scan.pdf"]


def test_pdf_digital_uses_markitdown(tmp_path, monkeypatch):
    """A digital PDF with a real text layer stays with MarkItDown (no OCR)."""
    src_root = tmp_path / "src"
    corpus = tmp_path / "corpus"
    src_root.mkdir()
    (src_root / "digital.pdf").write_bytes(b"%PDF-1.4 fake")

    def fake_mk(fp, op, mc):
        op.with_suffix(".md").write_text("A" * 200)  # meaningful text
        return True, 200, "markitdown"

    monkeypatch.setattr(p1, "_convert_with_markitdown", fake_mk)
    monkeypatch.setattr(p1, "ocr_pdf",
                        lambda *a: (_ for _ in ()).throw(AssertionError("ocr_pdf should not run")))
    rel, status, method, note = p1.process_file(
        src_root / "digital.pdf", src_root, corpus, VISION_CFG, None, True, md_client=object())
    assert status == "ok" and method == "markitdown"


def test_process_file_falls_back_to_gap_filler(tmp_path, monkeypatch):
    src_root = tmp_path / "src"
    corpus = tmp_path / "corpus"
    src_root.mkdir()
    (src_root / "msg.eml").write_text("From: x@y.com\r\nSubject: Hello\r\n\r\nBody text.")
    rel, status, detail, note = p1.process_file(
        src_root / "msg.eml", src_root, corpus, VISION_CFG, None, True, md_client=None)
    assert status == "ok"
    assert "Hello" in (corpus / "msg.eml.md").read_text()
    assert note is not None and "fallback" in note


def test_process_file_unconvertible(tmp_path):
    src_root = tmp_path / "src"
    corpus = tmp_path / "corpus"
    src_root.mkdir()
    (src_root / "mystery.xyz").write_text("???")
    rel, status, detail, note = p1.process_file(
        src_root / "mystery.xyz", src_root, corpus, VISION_CFG, None, True, md_client=None)
    assert status == "fail" and "no converter" in detail and note is None


# ── Reporting ─────────────────────────────────────────────────

def test_write_unconverted(tmp_path):
    p1._write_unconverted(tmp_path, [("bad.xyz", "no converter for .xyz")])
    assert (tmp_path / "UNCONVERTED.md").exists()
    assert "bad.xyz" in (tmp_path / "UNCONVERTED.md").read_text()


def test_clean_run_removes_unconverted(tmp_path):
    (tmp_path / "UNCONVERTED.md").write_text("stale")
    p1._write_unconverted(tmp_path, [])
    assert not (tmp_path / "UNCONVERTED.md").exists()


def test_write_needs_review(tmp_path):
    p1._write_needs_review(tmp_path, [("scan.png", "image transcribed via OCR")])
    assert (tmp_path / "NEEDS_REVIEW.md").exists()
    assert "scan.png" in (tmp_path / "NEEDS_REVIEW.md").read_text()


# ── run_prepare_1 end-to-end (gap-fillers only, no markitdown/whisper) ─────

def test_run_prepare_1_end_to_end(tmp_path, monkeypatch):
    src = tmp_path / "src"
    corpus = tmp_path / "corpus"
    src.mkdir()
    (src / "a.xml").write_text('<?xml version="1.0"?><doc>enough text to be fine</doc>')
    (src / "b.eml").write_text("From: x@y.com\r\nSubject: S\r\n\r\nBody text.")
    monkeypatch.setattr(p1, "_HAS_MARKITDOWN", False)
    report = p1.run_prepare_1(
        source_root=str(src), corpus_root=str(corpus),
        workers=2, vision_cfg=VISION_CFG, audio_cfg=AUDIO_CFG, force=True)
    assert report["failure_count"] == 0
    assert (corpus / "a.xml.md").exists() and (corpus / "b.eml.md").exists()
    assert not (corpus / "UNCONVERTED.md").exists()


def test_run_prepare_1_reports_unconvertible(tmp_path, monkeypatch):
    src = tmp_path / "src"
    corpus = tmp_path / "corpus"
    src.mkdir()
    (src / "mystery.xyz").write_text("???")
    monkeypatch.setattr(p1, "_HAS_MARKITDOWN", False)
    report = p1.run_prepare_1(
        source_root=str(src), corpus_root=str(corpus),
        workers=1, vision_cfg=VISION_CFG, audio_cfg=AUDIO_CFG, force=True)
    assert report["failure_count"] == 1
    assert (corpus / "UNCONVERTED.md").exists()


# ── Stem collision: sibling formats get unique .md names ─────────

def test_sibling_formats_produce_unique_md_files(tmp_path, monkeypatch):
    """When a .doc and .pdf share a stem, both get their own .md output.

    Before the fix, letter.doc and letter.pdf both mapped to letter.md, and the
    first one processed (alphabetically, .doc) would block the second via the
    already-converted check, silently discarding the PDF's content.
    """
    src = tmp_path / "src"
    corpus = tmp_path / "corpus"
    src.mkdir()

    # Create two sibling source files sharing a stem.
    (src / "letter.doc").write_bytes(b"\xd0\xcf\x11\xe0")  # OLE2 magic
    (src / "letter.pdf").write_bytes(b"%PDF-1.4 fake pdf content")

    # Simulate .doc conversion (via LibreOffice fallback) producing short content.
    def fake_libreoffice(inpath, md_path):
        md_path.write_text("short doc content " * 7, encoding="utf-8")  # ~133 chars, passes is_meaningful
        return True, 18, "libreoffice"

    # Simulate .pdf conversion (via MarkItDown) producing rich content.
    def fake_markitdown_pdf(inpath, md_path, md_client):
        md_path.write_text("rich pdf content with all pages " * 7, encoding="utf-8")  # passes is_meaningful
        return True, 34, "markitdown"

    # Route .pdf through MarkItDown, .doc through LibreOffice.
    def fake_markitdown(inpath, md_path, md_client):
        if inpath.suffix == ".pdf":
            return fake_markitdown_pdf(inpath, md_path, md_client)
        return False, 0, "markitdown-empty"

    monkeypatch.setattr(p1, "_convert_with_markitdown", fake_markitdown)
    monkeypatch.setattr(p1, "_convert_via_libreoffice", fake_libreoffice)

    # Process both files.
    rel1, status1, method1, note1 = p1.process_file(
        src / "letter.doc", src, corpus, VISION_CFG, None, True, md_client=object())
    rel2, status2, method2, note2 = p1.process_file(
        src / "letter.pdf", src, corpus, VISION_CFG, None, True, md_client=object())

    # Both should succeed.
    assert status1 == "ok"
    assert status2 == "ok"

    # Each produces its own .md file — no clobbering.
    doc_md = corpus / "letter.doc.md"
    pdf_md = corpus / "letter.pdf.md"
    assert doc_md.exists(), f"Expected {doc_md} to exist"
    assert pdf_md.exists(), f"Expected {pdf_md} to exist"

    # Each contains the correct content from its source.
    assert "short doc content" in doc_md.read_text()
    assert "rich pdf content with all pages" in pdf_md.read_text()

    # The old collision-prone name should NOT exist.
    assert not (corpus / "letter.md").exists()
