"""Tests for prepare-1 converter: routing (images/audio/pdf), MarkItDown, gap-fillers."""
from __future__ import annotations

from pathlib import Path

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
    ok, size, method, note = p1.convert_eml(src, tmp_path / "m.md")
    md = (tmp_path / "m.md").read_text()
    assert ok
    assert "Test Subject" in md and "alice@example.com" in md and "This is the body." in md
    assert method == "eml-extract"


def test_passthrough_text_is_byte_for_byte(tmp_path):
    """Text-native files are copied verbatim — the sidecar is a byte-for-byte
    twin of the source, including Windows CRLF (no text-mode newline translation)."""
    src = tmp_path / "d.csv"
    src.write_bytes(b"h1,h2\r\nv1,v2\r\nv3,v4\r\n")
    ok, size, method, note = p1.passthrough_text(src, tmp_path / "d.csv.md")
    assert ok and method == "text-passthrough" and note is None
    assert (tmp_path / "d.csv.md").read_bytes() == src.read_bytes()  # CRLF preserved


def test_process_file_reflows_long_line_through_finish(tmp_path, monkeypatch):
    """Proves the real wiring: process_file -> _finish -> _reflow_md_file_in_place,
    not just reflow_pipeline_text() called in isolation. A converter that bypasses
    _finish with a stub returning one line well over the 300-char threshold must
    come out wrapped once process_file has run, because _finish re-flows the
    written .md file in place on the way out.
    """
    src_root = tmp_path / "src"
    corpus = tmp_path / "corpus"
    src_root.mkdir()
    (src_root / "x.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    long_line = " ".join(["word"] * 100)  # 499 chars, no sentence punctuation
    assert len(long_line) > 300

    def fake_ocr_image(inp, outp, vc):
        outp.write_text(long_line)
        return True, len(long_line), "tesseract", "image transcribed via OCR"

    monkeypatch.setattr(p1, "ocr_image", fake_ocr_image)
    rel, status, method, note = p1.process_file(
        src_root / "x.png", src_root, corpus, VISION_CFG, None, True)
    assert status == "ok"
    md = (corpus / "x.png.md").read_text()
    assert "\n" in md
    for line in md.split("\n"):
        assert len(line) <= 300
    # Confirm no words were lost or altered by the wrap.
    assert md.split() == long_line.split()


def test_process_file_passthrough_text_not_reflowed(tmp_path):
    """Proves reflow=False actually suppresses reflow at its real call site:
    process_file's text-native branch calls _finish(..., md_path, reflow=False).
    A long single line routed through passthrough_text must survive untouched —
    still exactly one line, byte-for-byte — not wrapped like every other branch.
    """
    src_root = tmp_path / "src"
    corpus = tmp_path / "corpus"
    src_root.mkdir()
    long_line = " ".join(["word"] * 100)  # 499 chars, over the reflow threshold
    assert len(long_line) > 300
    (src_root / "wells.csv").write_text(long_line)

    rel, status, method, note = p1.process_file(
        src_root / "wells.csv", src_root, corpus, VISION_CFG, None, True,
        md_client=None, text_native_exts={".csv"})
    assert status == "ok" and method == "text-passthrough"
    out = (corpus / "wells.csv.md").read_text()
    assert out == long_line
    assert "\n" not in out


def test_text_native_routes_to_passthrough_not_converter(tmp_path):
    """A .csv is copied through verbatim, not converted to a pipe table.

    With md_client=None, a .csv that fell through to MarkItDown/gap-fillers would
    FAIL (there's no .csv gap-filler) — so 'ok' proves passthrough handled it.
    The contrast case (same file, empty text-native set) exercises the without-
    condition path and must differ: it falls through and fails.
    """
    src_root = tmp_path / "src"
    corpus = tmp_path / "corpus"
    src_root.mkdir()
    (src_root / "wells.csv").write_bytes(b"api,county\r\n01,Bandera\r\n")

    # WITH the condition: csv is text-native -> passthrough.
    rel, status, method, note = p1.process_file(
        src_root / "wells.csv", src_root, corpus, VISION_CFG, None, True,
        md_client=None, text_native_exts={".csv"})
    out = (corpus / "wells.csv.md").read_bytes()
    assert status == "ok" and method == "text-passthrough"
    assert out == (src_root / "wells.csv").read_bytes() and b"| --- |" not in out

    # WITHOUT the condition: csv not text-native -> falls through to the (absent)
    # converter and fails. Results must differ.
    rel2, status2, detail2, note2 = p1.process_file(
        src_root / "wells.csv", src_root, corpus, VISION_CFG, None, True,
        md_client=None, text_native_exts=set())
    assert status2 == "fail" and "no converter" in detail2
    assert status2 != status


def test_stale_sidecar_reconverts_when_source_is_newer(tmp_path):
    import os
    import time

    src_root = tmp_path / "src"
    corpus = tmp_path / "corpus"
    src_root.mkdir()
    src = src_root / "doc.txt"
    src.write_text("original content, long enough to look meaningful " * 3)

    # First conversion
    p1.process_file(
        src, src_root, corpus, VISION_CFG, None, True,
        md_client=None, text_native_exts={".txt"})
    md_path = corpus / "doc.txt.md"
    assert md_path.read_text().startswith("original content")

    # Touch the source with newer content and a later mtime
    time.sleep(0.01)
    src.write_text("UPDATED content, long enough to look meaningful too " * 3)
    newer_time = time.time() + 1
    os.utime(src, (newer_time, newer_time))

    rel, status, method, note = p1.process_file(
        src, src_root, corpus, VISION_CFG, None, False,  # force=False
        md_client=None, text_native_exts={".txt"})

    assert status == "ok"  # not "skip" -- re-converted because source is newer
    assert md_path.read_text().startswith("UPDATED content")


def test_untouched_sidecar_still_skips(tmp_path):
    src_root = tmp_path / "src"
    corpus = tmp_path / "corpus"
    src_root.mkdir()
    src = src_root / "doc.txt"
    src.write_text("original content, long enough to look meaningful " * 3)

    p1.process_file(
        src, src_root, corpus, VISION_CFG, None, True,
        md_client=None, text_native_exts={".txt"})
    rel, status, method, note = p1.process_file(
        src, src_root, corpus, VISION_CFG, None, False,
        md_client=None, text_native_exts={".txt"})

    assert status == "skip" and method == "already converted"


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

    def fake_ocr_image(inp, outp, vc):
        outp.write_text("ocr text")
        return True, 5, "tesseract", "image transcribed via OCR"

    monkeypatch.setattr(p1, "ocr_image", fake_ocr_image)
    rel, status, method, note = p1.process_file(
        src_root / "x.png", src_root, corpus, VISION_CFG, None, True)
    assert status == "ok" and method == "tesseract" and note == "image transcribed via OCR"


def test_audio_routes_to_whisper(tmp_path, monkeypatch):
    src_root = tmp_path / "src"
    corpus = tmp_path / "corpus"
    src_root.mkdir()
    (src_root / "a.mp3").write_bytes(b"ID3")

    def fake_convert_audio(inp, outp, wm):
        outp.write_text("audio transcript text")
        return True, 5, "whisper", "audio transcribed via whisper"

    monkeypatch.setattr(p1, "convert_audio", fake_convert_audio)
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
    """Gap-fillers still work for formats not handled directly (.doc, .rtf, etc.)."""
    src_root = tmp_path / "src"
    corpus = tmp_path / "corpus"
    src_root.mkdir()
    (src_root / "test.doc").write_bytes(b"\xd0\xcf\x11\xe0")  # OLE2 magic
    def fake_lo(inpath, md_path):
        md_path.write_text("libreoffice output")
        return True, 19, "libreoffice"
    monkeypatch.setattr(p1, "_convert_via_libreoffice", fake_lo)
    rel, status, detail, note = p1.process_file(
        src_root / "test.doc", src_root, corpus, VISION_CFG, None, True, md_client=None)
    assert status == "ok"
    assert "libreoffice output" in (corpus / "test.doc.md").read_text()
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


# ── Email attachment extraction ─────────────────────────────────

def test_eml_extracts_attachments(tmp_path):
    """Real multipart .eml with a PDF attachment → attachment extracted."""
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.application import MIMEApplication

    # Build a multipart email with body + PDF attachment
    msg = MIMEMultipart()
    msg["From"] = "sender@test.com"
    msg["To"] = "recipient@test.com"
    msg["Subject"] = "Test with attachment"
    msg["Date"] = "Mon, 1 Jan 2024 00:00:00 +0000"
    msg.attach(MIMEText("Email body text here."))
    pdf_bytes = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\nxref\n0 1\ntrailer\n<<>>\n%%EOF"
    att = MIMEApplication(pdf_bytes, _subtype="pdf")
    att.add_header("Content-Disposition", "attachment", filename="report.pdf")
    msg.attach(att)

    src = tmp_path / "test.eml"
    src.write_bytes(msg.as_bytes())

    ok, size, method, note = p1.convert_eml(src, tmp_path / "test.eml.md")
    assert ok and method == "eml-extract"
    assert note == "1 attachment(s) extracted"

    # Check attachment was extracted
    attach_dir = tmp_path / "test.eml_attachments"
    assert attach_dir.is_dir()
    extracted = list(attach_dir.iterdir())
    assert len(extracted) == 1
    assert extracted[0].name == "001_report.pdf"
    assert extracted[0].read_bytes() == pdf_bytes

    # Check email body
    md = (tmp_path / "test.eml.md").read_text()
    assert "Email body text here" in md
    assert "1 file(s) extracted" in md


def test_eml_extracts_nested_email(tmp_path):
    """Multipart .eml with a forwarded email → nested .eml saved."""
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.message import MIMEMessage

    # Build nested email
    inner = MIMEText("Forwarded message body.")
    inner["From"] = "nested@test.com"
    inner["Subject"] = "Forwarded: Important"

    outer = MIMEMultipart()
    outer["From"] = "sender@test.com"
    outer["Subject"] = "Fwd: Important"
    outer.attach(MIMEText("See below."))
    outer.attach(MIMEMessage(inner))

    src = tmp_path / "fwd.eml"
    src.write_bytes(outer.as_bytes())

    ok, size, method, note = p1.convert_eml(src, tmp_path / "fwd.eml.md")
    assert ok
    assert note == "1 attachment(s) extracted"

    attach_dir = tmp_path / "fwd.eml_attachments"
    nested_files = list(attach_dir.glob("*.eml"))
    assert len(nested_files) == 1
    assert "Forwarded" in nested_files[0].name


# ── Subtitles (.srt / .vtt) via pycaption ─────────────────────────

def test_convert_subtitle_vtt(tmp_path):
    src = tmp_path / "d.vtt"
    src.write_text(
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:02.000\n"
        "Hello there.\n\n"
        "00:00:02.000 --> 00:00:04.000\n"
        "General Kenobi.\n"
    )
    ok, size, method, note = p1.convert_subtitle(src, tmp_path / "d.md")
    md = (tmp_path / "d.md").read_text()
    assert ok and method == "pycaption-transcript" and note is None
    assert md.strip().endswith("Hello there. General Kenobi.")
    assert "# Transcript: d.vtt" in md
    # No cue numbers or timestamps leaked into the transcript
    assert "-->" not in md and "00:00" not in md


def test_convert_subtitle_srt(tmp_path):
    src = tmp_path / "d.srt"
    src.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nHello there.\n\n"
        "2\n00:00:02,000 --> 00:00:04,000\nGeneral Kenobi.\n"
    )
    ok, size, method, note = p1.convert_subtitle(src, tmp_path / "d.md")
    md = (tmp_path / "d.md").read_text()
    assert ok and method == "pycaption-transcript" and note is None
    assert md.strip().endswith("Hello there. General Kenobi.")
    assert "# Transcript: d.srt" in md


def test_convert_subtitle_header_makes_no_origin_claim(tmp_path):
    src = tmp_path / "d.vtt"
    src.write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nHello there.\n"
    )
    ok, size, method, note = p1.convert_subtitle(src, tmp_path / "d.md")
    md = (tmp_path / "d.md").read_text()
    assert "# Transcript: d.vtt" in md
    assert "cue numbers and timestamps removed" in md
    assert "Original file preserved" in md
    for phrase in ("auto-generated", "machine-generated", "transcription errors expected"):
        assert phrase.lower() not in md.lower()


def test_convert_subtitle_srt_youtube_blank_placeholder_pattern(tmp_path):
    """Real-world YouTube-exported .srt puts a bare CRLF (no space, no text)
    as a placeholder line WITHIN a cue block, before the real text arrives
    two lines later -- confirmed on a real corpus, where this pattern
    caused 20/22 .srt files to lose 100% of their content: pycaption's
    SRTReader mistakes the placeholder for the separator BETWEEN cues,
    discards the cue, then aborts the whole file when it next expects a
    cue-number digit and finds leftover text instead. Routing .srt through
    the WebVTTReader (which only closes a cue once it has real text
    collected) fixes this -- this test reproduces the exact byte pattern
    with generic content, not real transcript data."""
    src = tmp_path / "d.srt"
    src.write_bytes(
        b"1\r\n00:00:00,799 --> 00:00:58,110\r\n\r\nHello there.\r\n\r\n"
        b"2\r\n00:00:58,110 --> 00:00:58,120\r\n\r\n \r\n\r\n"
        b"3\r\n00:00:58,120 --> 00:00:59,470\r\n\r\nGeneral Kenobi.\r\n\r\n"
    )
    ok, size, method, note = p1.convert_subtitle(src, tmp_path / "d.md")
    md = (tmp_path / "d.md").read_text()
    assert ok and method == "pycaption-transcript" and note is None
    assert "Hello there." in md
    assert "General Kenobi." in md


def test_convert_subtitle_dedupes_rolling_captions(tmp_path):
    """Auto-generated 'rolling' captions repeat the prior line per cue -- collapsed."""
    src = tmp_path / "rolling.vtt"
    src.write_text(
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:02.000\n"
        "Hello there\n\n"
        "00:00:02.000 --> 00:00:04.000\n"
        "Hello there\n"
        "General Kenobi\n"
    )
    ok, size, method, note = p1.convert_subtitle(src, tmp_path / "rolling.md")
    md = (tmp_path / "rolling.md").read_text()
    assert ok
    assert md.strip().endswith("Hello there General Kenobi")  # not repeated twice


def test_convert_subtitle_malformed_fails(tmp_path):
    src = tmp_path / "bad.vtt"
    src.write_text("this is not a valid webvtt file at all")
    ok, size, method, note = p1.convert_subtitle(src, tmp_path / "bad.md")
    assert not ok and method == "pycaption-error" and note is None


def test_convert_subtitle_real_youtube_auto_captions(tmp_path):
    """Real messy auto-generated captions (yt-dlp, NASA press briefing excerpt):
    per-word <HH:MM:SS.mmm><c>...</c> timing tags, align:start position:0% cue
    settings, rolling captions that repeat the prior line, and blank filler
    cues (single-space payload). Exercises tag-stripping + dedup together.
    """
    fixture = Path(__file__).parent / "fixtures" / "sample_auto_captions.vtt"
    ok, size, method, note = p1.convert_subtitle(fixture, tmp_path / "out.md")
    md = (tmp_path / "out.md").read_text()

    assert ok and method == "pycaption-transcript"

    # No VTT syntax artifacts leaked into the transcript
    for artifact in ("-->", "align:start", "<c>", "</c>", "WEBVTT", "Kind:"):
        assert artifact not in md, f"{artifact!r} leaked into transcript"

    # Rolling captions stitched into continuous, non-repeated text: this exact
    # substring can only appear if "You are looking live at the Aremis 2" and
    # "space launch system rocket and Orion" each appear once, not per-cue.
    assert ("You are looking live at the Aremis 2 space launch system rocket "
            "and Orion spacecraft in the vehicle assembly building. Today we "
            "are joined by agency") in md
    assert "Everybody silence." in md
    assert "NASA's Kennedy Space Center" in md
    assert "I'm George Alderman" in md

    # Known duplication failure mode: a rolling line repeated back-to-back
    assert "Aremis 2 You are looking live at the Aremis 2" not in md
    assert "Orion spacecraft in the vehicle assembly spacecraft" not in md


def test_subtitle_routes_directly_bypassing_markitdown(tmp_path, monkeypatch):
    """.srt/.vtt go straight to convert_subtitle, not through MarkItDown."""
    src_root = tmp_path / "src"
    corpus = tmp_path / "corpus"
    src_root.mkdir()
    (src_root / "captions.vtt").write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHi\n")

    called = []

    def fake_convert_subtitle(inp, outp):
        called.append(1)
        outp.write_text("transcript text")
        return True, 2, "pycaption-transcript", None

    monkeypatch.setattr(p1, "convert_subtitle", fake_convert_subtitle)

    class ExplodingMarkItDown:
        def convert(self, *a, **kw):
            raise AssertionError("MarkItDown should not be called for .vtt")

    rel, status, method, note = p1.process_file(
        src_root / "captions.vtt", src_root, corpus, VISION_CFG, None, True,
        md_client=ExplodingMarkItDown())
    assert status == "ok" and method == "pycaption-transcript" and called == [1]
