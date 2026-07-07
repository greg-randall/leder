"""Tests for the OCR + vision fallback (prepare_ocr)."""
from __future__ import annotations

from pathlib import Path

import pipeline.prepare_ocr as ocr


def test_ocr_image_thin_escalates_to_vision(tmp_path, monkeypatch):
    img = tmp_path / "x.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    monkeypatch.setattr(ocr, "_tesseract", lambda p: "two words")  # thin -> escalate
    monkeypatch.setattr(ocr, "vision_extract",
                        lambda p, model: "## Transcription\nreal recovered text\n\n## Description\nA form.")
    vc = {"enabled": True, "model": "gpt-4o-mini", "min_words": 20, "ocr_images": True}
    ok, size, method, note = ocr.ocr_image(img, tmp_path / "x.md", vc)
    md = (tmp_path / "x.md").read_text()
    assert ok and method == "vision"
    assert "real recovered text" in md
    assert "verify against original" in md  # provenance banner
    assert note and "vision" in note


def test_ocr_image_rich_ocr_skips_vision(tmp_path, monkeypatch):
    img = tmp_path / "x.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    monkeypatch.setattr(ocr, "_tesseract", lambda p: " ".join(["word"] * 30))
    called = []
    monkeypatch.setattr(ocr, "vision_extract", lambda p, model: called.append(1) or "X")
    vc = {"enabled": True, "model": "m", "min_words": 20, "ocr_images": True}
    ok, size, method, note = ocr.ocr_image(img, tmp_path / "x.md", vc)
    assert ok and method == "tesseract" and called == []


def test_ocr_image_disabled_reference_stub(tmp_path):
    img = tmp_path / "x.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    vc = {"enabled": True, "model": "m", "min_words": 20, "ocr_images": False}
    ok, size, method, note = ocr.ocr_image(img, tmp_path / "x.md", vc)
    md = (tmp_path / "x.md").read_text()
    assert ok and method == "image-ref" and "OCR disabled" in md and note is None


def test_ocr_pdf_page_cap(tmp_path, monkeypatch):
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF")

    def fake_pages(pdf_path, outdir, dpi=300):
        pngs = []
        for i in range(1, 4):  # 3 pages
            p = Path(outdir) / f"page-{i}.png"
            p.write_bytes(b"x")
            pngs.append(p)
        return pngs

    monkeypatch.setattr(ocr, "_pdf_to_page_pngs", fake_pages)
    monkeypatch.setattr(ocr, "_tesseract", lambda p: "thin")  # every page thin -> wants vision
    vcalls = []
    monkeypatch.setattr(ocr, "vision_extract", lambda p, model: vcalls.append(1) or "VISION PAGE")
    vc = {"enabled": True, "model": "m", "min_words": 20, "max_pages_per_doc": 1}
    ok, size, method, note = ocr.ocr_pdf(pdf, tmp_path / "scan.md", vc)
    md = (tmp_path / "scan.md").read_text()
    assert ok and method == "ocr+vision"
    assert len(vcalls) == 1                 # cap of 1 honored
    assert "cap" in note
    assert "VISION PAGE" in md              # page 1 via vision
    assert "Page 3 (OCR)" in md             # overflow pages stay OCR-only


def test_ocr_pdf_no_pages_fails(tmp_path, monkeypatch):
    pdf = tmp_path / "empty.pdf"
    pdf.write_bytes(b"%PDF")
    monkeypatch.setattr(ocr, "_pdf_to_page_pngs", lambda *a, **k: [])
    ok, size, method, note = ocr.ocr_pdf(pdf, tmp_path / "empty", {"enabled": False})
    assert not ok
