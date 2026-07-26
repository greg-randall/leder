"""Tests for the OCR + vision fallback (prepare_ocr)."""
from __future__ import annotations

from pathlib import Path

import pymupdf

import pipeline.prepare_ocr as ocr
from pipeline.prepare_ocr import _is_tiny_image, _is_duplicate_image, _reset_dedup


# ── _tesseract: pytesseract wiring ──────────────────────────────

def test_tesseract_calls_pytesseract(tmp_path, monkeypatch):
    """_tesseract wires straight into pytesseract.image_to_string with lang/timeout."""
    captured = {}

    def fake_image_to_string(image, lang=None, timeout=0):
        captured["image"] = image
        captured["lang"] = lang
        captured["timeout"] = timeout
        return "  recognized text  "

    monkeypatch.setattr(ocr.pytesseract, "image_to_string", fake_image_to_string)
    img = tmp_path / "x.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")

    text = ocr._tesseract(img)

    assert text == "recognized text"  # stripped
    assert captured == {"image": str(img), "lang": "eng", "timeout": 120}


def test_tesseract_failure_returns_empty_string(tmp_path, monkeypatch):
    def boom(image, lang=None, timeout=0):
        raise RuntimeError("tesseract not found")

    monkeypatch.setattr(ocr.pytesseract, "image_to_string", boom)
    img = tmp_path / "x.png"
    img.write_bytes(b"x")
    assert ocr._tesseract(img) == ""


# ── _pdf_to_page_pngs: real PyMuPDF rasterization (self-contained, no binary) ──

def test_pdf_to_page_pngs_real_rasterization(tmp_path):
    doc = pymupdf.open()
    for _ in range(3):
        doc.new_page()
    pdf_path = tmp_path / "three_pages.pdf"
    doc.save(str(pdf_path))
    doc.close()

    outdir = tmp_path / "out"
    outdir.mkdir()
    pages = ocr._pdf_to_page_pngs(pdf_path, outdir, dpi=72)

    assert [p.name for p in pages] == ["page-0001.png", "page-0002.png", "page-0003.png"]
    assert all(p.exists() and p.stat().st_size > 0 for p in pages)


def test_pdf_to_page_pngs_bad_file_returns_empty(tmp_path):
    bad = tmp_path / "not_a_pdf.pdf"
    bad.write_bytes(b"this is not a pdf")
    outdir = tmp_path / "out"
    outdir.mkdir()
    assert ocr._pdf_to_page_pngs(bad, outdir) == []


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


# ── Tiny image filter ──────────────────────────────────────────

def test_is_tiny_image_real(tmp_path):
    """A real small PNG is detected as tiny."""
    from PIL import Image
    img = Image.new("RGB", (50, 50), color="red")
    path = tmp_path / "tiny.png"
    img.save(str(path))
    assert _is_tiny_image(path, 125) is True


def test_is_tiny_image_large(tmp_path):
    """A 500x500 image is not tiny."""
    from PIL import Image
    img = Image.new("RGB", (500, 500), color="blue")
    path = tmp_path / "big.png"
    img.save(str(path))
    assert _is_tiny_image(path, 125) is False


def test_is_tiny_image_fake_png_not_crash(tmp_path):
    """Fake/broken PNGs don't crash — return False."""
    path = tmp_path / "fake.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n")
    assert _is_tiny_image(path, 125) is False


def test_ocr_image_skips_tiny(tmp_path):
    """A tiny image is skipped with image-tiny status, not OCR'd."""
    from PIL import Image
    img = Image.new("RGB", (32, 32), color="red")
    path = tmp_path / "icon.png"
    img.save(str(path))
    vc = {"enabled": False, "min_image_dim": 125}
    ok, size, method, note = ocr.ocr_image(path, tmp_path / "icon.png.md", vc)
    assert ok and method == "image-tiny"
    assert "125x125" in note
    assert "skipped" in (tmp_path / "icon.png.md").read_text()


# ── Duplicate image filter ─────────────────────────────────────

def test_is_duplicate_first_is_false(tmp_path):
    """First image is never a duplicate."""
    _reset_dedup()
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (200, 200), color="white")
    d = ImageDraw.Draw(img)
    d.text((10, 10), "Unique content", fill="black")
    path = tmp_path / "a.png"
    img.save(str(path))
    is_dup, dup_of = _is_duplicate_image(path)
    assert is_dup is False and dup_of is None


def test_is_duplicate_second_is_true(tmp_path):
    """Second identical image is detected as duplicate, and the match names
    the first (surviving) file."""
    _reset_dedup()
    from PIL import Image, ImageDraw

    def make_img():
        img = Image.new("RGB", (200, 200), color="white")
        d = ImageDraw.Draw(img)
        d.text((10, 10), "Same content", fill="black")
        return img

    p1 = tmp_path / "a.png"
    p2 = tmp_path / "b.png"
    make_img().save(str(p1))
    make_img().save(str(p2))
    is_dup1, dup_of1 = _is_duplicate_image(p1)
    assert is_dup1 is False and dup_of1 is None
    is_dup2, dup_of2 = _is_duplicate_image(p2)
    assert is_dup2 is True and dup_of2 == p1


def test_is_duplicate_different_image_is_false(tmp_path):
    """Visually different images are not duplicates."""
    _reset_dedup()
    from PIL import Image, ImageDraw

    # Image 1: white background, black text
    img1 = Image.new("RGB", (200, 200), color="white")
    d1 = ImageDraw.Draw(img1)
    d1.text((10, 10), "Completely different content here", fill="black")
    p1 = tmp_path / "img1.png"
    img1.save(str(p1))

    # Image 2: dark background, white text, different layout
    img2 = Image.new("RGB", (200, 200), color="black")
    d2 = ImageDraw.Draw(img2)
    d2.text((50, 50), "Nothing alike at all", fill="white")
    p2 = tmp_path / "img2.png"
    img2.save(str(p2))

    is_dup1, dup_of1 = _is_duplicate_image(p1)
    assert is_dup1 is False and dup_of1 is None
    is_dup2, dup_of2 = _is_duplicate_image(p2)
    assert is_dup2 is False and dup_of2 is None


def test_ocr_image_skips_duplicate(tmp_path, monkeypatch):
    """Second identical image is skipped with image-dup status."""
    _reset_dedup()
    from PIL import Image, ImageDraw

    def make_img():
        img = Image.new("RGB", (200, 200), color="white")
        d = ImageDraw.Draw(img)
        d.text((10, 10), "Duplicate test", fill="black")
        return img

    def fake_ocr(p):
        return "real text from OCR with enough words to be meaningful content here"
    monkeypatch.setattr(ocr, "_tesseract", fake_ocr)

    p1 = tmp_path / "first.png"
    p2 = tmp_path / "second.png"
    make_img().save(str(p1))
    make_img().save(str(p2))

    vc = {"enabled": False, "dedup_images": True}
    ok1, _, method1, _ = ocr.ocr_image(p1, tmp_path / "first.png.md", vc)
    ok2, _, method2, note2 = ocr.ocr_image(p2, tmp_path / "second.png.md", vc)

    assert ok1 and method1 == "tesseract"  # first gets processed
    assert ok2 and method2 == "image-dup"   # second skipped
    assert "duplicate" in note2


def test_duplicate_image_stub_names_the_surviving_file(tmp_path):
    _reset_dedup()
    vision_cfg = {"enabled": False, "min_words": 20, "min_image_dim": 0, "dedup_images": True}

    import base64
    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first.write_bytes(png_bytes)
    second.write_bytes(png_bytes)

    ocr.ocr_image(first, tmp_path / "first.png.md", vision_cfg)
    ok, size, method, note = ocr.ocr_image(second, tmp_path / "second.png.md", vision_cfg)

    assert method == "image-dup"
    md = (tmp_path / "second.png.md").read_text()
    assert "first.png" in md


def test_duplicate_image_stub_disambiguates_same_basename_in_different_dirs(tmp_path):
    """Two same-named files in different subfolders (common for email
    attachments, e.g. image001.png repeated across many messages) must be
    disambiguated by a relative PATH in the duplicate note, not just a bare
    basename -- a basename alone can't tell the two apart."""
    _reset_dedup()
    vision_cfg = {"enabled": False, "min_words": 20, "min_image_dim": 0, "dedup_images": True}

    import base64
    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    dir_a = tmp_path / "email1_attachments"
    dir_b = tmp_path / "email2_attachments"
    dir_a.mkdir()
    dir_b.mkdir()
    first = dir_a / "image001.png"
    second = dir_b / "image001.png"
    first.write_bytes(png_bytes)
    second.write_bytes(png_bytes)

    ocr.ocr_image(first, dir_a / "image001.png.md", vision_cfg)
    ok, size, method, note = ocr.ocr_image(second, dir_b / "image001.png.md", vision_cfg)

    assert method == "image-dup"
    md = (dir_b / "image001.png.md").read_text()
    # A bare basename ("image001.png") is ambiguous -- both files share it.
    # The note/markdown must include enough of the path to disambiguate.
    assert "email1_attachments" in note
    assert "email1_attachments" in md
