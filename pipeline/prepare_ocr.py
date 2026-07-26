#!/usr/bin/env python3
"""OCR fallback for scanned PDFs and image files, with vision escalation.

MarkItDown does no OCR: scanned/image-only PDFs come back empty, and image
files get vision-only (no local text extraction). This module fills that gap:
rasterize/normalize -> tesseract -> escalate thin pages/images to the vision
model with the anti-fabrication prompt from prepare_vision. Local tesseract is
free, so we only pay for vision on the genuinely-hard pages (photos, handwriting,
bad scans).
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import threading as _threading

import pymupdf
import pytesseract
from pytesseract import pytesseract as _pytesseract_impl

from pipeline.prepare_vision import (
    needs_vision, vision_extract, PROVENANCE_BANNER,
)

# pytesseract has no per-call env override, so we replace the module-level
# `environ` it passes to subprocess with our own copy — this scopes
# OMP_THREAD_LIMIT to tesseract's own subprocesses only (avoids OpenMP thread
# contention when many tesseract calls run concurrently under our thread
# pool) without touching the real process environment for other tools.
TESSERACT_ENV = {**os.environ, "OMP_THREAD_LIMIT": "1"}
_pytesseract_impl.environ = TESSERACT_ENV

# Per-run dedup state — cleared by run_prepare_1
_seen_hashes: list = []
_dedup_lock = _threading.Lock()


def _reset_dedup() -> None:
    """Clear dedup state for a fresh run."""
    global _seen_hashes
    with _dedup_lock:
        _seen_hashes = []


def _is_tiny_image(filepath: Path, min_dim: int) -> bool:
    """True if image is <= min_dim in both dimensions."""
    try:
        from PIL import Image
        with Image.open(filepath) as img:
            w, h = img.size
            return w <= min_dim and h <= min_dim
    except Exception:
        return False


def _is_duplicate_image(filepath: Path) -> tuple[bool, Path | None]:
    """True + the matched file's path if a perceptually similar image has
    already been processed."""
    try:
        from imagehash import phash
        from PIL import Image
        with Image.open(filepath) as img:
            h = phash(img)
        with _dedup_lock:
            for seen_hash, seen_path in _seen_hashes:
                if abs(h - seen_hash) <= 4:
                    return True, seen_path
            _seen_hashes.append((h, filepath))
        return False, None
    except Exception:
        return False, None

# Image formats we handle. tesseract (via Leptonica) reads png/jpg/tiff
# directly; gif/bmp/webp are normalized to PNG with Pillow first.
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gif", ".bmp", ".webp")
_TESSERACT_NATIVE = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def _tesseract(img_path: Path) -> str:
    """OCR a single image via tesseract (through pytesseract). Returns '' on failure."""
    try:
        return pytesseract.image_to_string(str(img_path), lang="eng", timeout=120).strip()
    except Exception as ex:
        print(f"  prepare-1 tesseract failed: {img_path.name}: "
              f"{type(ex).__name__}: {ex}", file=sys.stderr)
        return ""


def _normalize_for_tesseract(img_path: Path, tmpdir: Path) -> Path:
    """Return a path tesseract can read; convert odd formats to PNG via Pillow."""
    if img_path.suffix.lower() in _TESSERACT_NATIVE:
        return img_path
    from PIL import Image
    out = tmpdir / (img_path.stem + ".png")
    Image.open(img_path).convert("RGB").save(out, format="PNG")
    return out


def _pdf_to_page_pngs(pdf_path: Path, outdir: Path, dpi: int = 300) -> list[Path]:
    """Rasterize PDF pages to PNGs via PyMuPDF. Returns page image paths, in page order."""
    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception as ex:
        print(f"  prepare-1 pymupdf open failed: {pdf_path.name}: "
              f"{type(ex).__name__}: {ex}", file=sys.stderr)
        return []
    pages: list[Path] = []
    try:
        for i, page in enumerate(doc, 1):
            out = outdir / f"page-{i:04d}.png"
            page.get_pixmap(dpi=dpi).save(str(out))
            pages.append(out)
    except Exception as ex:
        print(f"  prepare-1 pymupdf render failed: {pdf_path.name}: "
              f"{type(ex).__name__}: {ex}", file=sys.stderr)
    finally:
        doc.close()
    return pages


def ocr_image(inpath: Path, md_path: Path, vision_cfg: dict):
    """OCR an image file; escalate to vision when OCR is thin.

    Returns (ok, size, method, note). `note` (non-None) flags the file for
    NEEDS_REVIEW.md. Honors vision_cfg keys: ocr_images, enabled, model, min_words.
    """
    inpath = Path(inpath)
    model = vision_cfg.get("model", "gpt-4o-mini")

    # Skip tiny images (email signatures, social icons, etc.)
    min_dim = vision_cfg.get("min_image_dim", 125)
    if min_dim > 0 and _is_tiny_image(inpath, min_dim):
        rel = os.path.relpath(inpath, md_path.parent)
        content = f"![{inpath.name}]({rel})\n\n*Image skipped — below min_image_dim ({min_dim}px).*\n"
        md_path.write_text(content, encoding="utf-8")
        return True, len(content), "image-tiny", f"skipped — <= {min_dim}x{min_dim}px"

    # Skip duplicate images (same attachment forwarded multiple times)
    if vision_cfg.get("dedup_images", True):
        is_dup, dup_of = _is_duplicate_image(inpath)
        if is_dup:
            rel = os.path.relpath(inpath, md_path.parent)
            dup_note = f" (duplicate of `{dup_of.name}`)" if dup_of else ""
            content = (f"![{inpath.name}]({rel})\n\n"
                       f"*Image skipped — duplicate of previously processed image{dup_note}.*\n")
            md_path.write_text(content, encoding="utf-8")
            return True, len(content), "image-dup", f"skipped — duplicate{dup_note}"

    if not vision_cfg.get("ocr_images", True):
        rel = os.path.relpath(inpath, md_path.parent)
        content = f"![{inpath.name}]({rel})\n\n*Image file - OCR disabled (ocr_images=false).*\n"
        md_path.write_text(content, encoding="utf-8")
        return True, len(content), "image-ref", None

    with tempfile.TemporaryDirectory() as td:
        norm = _normalize_for_tesseract(inpath, Path(td))
        text = _tesseract(norm)

    if vision_cfg.get("enabled") and needs_vision(text, vision_cfg.get("min_words", 20),
                                                    language=vision_cfg.get("language", "en")):
        try:
            vtext = vision_extract(inpath, model=model)
            content = PROVENANCE_BANNER.format(model=model) + vtext
            md_path.write_text(content, encoding="utf-8")
            return True, len(content), "vision", "image OCR thin; recovered via vision"
        except Exception as ex:
            print(f"  prepare-1 vision failed: {inpath.name}: "
                  f"{type(ex).__name__}: {ex}", file=sys.stderr)
            # fall through to whatever OCR produced

    content = f"## OCR: {inpath.name}\n\n{text}\n"
    md_path.write_text(content, encoding="utf-8")
    note = "image transcribed via OCR" if text else "image produced no OCR text"
    return True, len(content), "tesseract", note


def ocr_pdf(inpath: Path, md_path: Path, vision_cfg: dict):
    """OCR a scanned PDF page-by-page; escalate thin pages to vision (capped).

    Returns (ok, size, method, note). ok is False if the PDF could not be
    rasterized or produced no text at all (caller reports it as unconvertible).
    """
    inpath = Path(inpath)
    model = vision_cfg.get("model", "gpt-4o-mini")
    min_words = vision_cfg.get("min_words", 20)
    max_vision_pages = vision_cfg.get("max_pages_per_doc", 30)
    vision_enabled = bool(vision_cfg.get("enabled"))

    with tempfile.TemporaryDirectory() as td:
        pages = _pdf_to_page_pngs(inpath, Path(td))
        if not pages:
            return False, 0, "ocr-pdf", None

        parts: list[str] = []
        vision_used = 0
        cap_hit = False
        for i, png in enumerate(pages, 1):
            text = _tesseract(png)
            if vision_enabled and needs_vision(text, min_words,
                                                  language=vision_cfg.get("language", "en")):
                if vision_used < max_vision_pages:
                    try:
                        vtext = vision_extract(png, model=model)
                        parts.append(f"## Page {i} (vision)\n\n{vtext}")
                        vision_used += 1
                        continue
                    except Exception as ex:
                        print(f"  prepare-1 vision failed (page {i}): {inpath.name}: "
                              f"{type(ex).__name__}: {ex}", file=sys.stderr)
                else:
                    cap_hit = True
            parts.append(f"## Page {i} (OCR)\n\n{text}")

    body = "\n\n".join(parts).strip()
    if not body:
        return False, 0, "ocr-pdf", None

    banner = PROVENANCE_BANNER.format(model=model) if vision_used else ""
    md_path.write_text(banner + body, encoding="utf-8")

    method = "ocr+vision" if vision_used else "ocr"
    note = (f"scanned PDF; {vision_used} page(s) via vision"
            if vision_used else "scanned PDF; OCR only")
    if cap_hit:
        note += f"; vision cap ({max_vision_pages}) reached, remaining pages OCR-only"
    return True, len(body), method, note
