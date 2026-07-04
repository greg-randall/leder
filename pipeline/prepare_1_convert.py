#!/usr/bin/env python3
"""prepare-1: convert raw source files to markdown under corpus.root.

Routing per file type:
  - Images (.png/.jpg/.tif/.gif/.bmp/.webp) -> local tesseract OCR, escalating
    thin results to the vision model (prepare_ocr). MarkItDown has no local OCR.
  - Audio (.wav/.mp3/.m4a/.mp4/.flac/.ogg/.aac/.wma) -> local faster-whisper
    (prepare_audio). MarkItDown transcribes over the network via Google Web Speech.
  - PDF -> MarkItDown first (digital text layer + tables); if the result is thin
    (scanned/image-only), fall back to page-by-page OCR + vision (prepare_ocr).
  - Everything else -> MarkItDown, then 7 gap-fillers (.eml .doc .ppt .rtf
    .odt/.ods/.odp .tsv .xml), then UNCONVERTED.md.

Files recovered via OCR/vision/whisper/gap-filler are recorded in NEEDS_REVIEW.md.
"""
from __future__ import annotations

import csv
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from pipeline.prepare_ocr import IMAGE_EXTS, ocr_image, ocr_pdf
from pipeline.prepare_audio import AUDIO_EXTS, convert_audio, get_whisper_model

results_lock = threading.Lock()
TEMP_FILE_PREFIXES = ("~$", "._")
MIN_CONTENT_BYTES = 100

# ── Optional: MarkItDown ──────────────────────────────────────

try:
    from markitdown import MarkItDown as _MarkItDown
    _HAS_MARKITDOWN = True
except ImportError:
    _MarkItDown = None  # type: ignore[assignment]
    _HAS_MARKITDOWN = False


def _get_markitdown():
    """Build a MarkItDown instance, or None if unavailable/construction fails.

    Images and audio are handled by our own OCR/whisper paths, so MarkItDown
    needs no LLM client here — it's used for digital PDFs, office docs, HTML,
    email, archives, etc.
    """
    if not _HAS_MARKITDOWN:
        return None
    try:
        return _MarkItDown()
    except Exception as ex:
        print(f"prepare-1: MarkItDown init failed ({type(ex).__name__}: {ex}); "
              "using gap-fillers only.", file=sys.stderr)
        return None


def _convert_with_markitdown(filepath: Path, outpath: Path, md_client):
    """Try MarkItDown on a file. Returns (ok, size, method)."""
    if md_client is None:
        return False, 0, "markitdown-unavailable"
    try:
        result = md_client.convert(str(filepath))
        text = result.text_content.strip()
        if not text:
            return False, 0, "markitdown-empty"
        outpath.with_suffix(".md").write_text(text, encoding="utf-8")
        return True, len(text), "markitdown"
    except Exception:
        return False, 0, "markitdown-error"


# ── Gap-fillers ───────────────────────────────────────────────

def _write_md(outpath: Path, content: str):
    (outpath.with_suffix(".md")).write_text(content, encoding="utf-8")


def convert_eml(inpath: Path, outpath: Path):
    """MIME .eml -> markdown with From/To/Date/Subject header block + decoded body."""
    from email import policy
    from email.parser import BytesParser
    try:
        with open(inpath, "rb") as f:
            msg = BytesParser(policy=policy.default).parse(f)
        sender = msg.get("From", "(unknown)")
        to = msg.get("To", "(unknown)")
        date = msg.get("Date", "(unknown)")
        subject = msg.get("Subject", "(unknown)")
        body_part = msg.get_body(preferencelist=("plain", "html"))
        body = body_part.get_content() if body_part else ""
        md = (f"# {subject}\n\n**From:** {sender}\n**To:** {to}\n"
              f"**Date:** {date}\n\n---\n\n{body}")
        _write_md(outpath, md)
        return True, len(md), "eml-stdlib"
    except Exception:
        return False, 0, "eml-stdlib"


def convert_tsv(inpath: Path, outpath: Path):
    """TSV -> markdown table (all rows, no cap)."""
    try:
        with open(inpath, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter="\t")
            rows = list(reader)
        if not rows:
            _write_md(outpath, "*(empty)*\n")
            return True, 0, "tsv-table"
        lines = []
        for i, row in enumerate(rows):
            cells = [str(c) for c in row]
            lines.append("| " + " | ".join(cells) + " |")
            if i == 0:
                lines.append("| " + " | ".join(["---"] * len(cells)) + " |")
        content = "\n".join(lines) + "\n"
        _write_md(outpath, content)
        return True, len(content), "tsv-table"
    except Exception:
        return False, 0, "tsv-table"


def convert_xml(inpath: Path, outpath: Path):
    """XML -> fenced code block."""
    try:
        raw = inpath.read_text(encoding="utf-8", errors="replace")
        content = f"```xml\n{raw}\n```\n"
        _write_md(outpath, content)
        return True, len(content), "xml-fenced"
    except Exception:
        return False, 0, "xml-fenced"


def _convert_via_libreoffice(inpath: Path, outpath: Path):
    """Generic LibreOffice headless text conversion.

    Each call gets an isolated -env:UserInstallation profile so concurrent
    headless invocations (under the thread pool) don't collide on the shared
    default profile and hang.
    """
    md = outpath.with_suffix(".md")
    txt = None
    profile = tempfile.mkdtemp(prefix="lo_profile_")
    try:
        subprocess.run(
            ["libreoffice", f"-env:UserInstallation=file://{profile}",
             "--headless", "--convert-to", "txt:Text",
             "--outdir", str(outpath.parent), str(inpath)],
            capture_output=True, timeout=180,
        )
        txt = outpath.parent / (inpath.stem + ".txt")
        if txt.exists() and txt.stat().st_size > 10:
            content = txt.read_text(encoding="utf-8", errors="replace")
            md.write_text(content, encoding="utf-8")
            return True, len(content), "libreoffice"
    except Exception as ex:
        print(f"  prepare-1 libreoffice failed: {inpath.name}: {type(ex).__name__}: {ex}",
              file=sys.stderr)
        return False, 0, "libreoffice"
    finally:
        if txt is not None and txt.exists():
            txt.unlink()
        shutil.rmtree(profile, ignore_errors=True)
    return False, 0, "libreoffice"


def convert_legacy_office(inpath: Path, outpath: Path):
    """Covers .doc .ppt .odt .ods .odp — all LibreOffice headless."""
    return _convert_via_libreoffice(inpath, outpath)


def convert_rtf(inpath: Path, outpath: Path):
    """RTF via pandoc; fallback to LibreOffice."""
    md = outpath.with_suffix(".md")
    try:
        subprocess.run(
            ["pandoc", str(inpath), "-f", "rtf", "-t", "gfm", "-o", str(md)],
            capture_output=True, timeout=60,
        )
        if md.exists() and md.stat().st_size > 10:
            return True, md.stat().st_size, "pandoc-rtf"
    except Exception as ex:
        print(f"  prepare-1 pandoc failed: {inpath.name}: {type(ex).__name__}: {ex}",
              file=sys.stderr)
    return _convert_via_libreoffice(inpath, outpath)


# ── Converter registry ────────────────────────────────────────
# MarkItDown is tried first for non-image/non-audio files; these gap-fillers
# handle formats MarkItDown lacks dedicated converters for (or when it fails).

GAP_FILLERS: dict[str, callable] = {
    ".eml": convert_eml,
    ".doc": convert_legacy_office,
    ".ppt": convert_legacy_office,
    ".odt": convert_legacy_office,
    ".ods": convert_legacy_office,
    ".odp": convert_legacy_office,
    ".rtf": convert_rtf,
    ".tsv": convert_tsv,
    ".xml": convert_xml,
}


# ── Quality gate ──────────────────────────────────────────────

def is_meaningful(content: str) -> bool:
    stripped = content.strip()
    if len(stripped) < MIN_CONTENT_BYTES:
        return False
    alpha_count = sum(1 for c in stripped if c.isalpha())
    return alpha_count >= 10


# ── Dispatch ──────────────────────────────────────────────────

def _finish(relpath: Path, result):
    """Turn an (ok, size, method, note) converter result into a process_file tuple."""
    ok, size, method, note = result
    if ok:
        return str(relpath), "ok", method, note
    return str(relpath), "fail", f"{method} produced no usable text", None


def process_file(filepath: Path, src_root: Path, out_root: Path,
                 vision_cfg: dict, whisper_model, force: bool, md_client=None):
    """Convert one file. Returns (relpath, status, detail, note_or_None)."""
    filepath = Path(filepath)
    name = filepath.name
    ext = filepath.suffix.lower()

    if name.startswith(TEMP_FILE_PREFIXES):
        return str(filepath.relative_to(src_root)), "skip", "temp/lock file", None

    relpath = filepath.relative_to(src_root)
    outpath = out_root / relpath.parent / relpath.stem
    md_path = outpath.with_suffix(".md")

    if not force and md_path.exists() and md_path.stat().st_size > 0:
        try:
            existing = md_path.read_text(encoding="utf-8", errors="replace")
            if is_meaningful(existing):
                return str(relpath), "skip", "already converted", None
        except Exception:
            pass

    (out_root / relpath.parent).mkdir(parents=True, exist_ok=True)

    # Images -> our OCR path (MarkItDown has no local OCR).
    if ext in IMAGE_EXTS:
        return _finish(relpath, ocr_image(filepath, outpath, vision_cfg))

    # Audio -> local whisper (MarkItDown uses network Google Web Speech).
    if ext in AUDIO_EXTS:
        ok, size, method, note = convert_audio(filepath, outpath, whisper_model)
        if ok:
            return str(relpath), "ok", method, note
        return str(relpath), "fail", f"audio not transcribed ({method})", None

    # PDF -> MarkItDown first (digital text/tables); OCR fallback if thin.
    if ext == ".pdf":
        if md_client is not None:
            ok, size, method = _convert_with_markitdown(filepath, outpath, md_client)
            if ok and is_meaningful(md_path.read_text(encoding="utf-8", errors="replace")):
                return str(relpath), "ok", method, None
        return _finish(relpath, ocr_pdf(filepath, outpath, vision_cfg))

    # Everything else -> MarkItDown, then gap-fillers.
    if md_client is not None:
        ok, size, method = _convert_with_markitdown(filepath, outpath, md_client)
        if ok:
            return str(relpath), "ok", method, None

    gap_filler = GAP_FILLERS.get(ext)
    if gap_filler is not None:
        try:
            ok, size, method = gap_filler(filepath, outpath)
            if ok:
                return str(relpath), "ok", method, f"used {method} fallback"
        except Exception as ex:
            print(f"  prepare-1 gap-filler failed: {relpath}: {type(ex).__name__}: {ex}",
                  file=sys.stderr)

    reason = f"no converter for {ext}" if gap_filler is None else "converter returned empty"
    return str(relpath), "fail", reason, None


# ── Reporting ─────────────────────────────────────────────────

def _write_unconverted(out_root: Path, failures: list):
    path = out_root / "UNCONVERTED.md"
    if not failures:
        if path.exists():
            path.unlink()
        return
    lines = ["# UNCONVERTED FILES\n",
             f"**{len(failures)} file(s) could not be converted.**\n\n",
             "| File | Reason |\n| --- | --- |\n"]
    for rel, reason in sorted(failures):
        lines.append(f"| {rel} | {reason} |\n")
    path.write_text("".join(lines), encoding="utf-8")


def _write_needs_review(out_root: Path, needs_review: list):
    path = out_root / "NEEDS_REVIEW.md"
    if not needs_review:
        if path.exists():
            path.unlink()
        return
    lines = ["# NEEDS REVIEW\n",
             f"**{len(needs_review)} document(s) were recovered via OCR, vision, "
             "audio transcription, or a fallback converter.** Verify their markdown "
             "against the originals.\n\n",
             "| File | Note |\n| --- | --- |\n"]
    for rel, note in sorted(needs_review):
        lines.append(f"| {rel} | {note} |\n")
    path.write_text("".join(lines), encoding="utf-8")


def _print_banner(failures: list):
    if not failures:
        return
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  ⚠  {len(failures)} FILE(S) COULD NOT BE CONVERTED")
    print(bar)
    for rel, reason in sorted(failures):
        print(f"  ✗ {rel}  —  {reason}")
    print(f"{bar}\n  See UNCONVERTED.md for the full list.\n{bar}\n")


# ── Entry point ───────────────────────────────────────────────

def run_prepare_1(source_root: str, corpus_root: str, workers: int,
                  vision_cfg: dict, audio_cfg: dict, force: bool) -> dict:
    """Convert every file under source_root into markdown under corpus_root.

    vision_cfg: {enabled, model, min_words, max_pages_per_doc, ocr_images}.
    audio_cfg:  {enabled, model, device}.
    """
    from tqdm import tqdm

    src_root = Path(source_root)
    out_root = Path(corpus_root)
    out_root.mkdir(parents=True, exist_ok=True)

    files = [f for f in sorted(src_root.rglob("*")) if f.is_file()]
    failures: list[tuple[str, str]] = []
    needs_review: list[tuple[str, str]] = []
    ok_ct = skip_ct = 0

    md_client = _get_markitdown()
    if md_client is None:
        print("prepare-1: MarkItDown unavailable; non-image/audio files rely on "
              "gap-fillers only. Install: pip install markitdown[all]", file=sys.stderr)

    # Only load the (large) whisper model if there are audio files to transcribe.
    whisper_model = None
    if audio_cfg.get("enabled") and any(f.suffix.lower() in AUDIO_EXTS for f in files):
        whisper_model, dev = get_whisper_model(
            audio_cfg.get("model", "medium"), audio_cfg.get("device", "auto"))
        if whisper_model is not None:
            print(f"prepare-1: whisper model '{audio_cfg.get('model', 'medium')}' "
                  f"loaded on {dev}", file=sys.stderr)

    with tqdm(total=len(files), desc="prepare-1 convert", unit="file") as pbar:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futmap = {}
            for f in files:
                fut = pool.submit(process_file, f, src_root, out_root,
                                  vision_cfg, whisper_model, force, md_client)
                futmap[fut] = f
            for future in as_completed(futmap):
                _, status, detail, note = future.result()
                rel = str(Path(futmap[future]).relative_to(src_root))
                with results_lock:
                    if status == "ok":
                        ok_ct += 1
                        if note is not None:
                            needs_review.append((rel, note))
                    elif status == "skip":
                        skip_ct += 1
                    else:
                        failures.append((rel, detail))
                    pbar.update(1)

    _write_unconverted(out_root, failures)
    _write_needs_review(out_root, needs_review)
    _print_banner(failures)
    print(f"prepare-1 done: {ok_ct} converted, {skip_ct} skipped, "
          f"{len(failures)} failed, {len(needs_review)} recovered-via-fallback")
    return {"failures": failures, "needs_review": needs_review,
            "failure_count": len(failures)}
