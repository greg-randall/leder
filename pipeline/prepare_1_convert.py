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
import random
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from pipeline.prepare_ocr import IMAGE_EXTS, ocr_image, ocr_pdf
from pipeline.prepare_audio import AUDIO_EXTS, convert_audio, get_whisper_model
from pipeline.prepare_vision import is_garbled

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


def _clean_table_newlines(text: str) -> str:
    """Collapse embedded newlines inside markdown pipe-table cells.

    Some converters preserve line breaks from spreadsheet cells as literal
    newlines, which breaks table rendering. A pipe-table row must be a single
    line — it starts and ends with |. When a row is split across lines (mid-cell
    newlines), we join the fragments back together.

    Only touches lines that contain | (pipe-table rows). Non-table content and
    already-correct rows pass through unchanged. Separate rows are never merged
    because each complete row ends with |.
    """
    lines = text.splitlines()
    out: list[str] = []
    buf: str | None = None  # accumulates a broken row

    for line in lines:
        has_pipe = "|" in line
        if not has_pipe:
            if buf is not None:
                out.append(buf)
                buf = None
            out.append(line)
        elif line.strip().endswith("|"):
            # Row is complete (ends with |)
            if buf is not None:
                out.append(buf + " " + line.strip())
                buf = None
            else:
                out.append(line)
        else:
            # Row is broken mid-cell — accumulate fragments
            if buf is not None:
                buf = buf + " " + line.strip()
            else:
                buf = line.strip()

    if buf is not None:
        out.append(buf)
    return "\n".join(out)


# Only clean table newlines for formats we know produce pipe tables.
# Avoids false positives on documents containing | in prose or code.
_TABLE_FORMATS = {".xlsx", ".xlsm", ".xltx", ".xltm", ".csv", ".tsv", ".ods"}


def _convert_with_markitdown(filepath: Path, md_path: Path, md_client):
    """Try MarkItDown on a file. Returns (ok, size, method)."""
    if md_client is None:
        return False, 0, "markitdown-unavailable"
    try:
        result = md_client.convert(str(filepath))
        text = result.text_content.strip()
        if not text:
            return False, 0, "markitdown-empty"
        if filepath.suffix.lower() in _TABLE_FORMATS:
            text = _clean_table_newlines(text)
        md_path.write_text(text, encoding="utf-8")
        return True, len(text), "markitdown"
    except Exception:
        return False, 0, "markitdown-error"


# ── Gap-fillers ───────────────────────────────────────────────

def _write_md(md_path: Path, content: str):
    md_path.write_text(content, encoding="utf-8")


def _sanitize_filename(name: str) -> str:
    """Strip path separators and nulls from attachment filenames."""
    return name.replace("/", "_").replace("\\", "_").replace("\x00", "")


def _extract_attachments(msg, attach_dir: Path) -> tuple[int, list[str]]:
    """Walk a parsed email message and extract attachments/nested emails.

    Returns (count, notes). Nested .eml files are saved as .eml so
    prepare-1 picks them up on the next pass.
    """
    attach_dir.mkdir(parents=True, exist_ok=True)
    idx = 0
    body_parts = []

    for part in msg.walk():
        content_type = part.get_content_type()
        disp = str(part.get("Content-Disposition", ""))
        filename = part.get_filename()

        if content_type == "message/rfc822":
            # Nested email — save as .eml
            idx += 1
            payload = part.get_payload()
            if isinstance(payload, list):
                payload = payload[0] if payload else None
            if payload:
                nested = payload
                subj = ""
                if hasattr(nested, 'get'):
                    subj = nested.get("Subject", "")
                subj = _sanitize_filename(subj[:60]) if subj else "forwarded"
                out = attach_dir / f"{idx:03d}_{subj}.eml"
                try:
                    out.write_bytes(nested.as_bytes())
                except Exception:
                    out.write_text(str(nested), encoding="utf-8")
                body_parts.append(f"(nested email: {out.name})")

        elif filename:
            # Attachment
            idx += 1
            name = _sanitize_filename(filename)
            out = attach_dir / f"{idx:03d}_{name}"
            try:
                payload = part.get_payload(decode=True)
                if payload:
                    out.write_bytes(payload)
                else:
                    out.write_text(str(part.get_payload()), encoding="utf-8")
            except Exception:
                pass

        elif content_type in ("text/plain", "text/html"):
            try:
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    body_parts.append(payload.decode(charset, errors="replace"))
            except Exception:
                pass

    body = "\n\n".join(body_parts).strip() if body_parts else "(no text body)"
    notes = f"{idx} attachment(s) extracted" if idx else ""
    return idx, body, notes


def convert_eml(inpath: Path, md_path: Path):
    """MIME .eml -> markdown body + extract attachments/nested emails.

    Attachments and nested emails are saved to {stem}_attachments/ with
    sequential numbering so prepare-1 converts them on the next pass.
    """
    from email import policy
    from email.parser import BytesParser
    try:
        with open(inpath, "rb") as f:
            msg = BytesParser(policy=policy.default).parse(f)
        sender = msg.get("From", "(unknown)")
        to = msg.get("To", "(unknown)")
        date = msg.get("Date", "(unknown)")
        subject = msg.get("Subject", "(unknown)")

        attach_dir = md_path.parent / (inpath.stem + "_attachments")
        count, body, notes = _extract_attachments(msg, attach_dir)

        md = (f"# {subject}\n\n**From:** {sender}\n**To:** {to}\n"
              f"**Date:** {date}\n\n---\n\n{body}")
        if count:
            md += (f"\n\n---\n\n**Attachments:** {count} file(s) "
                   f"extracted to [{attach_dir.name}/]({attach_dir.name}/)\n")
        _write_md(md_path, md)
        return True, len(md), "eml-extract", notes or None
    except Exception:
        return False, 0, "eml-extract"


def convert_msg(inpath: Path, md_path: Path):
    """Outlook .msg -> markdown body + extract attachments.

    Same convention as convert_eml: attachments saved to
    {stem}_attachments/ with sequential numbering.
    """
    try:
        import extract_msg as _em
    except ImportError:
        return False, 0, "extract-msg-unavailable"
    try:
        msg = _em.Message(str(inpath))
        sender = msg.sender or "(unknown)"
        to = msg.to or "(unknown)"
        date = str(msg.date) if msg.date else "(unknown)"
        subject = msg.subject or "(unknown)"
        body = msg.body or "(no text body)"

        attach_dir = md_path.parent / (inpath.stem + "_attachments")
        count = 0
        for i, att in enumerate(msg.attachments, 1):
            try:
                name = att.longFilename or att.shortFilename or f"attachment_{i}"
                name = _sanitize_filename(name)
                attach_dir.mkdir(parents=True, exist_ok=True)
                out = attach_dir / f"{i:03d}_{name}"
                with open(str(out), "wb") as f:
                    if isinstance(att.data, bytes):
                        f.write(att.data)
                    else:
                        f.write(str(att.data).encode("utf-8", errors="replace"))
                count += 1
            except Exception:
                pass
        msg.close()

        md = (f"# {subject}\n\n**From:** {sender}\n**To:** {to}\n"
              f"**Date:** {date}\n\n---\n\n{body}")
        if count:
            md += (f"\n\n---\n\n**Attachments:** {count} file(s) "
                   f"extracted to [{attach_dir.name}/]({attach_dir.name}/)\n")
        _write_md(md_path, md)
        return True, len(md), "extract-msg", f"{count} attachment(s) extracted" if count else None
    except Exception:
        return False, 0, "extract-msg"


def convert_tsv(inpath: Path, md_path: Path):
    """TSV -> markdown table (all rows, no cap)."""
    try:
        with open(inpath, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter="\t")
            rows = list(reader)
        if not rows:
            _write_md(md_path, "*(empty)*\n")
            return True, 0, "tsv-table"
        lines = []
        for i, row in enumerate(rows):
            cells = [str(c) for c in row]
            lines.append("| " + " | ".join(cells) + " |")
            if i == 0:
                lines.append("| " + " | ".join(["---"] * len(cells)) + " |")
        content = "\n".join(lines) + "\n"
        _write_md(md_path, content)
        return True, len(content), "tsv-table"
    except Exception:
        return False, 0, "tsv-table"


def convert_xml(inpath: Path, md_path: Path):
    """XML -> fenced code block."""
    try:
        raw = inpath.read_text(encoding="utf-8", errors="replace")
        content = f"```xml\n{raw}\n```\n"
        _write_md(md_path, content)
        return True, len(content), "xml-fenced"
    except Exception:
        return False, 0, "xml-fenced"


def _convert_via_libreoffice(inpath: Path, md_path: Path):
    """Generic LibreOffice headless text conversion.

    Each call gets an isolated -env:UserInstallation profile so concurrent
    headless invocations (under the thread pool) don't collide on the shared
    default profile and hang.
    """
    txt = None
    profile = tempfile.mkdtemp(prefix="lo_profile_")
    try:
        subprocess.run(
            ["libreoffice", f"-env:UserInstallation=file://{profile}",
             "--headless", "--convert-to", "txt:Text",
             "--outdir", str(md_path.parent), str(inpath)],
            capture_output=True, timeout=180,
        )
        txt = md_path.parent / (inpath.stem + ".txt")
        if txt.exists() and txt.stat().st_size > 10:
            content = txt.read_text(encoding="utf-8", errors="replace")
            md_path.write_text(content, encoding="utf-8")
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


def convert_legacy_office(inpath: Path, md_path: Path):
    """Covers .doc .ppt .odt .ods .odp — all LibreOffice headless."""
    return _convert_via_libreoffice(inpath, md_path)


def convert_rtf(inpath: Path, md_path: Path):
    """RTF via pandoc; fallback to LibreOffice."""
    try:
        subprocess.run(
            ["pandoc", str(inpath), "-f", "rtf", "-t", "gfm", "-o", str(md_path)],
            capture_output=True, timeout=60,
        )
        if md_path.exists() and md_path.stat().st_size > 10:
            return True, md_path.stat().st_size, "pandoc-rtf"
    except Exception as ex:
        print(f"  prepare-1 pandoc failed: {inpath.name}: {type(ex).__name__}: {ex}",
              file=sys.stderr)
    return _convert_via_libreoffice(inpath, md_path)


# ── Converter registry ────────────────────────────────────────
# MarkItDown is tried first for non-image/non-audio files; these gap-fillers
# handle formats MarkItDown lacks dedicated converters for (or when it fails).

GAP_FILLERS: dict[str, callable] = {
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
    md_path = out_root / relpath.parent / (relpath.name + ".md")

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
        return _finish(relpath, ocr_image(filepath, md_path, vision_cfg))

    # Audio -> local whisper (MarkItDown uses network Google Web Speech).
    if ext in AUDIO_EXTS:
        ok, size, method, note = convert_audio(filepath, md_path, whisper_model)
        if ok:
            return str(relpath), "ok", method, note
        return str(relpath), "fail", f"audio not transcribed ({method})", None

    # PDF -> MarkItDown first (digital text/tables); OCR fallback if thin.
    if ext == ".pdf":
        if md_client is not None:
            ok, size, method = _convert_with_markitdown(filepath, md_path, md_client)
            if ok:
                text = md_path.read_text(encoding="utf-8", errors="replace")
                if is_meaningful(text) and not is_garbled(
                    text, language=vision_cfg.get("language", "en")):
                    return str(relpath), "ok", method, None
        return _finish(relpath, ocr_pdf(filepath, md_path, vision_cfg))

    # .msg / .eml -> our extractors (MarkItDown only gets body, not attachments).
    if ext == ".msg":
        return _finish(relpath, convert_msg(filepath, md_path))
    if ext == ".eml":
        return _finish(relpath, convert_eml(filepath, md_path))

    # Everything else -> MarkItDown, then gap-fillers.
    if md_client is not None:
        ok, size, method = _convert_with_markitdown(filepath, md_path, md_client)
        if ok:
            return str(relpath), "ok", method, None

    gap_filler = GAP_FILLERS.get(ext)
    if gap_filler is not None:
        try:
            result = gap_filler(filepath, md_path)
            if len(result) == 3:
                result = (*result, None)
            ok, size, method, note = result
            if ok:
                return str(relpath), "ok", method, note or f"used {method} fallback"
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

    files = [f for f in sorted(src_root.rglob("*"))
             if f.is_file()
             and f.suffix.lower() != ".md"
             and f.name != ".gitkeep"
             and not f.name.startswith("._")]  # macOS resource forks
    random.shuffle(files)  # mix heavy and light files for smoother progress
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

    # Second pass: process any new files extracted from email attachments
    new_files = [f for f in sorted(src_root.rglob("*"))
                 if f.is_file()
                 and f.suffix.lower() != ".md"
                 and f.name != ".gitkeep"
                 and not f.name.startswith("._")
                 and f not in futmap.values()]
    if new_files:
        random.shuffle(new_files)
        print(f"\nprepare-1 pass 2: {len(new_files)} extracted file(s) from "
              f"email attachments", file=sys.stderr)
        with tqdm(total=len(new_files), desc="prepare-1 pass 2", unit="file") as pbar2:
            with ThreadPoolExecutor(max_workers=workers) as pool2:
                futmap2 = {}
                for f in new_files:
                    fut = pool2.submit(process_file, f, src_root, out_root,
                                       vision_cfg, whisper_model, force, md_client)
                    futmap2[fut] = f
                for future in as_completed(futmap2):
                    _, status, detail, note = future.result()
                    rel = str(Path(futmap2[future]).relative_to(src_root))
                    with results_lock:
                        if status == "ok":
                            ok_ct += 1
                            if note is not None:
                                needs_review.append((rel, note))
                        elif status == "skip":
                            skip_ct += 1
                        else:
                            failures.append((rel, detail))
                        pbar2.update(1)

    _write_unconverted(out_root, failures)
    _write_needs_review(out_root, needs_review)
    _print_banner(failures)
    print(f"prepare-1 done: {ok_ct} converted, {skip_ct} skipped, "
          f"{len(failures)} failed, {len(needs_review)} recovered-via-fallback")
    return {"failures": failures, "needs_review": needs_review,
            "failure_count": len(failures)}
