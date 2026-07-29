#!/usr/bin/env python3
"""prepare-1: convert raw source files to markdown under corpus.root.

Routing per file type:
  - Images (.png/.jpg/.tif/.gif/.bmp/.webp) -> local tesseract OCR, escalating
    thin results to the vision model (prepare_ocr). MarkItDown has no local OCR.
  - Audio (.wav/.mp3/.m4a/.mp4/.flac/.ogg/.aac/.wma) -> local faster-whisper
    (prepare_audio). MarkItDown transcribes over the network via Google Web Speech.
  - PDF -> MarkItDown first (digital text layer + tables); if the result is thin
    (scanned/image-only), fall back to page-by-page OCR + vision (prepare_ocr).
  - .srt/.vtt -> pycaption, flattened to a plain transcript (no timestamps).
  - Already-text formats (prepare.text_native_extensions) -> copied verbatim.
  - Everything else -> MarkItDown, then 7 gap-fillers (.eml .doc .ppt .rtf
    .odt/.ods/.odp .xml), then UNCONVERTED.md.

Files recovered via OCR/vision/whisper/gap-filler are recorded in NEEDS_REVIEW.md.
"""
from __future__ import annotations

import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import pycaption
import pymupdf
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait

from pipeline.prepare_ocr import IMAGE_EXTS, ocr_image, ocr_pdf, _reset_dedup
from pipeline.prepare_audio import AUDIO_EXTS, convert_audio, get_whisper_model
from pipeline.prepare_vision import is_garbled
from pipeline.config import DEFAULT_TEXT_NATIVE_EXTS

results_lock = threading.Lock()
TEMP_FILE_PREFIXES = ("~$", "._")
MIN_CONTENT_BYTES = 100  # default; run_prepare_1 may override module state via _configure()
# Short-but-real digital PDFs (e.g. a 1-page memo under 200 chars) may fall
# through to OCR unnecessarily under this per-page threshold; content isn't
# lost, just re-extracted -- an accepted tradeoff, not a bug.
_MIN_CHARS_PER_PAGE = 200
FILE_TIMEOUT = 300  # default; per-file timeout in seconds -- see above
_POLL_INTERVAL = 1.0  # seconds between per-future timeout checks; small relative to FILE_TIMEOUT


def _configure(file_timeout: int, min_content_bytes: int) -> None:
    """Override MIN_CONTENT_BYTES/FILE_TIMEOUT module state for this run.

    These are read by is_meaningful() and the wait()-based timeout loops
    below via module-level globals rather than threaded through every call
    site, matching how they were already used before they became
    configurable (prepare.convert.file_timeout / .min_content_bytes).
    """
    global FILE_TIMEOUT, MIN_CONTENT_BYTES
    FILE_TIMEOUT = file_timeout
    MIN_CONTENT_BYTES = min_content_bytes


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


def _html_to_markdown(html: str, md_client) -> str:
    """Convert an HTML email body to markdown via MarkItDown, writing to a
    temp .html file first (MarkItDown converts by file path). Falls back to
    the raw HTML if MarkItDown is unavailable or fails -- better to ship
    something than crash the whole email conversion over one part.

    Takes md_client rather than building its own -- callers (ultimately the
    batch driver in run_prepare_1) construct MarkItDown exactly once per run
    and thread it down through process_file, the same convention
    _convert_with_markitdown follows; rebuilding it per email here would
    defeat that for a batch of thousands of messages.
    """
    if md_client is None:
        return html
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td) / "body.html"
            tmp_path.write_text(html, encoding="utf-8")
            result = md_client.convert(str(tmp_path))
            return result.text_content.strip()
    except Exception:
        return html


def _extract_attachments(msg, attach_dir: Path, md_client=None) -> tuple[int, list[str]]:
    """Walk a parsed email message and extract attachments/nested emails.

    Returns (count, notes). Nested .eml files are saved as .eml so
    prepare-1 picks them up on the next pass.

    md_client: shared MarkItDown instance (built once by the batch driver),
    used only if an HTML-only body needs converting -- see _html_to_markdown.
    """
    attach_dir.mkdir(parents=True, exist_ok=True)
    idx = 0
    body_parts: list[str] = []
    plain_parts: list[str] = []
    html_parts: list[str] = []

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

        elif content_type == "text/plain":
            try:
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    plain_parts.append(payload.decode(charset, errors="replace"))
            except Exception:
                pass

        elif content_type == "text/html":
            try:
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    html_parts.append(payload.decode(charset, errors="replace"))
            except Exception:
                pass

    # Prefer text/plain when present -- avoids concatenating near-duplicate
    # plain + HTML renderings of the same message. HTML-only bodies are
    # converted to markdown rather than dumped as raw tag soup.
    if plain_parts:
        text_body = "\n\n".join(plain_parts).strip()
    elif html_parts:
        text_body = _html_to_markdown("\n\n".join(html_parts), md_client).strip()
    else:
        text_body = ""

    # attachment/nested-email markers are grouped before body text, not
    # interleaved by original MIME position
    all_parts = body_parts + ([text_body] if text_body else [])
    body = "\n\n".join(all_parts).strip() if all_parts else "(no text body)"
    notes = f"{idx} attachment(s) extracted" if idx else ""
    return idx, body, notes


def convert_eml(inpath: Path, md_path: Path, md_client=None):
    """MIME .eml -> markdown body + extract attachments/nested emails.

    Attachments and nested emails are saved to {stem}_attachments/ with
    sequential numbering so prepare-1 converts them on the next pass.

    md_client: shared MarkItDown instance from the batch driver, forwarded
    to _extract_attachments for HTML-only bodies (see _html_to_markdown).
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

        attach_dir = md_path.parent / (md_path.stem + "_attachments")
        count, body, notes = _extract_attachments(msg, attach_dir, md_client)

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

        attach_dir = md_path.parent / (md_path.stem + "_attachments")
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


def passthrough_text(inpath: Path, md_path: Path):
    """Copy an already-text file straight into its .md sidecar — no conversion.

    Text-native formats (.csv, .txt, .json, source code, …) are already
    readable; running them through MarkItDown only reformats/bloats them (a CSV
    becomes a giant pipe table). We copy the raw bytes verbatim, so the sidecar
    is a byte-for-byte twin of the source (`diff src src.md` is empty) — no
    encoding or newline (\\r\\n → \\n) translation touches it. Which extensions
    land here is driven by prepare.text_native_extensions in config.yaml.
    """
    try:
        shutil.copyfile(inpath, md_path)
        return True, md_path.stat().st_size, "text-passthrough", None
    except Exception as ex:
        return False, 0, f"text-passthrough ({type(ex).__name__}: {ex})", None


def convert_xml(inpath: Path, md_path: Path):
    """XML -> fenced code block."""
    try:
        raw = inpath.read_text(encoding="utf-8", errors="replace")
        content = f"```xml\n{raw}\n```\n"
        _write_md(md_path, content)
        return True, len(content), "xml-fenced"
    except Exception:
        return False, 0, "xml-fenced"


def convert_html(inpath: Path, md_path: Path):
    """HTML -> markdown via markdownify.

    MarkItDown is tried first (see the dispatch in process_file) but returns an
    empty string on some real-world pages -- notably JSF/PrimeFaces apps such as
    the RRC inspection lookup, where the record is a grid of
    `<label>Field:</label><span>value</span>` pairs rather than a <table>.

    markdownify is used rather than a main-content extractor (trafilatura, or
    Jina Reader's /r endpoint): both treat those header panels as boilerplate
    and strip the fields identifying which facility, operator and lease the
    record belongs to, keeping only the body. For a document corpus we want the
    whole page, and we want `Field: value` to stay on one line so a
    fact-checking agent can grep for a value and land on its label.
    """
    try:
        from bs4 import BeautifulSoup, NavigableString
        from markdownify import markdownify as _md
    except ImportError:
        return False, 0, "html-markdownify"

    try:
        raw = inpath.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(raw, "html.parser")
        # Scripts/styles would otherwise survive as literal text.
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        # markdownify preserves source whitespace, so a pretty-printed
        # `<label>Field:</label>\n<span>value</span>` would come out split
        # across two lines -- the exact break this converter exists to avoid.
        # Drop whitespace-only nodes sitting between a label and its value.
        for label in soup.find_all("label"):
            sib = label.next_sibling
            if isinstance(sib, NavigableString) and not sib.strip():
                sib.extract()
        content = _md(str(soup))
        # markdownify leaves long runs of blank lines where layout divs were.
        content = re.sub(r"\n{3,}", "\n\n", content).strip() + "\n"
        if not is_meaningful(content):
            return False, 0, "html-markdownify"
        _write_md(md_path, content)
        return True, len(content), "html-markdownify"
    except Exception:
        return False, 0, "html-markdownify"


def _normalize_vtt_header(text: str) -> str:
    """pycaption requires the blank line immediately after WEBVTT, but many
    generators (e.g. yt-dlp's downloaded YouTube captions) insert extra
    metadata lines ('Kind: captions', 'Language: en') before it. Collapse
    everything between WEBVTT and the first truly-blank line down to just
    WEBVTT, so real-world files aren't rejected on this technicality.
    """
    lines = text.splitlines()
    if lines and lines[0].startswith("WEBVTT"):
        try:
            blank_idx = lines.index("", 1)
            lines = [lines[0]] + lines[blank_idx:]
        except ValueError:
            pass
    return "\n".join(lines)


_SRT_TIMESTAMP_RE = re.compile(r'(\d{2}:\d{2}:\d{2}),(\d{3})')


def _srt_to_vtt_text(text: str) -> str:
    """Rewrite raw SRT text into WebVTT syntax, so it can go through
    pycaption's WebVTTReader instead of its SRTReader.

    pycaption's SRTReader mishandles YouTube's "rolling caption" blank
    placeholder line when it appears WITHIN a cue block, before the real
    text arrives: it mistakes the placeholder for the separator BETWEEN
    cues, discards the cue, then aborts the whole file on the leftover
    text — confirmed on a real corpus of YouTube-exported .srt files,
    where this pattern is dominant (20/22 files failed this way, each
    losing 100% of their content). pycaption's WebVTTReader only closes a
    cue once it has actually collected text nodes, so an empty line with
    nothing collected yet is correctly ignored rather than treated as a
    cue boundary — the same robustness that led us to pycaption over
    webvtt-py for .vtt in the first place. Converting SRT's comma-decimal
    timestamps to VTT's period-decimal and prepending a WEBVTT header is
    enough for that already-trusted path to parse the same file
    correctly. Numeric SRT cue-index lines are left as-is; WebVTT treats
    an optional line before the timing line as a harmless cue identifier.
    """
    lines = text.splitlines()
    vtt_lines = ["WEBVTT", ""]
    for line in lines:
        if "-->" in line:
            line = _SRT_TIMESTAMP_RE.sub(r'\1.\2', line)
        vtt_lines.append(line)
    return "\n".join(vtt_lines)


def _drop_empty_cues(text: str) -> str:
    """Remove cue blocks whose content is entirely blank/whitespace.

    pycaption's WebVTTReader only resets its "inside a cue" state when a
    blank line arrives AFTER it has collected at least one real text node
    for that cue. A cue whose only content is whitespace (a common
    YouTube "still rolling, no text yet" placeholder) never collects a
    real node — even a lone-space line strips to nothing inside
    pycaption's own cue-text parser — so that blank-line reset never
    fires. The parser stays stuck "inside" that cue, and the NEXT cue's
    plain identifier line gets wrongly ingested as if it were real cue
    text, silently dropping that next cue's actual content. Confirmed on
    a real corpus: not pre-filtering these left real spoken content
    missing with no error.

    Cue boundaries here are found via the unambiguous "-->" timing line
    (not blank lines, which appear both as real separators and as
    in-cue placeholders — that ambiguity is the whole problem above).
    Dropping a wholly-empty cue is also the semantically correct outcome:
    it has nothing to contribute to the transcript anyway.
    """
    lines = text.splitlines()
    timing_idxs = [i for i, line in enumerate(lines) if "-->" in line]
    if not timing_idxs:
        return text

    def cue_start(idx: int) -> int:
        # A non-blank, non-timing line immediately before a timing line
        # is that cue's (optional, WebVTT-legal) identifier line.
        if idx > 0 and lines[idx - 1].strip() != "" and "-->" not in lines[idx - 1]:
            return idx - 1
        return idx

    starts = [cue_start(i) for i in timing_idxs]
    ends = starts[1:] + [len(lines)]

    drop_idxs: set[int] = set()
    for timing_idx, start, end in zip(timing_idxs, starts, ends):
        content = lines[timing_idx + 1:end]
        if not any(line.strip() for line in content):
            drop_idxs.update(range(start, end))

    if not drop_idxs:
        return text
    return "\n".join(line for i, line in enumerate(lines) if i not in drop_idxs)


def convert_subtitle(inpath: Path, md_path: Path):
    """SRT/WebVTT -> flowing transcript text via pycaption's WebVTTReader.

    Cue numbers and timestamps are dropped — the pipeline only needs to
    confirm a quote is IN the source, not when it was said (the original
    file is untouched in source_root if a timestamp is ever needed).
    Consecutive duplicate lines (common in auto-generated "rolling"
    captions, where each cue repeats the previous line) are collapsed so
    the same sentence doesn't appear back-to-back.

    Both .srt and .vtt route through the WebVTTReader (see
    _srt_to_vtt_text for why .srt is converted rather than parsed by
    pycaption's own SRTReader), with wholly-empty cues pre-filtered out
    (see _drop_empty_cues for why that has to happen before parsing,
    not after).
    """
    try:
        text = inpath.read_text(encoding="utf-8", errors="replace")
        if inpath.suffix.lower() == ".srt":
            text = _srt_to_vtt_text(text)
        else:
            text = _normalize_vtt_header(text)
        text = _drop_empty_cues(text)
        caption_set = pycaption.WebVTTReader().read(text)
    except Exception as ex:
        print(f"  prepare-1 pycaption failed: {inpath.name}: {type(ex).__name__}: {ex}",
              file=sys.stderr)
        return False, 0, "pycaption-error", None

    langs = caption_set.get_languages()
    captions = caption_set.get_captions(langs[0]) if langs else []

    lines: list[str] = []
    last_line = None
    for caption in captions:
        for line in caption.get_text().splitlines():
            line = line.strip()
            if line and line != last_line:
                lines.append(line)
                last_line = line

    transcript = " ".join(lines).strip()
    if not transcript:
        return False, 0, "pycaption-empty", None
    header = (
        f"# Transcript: {inpath.name}\n\n"
        "*Extracted from a caption file; cue numbers and timestamps removed; "
        "consecutive duplicate caption lines collapsed. Original file preserved "
        "alongside.*\n\n---\n\n"
    )
    content = header + transcript + "\n"
    _write_md(md_path, content)
    return True, len(content), "pycaption-transcript", None


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
    ".xml": convert_xml,
    ".html": convert_html,
    ".htm": convert_html,
}


# ── Quality gate ──────────────────────────────────────────────

def is_meaningful(content: str) -> bool:
    stripped = content.strip()
    if len(stripped) < MIN_CONTENT_BYTES:
        return False
    alpha_count = sum(1 for c in stripped if c.isalpha())
    return alpha_count >= 10


def _pdf_page_count(filepath: Path) -> int:
    """Return a PDF's page count, or 1 if it can't be opened (fail open --
    a page count of 1 makes the per-page density check equivalent to the
    old absolute-length check for a PDF that can't even be introspected)."""
    try:
        with pymupdf.open(str(filepath)) as doc:
            return max(1, doc.page_count)
    except Exception:
        return 1


# ── Dispatch ──────────────────────────────────────────────────

def _reflow_md_file_in_place(md_path: Path) -> None:
    from pipeline.prepare_reflow import reflow_pipeline_text
    text = md_path.read_text(encoding="utf-8", errors="replace")
    reflowed = reflow_pipeline_text(text)
    if reflowed != text:
        md_path.write_text(reflowed, encoding="utf-8")


def _finish(relpath: Path, result, md_path: Path | None = None, reflow: bool = True):
    """Turn an (ok, size, method, note) converter result into a process_file tuple.

    Re-flows the written .md file in place on success, unless reflow=False
    (used only for passthrough_text, whose byte-for-byte-identical guarantee
    must never be touched).
    """
    ok, size, method, note = result
    if ok:
        if reflow and md_path is not None:
            _reflow_md_file_in_place(md_path)
        return str(relpath), "ok", method, note
    return str(relpath), "fail", f"{method} produced no usable text", None


def process_file(filepath: Path, src_root: Path, out_root: Path,
                 vision_cfg: dict, whisper_model, force: bool, md_client=None,
                 text_native_exts: set[str] | None = None,
                 audio_cfg: dict | None = None):
    """Convert one file. Returns (relpath, status, detail, note_or_None)."""
    if text_native_exts is None:
        text_native_exts = {e.lower() for e in DEFAULT_TEXT_NATIVE_EXTS}
    filepath = Path(filepath)
    name = filepath.name
    ext = filepath.suffix.lower()

    if name.startswith(TEMP_FILE_PREFIXES):
        return str(filepath.relative_to(src_root)), "skip", "temp/lock file", None

    relpath = filepath.relative_to(src_root)
    md_path = out_root / relpath.parent / (relpath.name + ".md")

    if not force and md_path.exists() and md_path.stat().st_size > 0:
        try:
            source_is_newer = filepath.stat().st_mtime > md_path.stat().st_mtime
            existing = md_path.read_text(encoding="utf-8", errors="replace")
            if not source_is_newer and is_meaningful(existing):
                return str(relpath), "skip", "already converted", None
        except Exception:
            pass

    (out_root / relpath.parent).mkdir(parents=True, exist_ok=True)

    # Images -> our OCR path (MarkItDown has no local OCR).
    if ext in IMAGE_EXTS:
        return _finish(relpath, ocr_image(filepath, md_path, vision_cfg), md_path)

    # Audio -> local whisper (MarkItDown uses network Google Web Speech).
    if ext in AUDIO_EXTS:
        vocabulary = audio_cfg.get("vocabulary") if audio_cfg else None
        ok, size, method, note = convert_audio(filepath, md_path, whisper_model, vocabulary=vocabulary)
        if ok:
            _reflow_md_file_in_place(md_path)
            return str(relpath), "ok", method, note
        return str(relpath), "fail", f"audio not transcribed ({method})", None

    # PDF -> MarkItDown first (digital text/tables); OCR fallback if thin,
    # including "thin relative to page count" (e.g. one digital cover page
    # on an otherwise-scanned document, which an absolute-length check alone
    # would wrongly accept).
    if ext == ".pdf":
        if md_client is not None:
            ok, size, method = _convert_with_markitdown(filepath, md_path, md_client)
            if ok:
                text = md_path.read_text(encoding="utf-8", errors="replace")
                page_count = _pdf_page_count(filepath)
                dense_enough = len(text.strip()) >= _MIN_CHARS_PER_PAGE * page_count
                if (is_meaningful(text) and dense_enough
                        and not is_garbled(text, language=vision_cfg.get("language", "en"))):
                    _reflow_md_file_in_place(md_path)
                    return str(relpath), "ok", method, None
        return _finish(relpath, ocr_pdf(filepath, md_path, vision_cfg), md_path)

    # .msg / .eml -> our extractors (MarkItDown only gets body, not attachments).
    if ext == ".msg":
        return _finish(relpath, convert_msg(filepath, md_path), md_path)
    if ext == ".eml":
        return _finish(relpath, convert_eml(filepath, md_path, md_client), md_path)

    # Already-text formats (.csv, .txt, .json, source code, …) -> copy verbatim.
    # These are readable as-is; the converter would only reformat/bloat them.
    if ext in text_native_exts:
        return _finish(relpath, passthrough_text(filepath, md_path), md_path, reflow=False)

    # .srt / .vtt -> our own pycaption based extractor (not through
    # MarkItDown, so we control the transcript cleanup ourselves).
    if ext in (".srt", ".vtt"):
        return _finish(relpath, convert_subtitle(filepath, md_path), md_path)

    # Everything else -> MarkItDown, then gap-fillers.
    md_method = None
    if md_client is not None:
        ok, size, method = _convert_with_markitdown(filepath, md_path, md_client)
        if ok:
            _reflow_md_file_in_place(md_path)
            return str(relpath), "ok", method, None
        # Remember *why* MarkItDown declined -- "markitdown-empty" (converted
        # to nothing) and "markitdown-error" (threw) are very different
        # diagnoses, and the old reason string discarded both.
        md_method = method

    gap_filler = GAP_FILLERS.get(ext)
    if gap_filler is not None:
        try:
            result = gap_filler(filepath, md_path)
            if len(result) == 3:
                result = (*result, None)
            ok, size, method, note = result
            if ok:
                _reflow_md_file_in_place(md_path)
                return str(relpath), "ok", method, note or f"used {method} fallback"
        except Exception as ex:
            print(f"  prepare-1 gap-filler failed: {relpath}: {type(ex).__name__}: {ex}",
                  file=sys.stderr)

    # Report what actually happened. "no converter for .html" was previously
    # emitted whenever no gap-filler existed, even when MarkItDown had run and
    # returned an empty document -- which reads as "unsupported format" and
    # sends the next reader down the wrong path entirely.
    if gap_filler is not None:
        reason = "converter returned empty"
    elif md_method:
        reason = f"no gap-filler for {ext} after {md_method}"
    else:
        reason = f"no converter for {ext}"
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
                  vision_cfg: dict, audio_cfg: dict, force: bool,
                  text_native_exts: set[str] | None = None,
                  convert_cfg: dict | None = None) -> dict:
    """Convert every file under source_root into markdown under corpus_root.

    vision_cfg: {enabled, model, min_words, max_pages_per_doc, ocr_images}.
    audio_cfg:  {enabled, model, device, vocabulary}.
    text_native_exts: extensions copied through verbatim (see config.yaml
        prepare.text_native_extensions). Defaults to DEFAULT_TEXT_NATIVE_EXTS.
    convert_cfg: {file_timeout, min_content_bytes} (see config.yaml
        prepare.convert). Defaults to the module's hardcoded defaults.
    """
    from tqdm import tqdm

    convert_cfg = convert_cfg or {}
    _configure(convert_cfg.get("file_timeout", 300), convert_cfg.get("min_content_bytes", 100))

    if text_native_exts is None:
        text_native_exts = {e.lower() for e in DEFAULT_TEXT_NATIVE_EXTS}

    src_root = Path(source_root)
    out_root = Path(corpus_root)
    out_root.mkdir(parents=True, exist_ok=True)

    _reset_dedup()  # fresh dedup state for this run

    files = [f for f in sorted(src_root.rglob("*"))
             if f.is_file()
             and f.suffix.lower() != ".md"
             and f.name != ".gitkeep"
             and not f.name.startswith("._")]  # macOS resource forks
    random.shuffle(files)  # mix heavy and light files for smoother progress
    failures: list[tuple[str, str]] = []
    needs_review: list[tuple[str, str]] = []
    ok_ct = skip_ct = tiny_ct = dup_ct = 0

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
        # Not a context manager: a genuinely-hung converter thread can never
        # be forcibly killed (Python has no API to terminate a running
        # thread), so `with ThreadPoolExecutor(...) as pool:` would block
        # run_prepare_1 forever at teardown -- __exit__ calls
        # shutdown(wait=True), which joins every worker thread including
        # the stuck one. Instead we shut down with wait=False below so this
        # function can return/continue even while a worker is still stuck;
        # the abandoned thread keeps running in the background until it
        # naturally finishes or the process exits. Note this only protects
        # run_prepare_1 ITSELF: concurrent.futures.thread registers a
        # process-wide atexit hook (_python_exit) that joins EVERY thread any
        # ThreadPoolExecutor ever created, regardless of this pool's own
        # shutdown(wait=...) call -- so a standalone `prepare-1` CLI run may
        # still hang at interpreter exit after run_prepare_1 has already
        # returned and correctly reported the timeout. That's an accepted,
        # inherent Python limitation (threads can't be forcibly killed); a
        # real fix would mean switching to ProcessPoolExecutor, out of scope.
        pool = ThreadPoolExecutor(max_workers=workers)
        futmap = {}
        for f in files:
            fut = pool.submit(process_file, f, src_root, out_root,
                              vision_cfg, whisper_model, force, md_client,
                              text_native_exts, audio_cfg)
            futmap[fut] = f
        # Each future is judged against its OWN start time, not against "has
        # anything in the remaining set completed recently" -- re-arming
        # as_completed(pending, timeout=FILE_TIMEOUT) every iteration resets
        # its deadline on ANY completion in `pending`, so with multiple
        # workers one truly-hung file could hide behind other workers'
        # steady progress for the whole rest of the run. Polling wait() on a
        # short interval and comparing elapsed time per-future avoids that.
        start_times = {fut: time.monotonic() for fut in futmap}
        pending = set(futmap.keys())
        try:
            while pending:
                done, pending = wait(pending, timeout=_POLL_INTERVAL,
                                     return_when=FIRST_COMPLETED)
                for future in done:
                    f = futmap[future]
                    rel = str(Path(f).relative_to(src_root))
                    try:
                        _, status, method, note = future.result()
                    except Exception as ex:
                        status, method, note = "fail", f"error: {type(ex).__name__}: {ex}", None
                    with results_lock:
                        if status == "ok":
                            ok_ct += 1
                            if method == "image-tiny":
                                tiny_ct += 1
                            elif method == "image-dup":
                                dup_ct += 1
                            if note is not None:
                                needs_review.append((rel, note))
                        elif status == "skip":
                            skip_ct += 1
                        else:
                            failures.append((rel, method))
                        pbar.update(1)

                now = time.monotonic()
                timed_out_now = {fut for fut in pending
                                 if now - start_times[fut] > FILE_TIMEOUT}
                if timed_out_now:
                    stalled = [str(Path(futmap[fut]).relative_to(src_root)) for fut in timed_out_now]
                    print(f"\n  ⚠  {len(stalled)} file(s) still running after "
                          f"{FILE_TIMEOUT}s and were not waited on further: "
                          f"{', '.join(stalled[:5])}"
                          f"{' ...' if len(stalled) > 5 else ''}", file=sys.stderr)
                    with results_lock:
                        for fut in timed_out_now:
                            pending.discard(fut)
                            rel = str(Path(futmap[fut]).relative_to(src_root))
                            failures.append((rel, f"timed out after {FILE_TIMEOUT}s"))
                            pbar.update(1)
        finally:
            pool.shutdown(wait=False)

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
            # See the pass-1 comments above: not a context manager (a
            # genuinely-hung converter thread can't be killed, so
            # shutdown(wait=True) at __exit__ would block forever); the
            # standalone CLI process may still be delayed at interpreter
            # exit by Python's atexit thread-join regardless of
            # shutdown(wait=False) here -- accepted, inherent limitation.
            # Per-future start times avoid the global-stall-detector trap of
            # re-arming as_completed() on the whole remaining set each pass.
            pool2 = ThreadPoolExecutor(max_workers=workers)
            futmap2 = {}
            for f in new_files:
                fut = pool2.submit(process_file, f, src_root, out_root,
                                   vision_cfg, whisper_model, force, md_client,
                                   text_native_exts, audio_cfg)
                futmap2[fut] = f
            start_times2 = {fut: time.monotonic() for fut in futmap2}
            pending2 = set(futmap2.keys())
            try:
                while pending2:
                    done2, pending2 = wait(pending2, timeout=_POLL_INTERVAL,
                                           return_when=FIRST_COMPLETED)
                    for future in done2:
                        f2 = futmap2[future]
                        rel = str(Path(f2).relative_to(src_root))
                        try:
                            _, status, method, note = future.result()
                        except Exception as ex:
                            status, method, note = "fail", f"error: {type(ex).__name__}: {ex}", None
                        with results_lock:
                            if status == "ok":
                                ok_ct += 1
                                if method == "image-tiny":
                                    tiny_ct += 1
                                elif method == "image-dup":
                                    dup_ct += 1
                                if note is not None:
                                    needs_review.append((rel, note))
                            elif status == "skip":
                                skip_ct += 1
                            else:
                                failures.append((rel, method))
                            pbar2.update(1)

                    now = time.monotonic()
                    timed_out_now2 = {fut for fut in pending2
                                      if now - start_times2[fut] > FILE_TIMEOUT}
                    if timed_out_now2:
                        stalled2 = [str(Path(futmap2[fut]).relative_to(src_root)) for fut in timed_out_now2]
                        print(f"\n  ⚠  {len(stalled2)} file(s) still running after "
                              f"{FILE_TIMEOUT}s and were not waited on further: "
                              f"{', '.join(stalled2[:5])}"
                              f"{' ...' if len(stalled2) > 5 else ''}", file=sys.stderr)
                        with results_lock:
                            for fut in timed_out_now2:
                                pending2.discard(fut)
                                rel = str(Path(futmap2[fut]).relative_to(src_root))
                                failures.append((rel, f"timed out after {FILE_TIMEOUT}s"))
                                pbar2.update(1)
            finally:
                pool2.shutdown(wait=False)

    _write_unconverted(out_root, failures)
    _write_needs_review(out_root, needs_review)
    _print_banner(failures)

    # Detect nested email attachments — user may need another pass
    nested = [d for d in sorted(out_root.rglob("*_attachments"))
              if d.is_dir() and any(d.glob("*.eml")) or any(d.glob("*.msg"))]
    if nested:
        print(f"\n  ⚠  {len(nested)} attachment folder(s) contain .eml/.msg files "
              f"— re-run prepare-1 to process them.",
              file=sys.stderr)

    details = []
    if tiny_ct:
        details.append(f"{tiny_ct} tiny images skipped")
    if dup_ct:
        details.append(f"{dup_ct} duplicates skipped")
    detail_str = f" ({', '.join(details)})" if details else ""
    print(f"prepare-1 done: {ok_ct} converted{detail_str}, {skip_ct} skipped, "
          f"{len(failures)} failed, {len(needs_review)} recovered-via-fallback")
    return {"failures": failures, "needs_review": needs_review,
            "failure_count": len(failures)}
