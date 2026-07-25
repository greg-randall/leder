#!/usr/bin/env python3
"""prepare-2: batch-summarize converted markdown files into _summary.md."""
from __future__ import annotations

import sys
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import tiktoken

from pipeline.llm import call_text_llm

MAX_TOKENS = 10000
MIN_FILE_BYTES = 80
CONTEXT_WINDOW = 800000
PROMPT_RESERVE = 1000
TIKTOKEN_ENCODING = "o200k_base"
MAX_CONTENT_TOKENS = CONTEXT_WINDOW - PROMPT_RESERVE - MAX_TOKENS

_ENC = tiktoken.get_encoding(TIKTOKEN_ENCODING)
_results_lock = threading.Lock()


def build_prepare_2_prompt(corpus_description: str) -> str:
    domain_block = ""
    if corpus_description:
        domain_block = (
            f"This document is from the following collection:\n{corpus_description}\n\n"
        )

    return f"""Read this document. {domain_block}The filename may contain dates
(YYYYMMDD) and topic hints.

These summaries are the search index for a fact-checking agent that will
later need to locate specific statements in the original document. Optimize
for findability: include the exact names, numbers, dates, and distinctive
phrases someone would grep for.

Reply with two sections:

**Summary:** 2-4 sentences describing what kind of document this is and what
it's about. Include the document type (meeting transcript, email, report,
official filing, news article, etc.).

**Facts:** Bullet list of specific numbers, dates, names, amounts, decisions,
or issues that appear in the text. Quote or near-quote. If the document
states a number, include it. If it names a person, company, or place,
include it. Err on the side of including rather than omitting.

- When a fact comes from a specific speaker (public comment, expert
  testimony, staff report), name the speaker in the bullet: "Jerry Carill
  (public comment): alleges 20,000 barrels/day injected." For votes and
  motions, record who moved, the vote count, and the outcome.
- If a speaker corrects themselves, or two speakers contradict each other,
  record both versions explicitly -- these moments matter most to
  fact-checkers.
- These may be auto-generated captions with inconsistent name spellings.
  When a name appears in multiple spellings, list the variants: "Jerry
  Carill (also rendered 'Carville')."
- Where practical, anchor a key fact to a short verbatim quote fragment,
  e.g. "as said: 'contamination in every single groundwater monitoring
  well'" -- this lets a downstream agent grep for the exact phrase.

If the file is empty, boilerplate-only, or unreadable, say so in the
summary and leave facts blank."""


def truncate_to_tokens(text: str, max_tokens: int):
    """Return (text, was_truncated, orig_token_count) on a token boundary."""
    toks = _ENC.encode(text)
    if len(toks) <= max_tokens:
        return text, False, len(toks)
    return _ENC.decode(toks[:max_tokens]), True, len(toks)


def build_user_msg(md_path: Path, content: str):
    content, truncated, orig_tokens = truncate_to_tokens(content, MAX_CONTENT_TOKENS)
    note = ""
    if truncated:
        note = ("\n\n[NOTE: document was truncated to fit the model context "
                "window; content above is the beginning of a larger file.]")
    user_msg = f"## Filename\n{md_path.name}\n\n## Document content\n\n{content}{note}"
    return user_msg, truncated, orig_tokens


def _is_skipped_image_stub(md_path: Path) -> bool:
    """True if this .md is a stub for an image skipped by OCR (tiny/duplicate)."""
    try:
        head = md_path.read_text(encoding="utf-8", errors="replace")[:200]
        return "Image skipped" in head
    except Exception:
        return False


def collect_targets(corpus_root: Path):
    """All .md files eligible for summarization (skip INDEX, *_summary.md, and rollup artifacts)."""
    targets = []
    skip_names = {
        "ALL_SUMMARIES.md", "CORPUS_OVERVIEW.md", "CORPUS_CROSSCUTTING.md",
        "CORPUS_ROLLUP.md", "UNCONVERTED.md", "NEEDS_REVIEW.md",
        "page.md",  # web_cache raw fetched pages — summarized by stage-b instead
    }
    for p in sorted(corpus_root.rglob("*.md")):
        if p.name == "INDEX.md" or p.name.endswith("_summary.md"):
            continue
        if p.name.endswith("_FOLDER_SUMMARY.md") or p.name in skip_names:
            continue
        # Skip image stubs (tiny/duplicate images that were not OCR'd)
        if _is_skipped_image_stub(p):
            continue
        targets.append(p)
    return targets


def summarize_one(md_path: Path, corpus_root: Path, model: str, force: bool,
                  corpus_description: str = ""):
    out_path = md_path.parent / (md_path.stem + "_summary.md")
    if not force and out_path.exists() and out_path.stat().st_size > 20:
        return str(md_path.relative_to(corpus_root)), "skip"

    raw = md_path.read_text(encoding="utf-8", errors="replace")
    if len(raw.strip()) < MIN_FILE_BYTES:
        out_path.write_text(
            "**Summary:** File is too short to summarize (likely empty or stub).\n\n"
            "**Facts:** *(none)*\n", encoding="utf-8")
        return str(md_path.relative_to(corpus_root)), "skip"

    user_msg, truncated, orig_tokens = build_user_msg(md_path, raw)
    try:
        prompt = build_prepare_2_prompt(corpus_description)
        text = call_text_llm(prompt, user_msg, model=model, max_tokens=MAX_TOKENS)
        if truncated:
            banner = (f"> ⚠️ **TRUNCATED:** Source document was {orig_tokens:,} tokens; "
                      f"only the first {MAX_CONTENT_TOKENS:,} tokens were sent to the "
                      f"model. Summary below reflects the beginning of the file only.\n\n")
            text = banner + text
        out_path.write_text(text, encoding="utf-8")
        return str(md_path.relative_to(corpus_root)), ("ok_truncated" if truncated else "ok")
    except Exception as ex:
        print(f"  prepare-2 failed: {md_path.name}: {type(ex).__name__}: {ex}", file=sys.stderr)
        return str(md_path.relative_to(corpus_root)), "fail"


def run_prepare_2(corpus_root: str, model: str, workers: int, force: bool,
                  corpus_description: str = "") -> dict:
    from tqdm import tqdm

    root = Path(corpus_root)
    targets = collect_targets(root)
    counts = {"ok": 0, "skip": 0, "fail": 0, "trunc": 0}

    with tqdm(total=len(targets), desc="prepare-2 summarize", unit="file") as pbar:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(summarize_one, p, root, model, force, corpus_description): p
                for p in targets
            }
            for future in as_completed(futures):
                _, status = future.result()
                with _results_lock:
                    if status == "ok":
                        counts["ok"] += 1
                    elif status == "ok_truncated":
                        counts["ok"] += 1
                        counts["trunc"] += 1
                    elif status == "skip":
                        counts["skip"] += 1
                    else:
                        counts["fail"] += 1
                    pbar.update(1)

    print(f"prepare-2 done: {counts['ok']} summarized "
          f"({counts['trunc']} truncated), {counts['skip']} skipped, {counts['fail']} failed")
    return counts
