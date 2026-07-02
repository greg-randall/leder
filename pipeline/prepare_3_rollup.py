#!/usr/bin/env python3
"""prepare-3: recursive folder-summary rollup + crosscutting corpus overview.

Post-order over the corpus tree: every folder gets a _FOLDER_SUMMARY.md
synthesized from its direct *_summary.md files plus its immediate subfolders'
_FOLDER_SUMMARY.md. Root summary is aliased to CORPUS_OVERVIEW.md. A single
giant call over all *_summary.md (default on) writes CORPUS_CROSSCUTTING.md.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from pipeline.llm import call_text_llm

BIG_CALL_MAX_TOKENS = 384000
FOLDER_MAX_TOKENS = 12000

FOLDER_PROMPT = """You are writing an overview of ONE folder in a document
collection. The folder may contain individual document summaries and/or
summaries of sub-collections (subfolders). Synthesize them into a single
overview.

Write these sections:

**Overview:** 3-5 sentences describing what this folder contains as a whole.

**Key contents & findings:** Bullet list of the substantive matters across the
documents and sub-collections — who/what/where, notable numbers, dates, names,
decisions, deficiencies, or issues. Cite specifics where the inputs give them.

**Open questions / gaps:** Anything the inputs suggest is unresolved, missing,
or unclear.

Stay grounded in what the summaries actually say. If the folder is thin or
mostly administrative, say so rather than padding."""

CROSSCUTTING_PROMPT = """You are writing the definitive analytical overview for
an ENTIRE document collection. Below are the per-document summaries for every
document across every folder. You have the complete picture — draw connections
across folders that no single-folder view could reveal.

Write these sections:

**What this collection is:** A few paragraphs describing the dataset as a whole —
scale, the range of subjects/sources, the time span, and the common structure.

**Cross-cutting patterns:** Recurring themes, issues, people, or organizations
that appear across multiple folders. Cite specific folders and numbers.

**Notable / outlier items:** Documents or folders that stand out and why.

**Data highlights:** Concrete quantitative findings worth surfacing.

**Suggested next analyses:** Specific, grounded follow-ups.

Be thorough — you have a large output budget. Stay grounded in the summaries;
do not invent facts not present in them."""


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace").strip()


def gather_folder_inputs(folder: Path):
    """(file_summaries, subfolder_summaries) for one folder.
    file_summaries: (rel_name, text) for direct *_summary.md.
    subfolder_summaries: (subdir_name, text) for immediate subdir _FOLDER_SUMMARY.md."""
    file_summaries = []
    for p in sorted(folder.glob("*_summary.md")):
        text = _read(p)
        if text:
            file_summaries.append((p.name.replace("_summary.md", ""), text))
    subfolder_summaries = []
    for d in sorted(c for c in folder.iterdir() if c.is_dir()):
        fs = d / "_FOLDER_SUMMARY.md"
        if fs.exists():
            text = _read(fs)
            if text:
                subfolder_summaries.append((d.name, text))
    return file_summaries, subfolder_summaries


def build_folder_user_msg(folder_name, file_summaries, subfolder_summaries) -> str:
    parts = [f"# Folder: {folder_name}\n"]
    if subfolder_summaries:
        parts.append("\n## Sub-collection summaries\n")
        for name, text in subfolder_summaries:
            parts.append(f"\n### Subfolder: {name}\n\n{text}\n")
    if file_summaries:
        parts.append("\n## Document summaries\n")
        for name, text in file_summaries:
            parts.append(f"\n### Document: {name}\n\n{text}\n")
    return "\n".join(parts)


def summarize_folder(folder: Path, corpus_root: Path, model: str, force: bool):
    """Post-order: summarize children first, then this folder. Returns True if a
    _FOLDER_SUMMARY.md was written (folder had content)."""
    subdirs = sorted(c for c in folder.iterdir() if c.is_dir())
    for d in subdirs:
        summarize_folder(d, corpus_root, model, force)

    out = folder / "_FOLDER_SUMMARY.md"
    if not force and out.exists() and out.stat().st_size > 20:
        return True

    file_summaries, subfolder_summaries = gather_folder_inputs(folder)
    if not file_summaries and not subfolder_summaries:
        return False  # nothing to summarize

    # Single-file, no-subfolder folder -> reuse the child summary (no LLM call).
    if len(file_summaries) == 1 and not subfolder_summaries:
        out.write_text(file_summaries[0][1], encoding="utf-8")
        return True

    user_msg = build_folder_user_msg(folder.name, file_summaries, subfolder_summaries)
    try:
        overview = call_text_llm(FOLDER_PROMPT, user_msg, model=model,
                                 max_tokens=FOLDER_MAX_TOKENS)
    except Exception as ex:
        overview = f"*(folder summary failed: {type(ex).__name__}: {ex})*"
    out.write_text(overview, encoding="utf-8")
    return True


def build_all_summaries(corpus_root: Path):
    """Mechanical concat of every *_summary.md, grouped by folder."""
    lines = ["# All Document Summaries\n",
             "Mechanical concatenation of every per-document summary. "
             "Generated by prepare-3.\n"]
    total = 0
    for p in sorted(corpus_root.rglob("*_summary.md")):
        rel = p.relative_to(corpus_root)
        doc_rel = str(rel).replace("_summary.md", "")
        lines.append(f"\n## {doc_rel}\n")
        lines.append(f"[open document]({doc_rel}.md)\n")
        lines.append(_read(p) + "\n")
        total += 1
    (corpus_root / "ALL_SUMMARIES.md").write_text("\n".join(lines), encoding="utf-8")
    return total


def build_crosscutting(corpus_root: Path, big_call_model: str):
    parts = ["# All document summaries\n"]
    for p in sorted(corpus_root.rglob("*_summary.md")):
        doc_rel = str(p.relative_to(corpus_root)).replace("_summary.md", "")
        parts.append(f"\n## Document: {doc_rel}\n\n{_read(p)}\n")
    user_msg = "\n".join(parts)
    try:
        overview = call_text_llm(CROSSCUTTING_PROMPT, user_msg,
                                 model=big_call_model, max_tokens=BIG_CALL_MAX_TOKENS)
    except Exception as ex:
        overview = f"*(crosscutting overview failed: {type(ex).__name__}: {ex})*"
    (corpus_root / "CORPUS_CROSSCUTTING.md").write_text(overview, encoding="utf-8")


def run_prepare_3(corpus_root: str, model: str, big_call_model: str,
                  workers: int, crosscutting: bool, force: bool,
                  only: str | None = None) -> None:
    root = Path(corpus_root)

    if only in (None, "tree"):
        print("prepare-3: building recursive folder summaries...")
        for d in sorted(c for c in root.iterdir() if c.is_dir()):
            summarize_folder(d, root, model, force)
        # Summarize the root folder from top-level file + subfolder summaries.
        summarize_folder(root, root, model, force)
        root_summary = root / "_FOLDER_SUMMARY.md"
        if root_summary.exists():
            shutil.copyfile(root_summary, root / "CORPUS_OVERVIEW.md")

    if only in (None, "concat"):
        n = build_all_summaries(root)
        print(f"prepare-3: wrote ALL_SUMMARIES.md ({n} summaries)")

    if crosscutting and only in (None, "crosscutting"):
        print("prepare-3: building crosscutting overview (one big call)...")
        build_crosscutting(root, big_call_model)
        print("prepare-3: wrote CORPUS_CROSSCUTTING.md")
