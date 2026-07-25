#!/usr/bin/env python3
"""prepare-3: recursive folder-summary rollup + crosscutting corpus overview.

Post-order over the corpus tree: every folder gets a _FOLDER_SUMMARY.md
synthesized from its direct *_summary.md files plus its immediate subfolders'
_FOLDER_SUMMARY.md. Root summary is aliased to CORPUS_OVERVIEW.md. A single
giant call over all *_summary.md (default on) writes CORPUS_CROSSCUTTING.md.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from pipeline.llm import call_text_llm

BIG_CALL_MAX_TOKENS = 384000
FOLDER_MAX_TOKENS = 12000


def build_folder_prompt(corpus_description: str) -> str:
    domain_block = ""
    if corpus_description:
        domain_block = f"This collection is:\n{corpus_description}\n\n"

    return f"""You are writing an overview of ONE folder in a document
collection. {domain_block}The folder may contain individual document summaries
and/or summaries of sub-collections (subfolders). Synthesize them into a
single overview.

This overview will be read by a downstream fact-checking agent deciding which
original document to open next. It is a routing table, not just a summary —
every Timeline and Key-contents-&-findings bullet must end with the source
document name(s) it came from, in brackets, so the agent can jump straight to
the right file. For example: "The council approved the water rate increase on
a 4-1 vote. [20260717 - Meeting for 7-14-26 [XQ3PK-qjSdk].en.srt]"

Write these sections:

**Overview:** 3-5 sentences describing what this folder contains as a whole.

**Timeline (if applicable):** A chronological bullet list of the key events you
can reconstruct from the summaries — meetings held, votes taken, contracts
awarded, incidents reported, applications filed, decisions made — whatever
event types the summaries actually contain. Put the date (or best-known date)
at the start of each bullet, and the source document name(s) in brackets at
the end. Only include events actually supported by the summaries. If dates
are sparse or the folder isn't chronological (e.g., reference materials, form
letters), skip this section.

**Key contents & findings:** Bullet list of the substantive matters across the
documents and sub-collections — who/what/where, notable numbers, names,
decisions, deficiencies, or issues. Cite specifics where the inputs give them,
and end each bullet with the source document name(s) in brackets.

**Open questions / gaps:** Anything the inputs suggest is unresolved, missing,
or unclear.

Stay grounded in what the summaries actually say. If the folder is thin or
mostly administrative, say so rather than padding."""


def build_crosscutting_prompt(corpus_description: str) -> str:
    domain_block = ""
    if corpus_description:
        domain_block = f"This collection is:\n{corpus_description}\n\n"

    return f"""You are writing the definitive analytical overview for
an ENTIRE document collection. {domain_block}Below are the per-document
summaries for every document across every folder. You have the complete
picture — draw connections across folders that no single-folder view could
reveal.

Write these sections:

**What this collection is:** A few paragraphs describing the dataset as a whole —
scale, the range of subjects/sources, the time span, and the common structure.

**Cross-cutting patterns:** Recurring themes, issues, people, or organizations
that appear across multiple folders. Cite specific folders and numbers.

**Notable / outlier items:** Documents or folders that stand out and why.

**Data highlights:** Concrete quantitative findings worth surfacing.

**Entity index:** Every person, organization, facility, and place that appears
in multiple documents. For each, list the variant spellings/renderings seen
and which documents it appears in, so a fact-checking agent can resolve a
name to every place it's discussed.

**Topic → document map:** For each major recurring topic, the list of
documents that discuss it, so a fact-checking agent can jump straight to the
relevant source material for a given subject.

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
        if p.name.startswith("_"):
            continue
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


def summarize_folder(folder: Path, corpus_root: Path, model: str, force: bool,
                     failures: list, pbar=None, corpus_description: str = ""):
    """Post-order: summarize children first, then this folder. Returns True if a
    _FOLDER_SUMMARY.md was written (folder had content)."""
    subdirs = sorted(c for c in folder.iterdir() if c.is_dir())
    if len(subdirs) > 1:
        from concurrent.futures import ThreadPoolExecutor as _TPE, as_completed as _ac
        with _TPE(max_workers=len(subdirs)) as pool:
            futures = {
                pool.submit(summarize_folder, d, corpus_root, model, force, failures, pbar,
                            corpus_description): d
                for d in subdirs
            }
            for future in _ac(futures):
                future.result()
    else:
        for d in subdirs:
            summarize_folder(d, corpus_root, model, force, failures, pbar, corpus_description)

    out = folder / "_FOLDER_SUMMARY.md"
    if not force and out.exists() and out.stat().st_size > 20:
        if pbar is not None:
            pbar.update(1)
        return True

    file_summaries, subfolder_summaries = gather_folder_inputs(folder)
    if not file_summaries and not subfolder_summaries:
        return False  # nothing to summarize

    # Single-file, no-subfolder folder -> reuse the child summary (no LLM call).
    if len(file_summaries) == 1 and not subfolder_summaries:
        out.write_text(file_summaries[0][1], encoding="utf-8")
        if pbar is not None:
            pbar.update(1)
        return True

    if pbar is not None:
        pbar.update(1)
    user_msg = build_folder_user_msg(folder.name, file_summaries, subfolder_summaries)
    try:
        overview = call_text_llm(build_folder_prompt(corpus_description), user_msg, model=model,
                                 max_tokens=FOLDER_MAX_TOKENS)
    except Exception as ex:
        rel = str(folder.relative_to(corpus_root)) or "."
        print(f"  prepare-3 folder summary failed: {rel}: {type(ex).__name__}: {ex}",
              file=sys.stderr)
        failures.append((rel, f"{type(ex).__name__}: {ex}"))
        return False  # don't persist a poison summary; re-run will retry
    out.write_text(overview, encoding="utf-8")
    return True


def build_all_summaries(corpus_root: Path):
    """Mechanical concat of every *_summary.md, grouped by folder."""
    lines = ["# All Document Summaries\n",
             "Mechanical concatenation of every per-document summary. "
             "Generated by prepare-3.\n"]
    total = 0
    for p in sorted(corpus_root.rglob("*_summary.md")):
        if p.name.startswith("_"):
            continue
        rel = p.relative_to(corpus_root)
        doc_rel = str(rel).replace("_summary.md", "")
        lines.append(f"\n## {doc_rel}\n")
        lines.append(f"[open document]({doc_rel}.md)\n")
        lines.append(_read(p) + "\n")
        total += 1
    (corpus_root / "ALL_SUMMARIES.md").write_text("\n".join(lines), encoding="utf-8")
    return total


def build_crosscutting(corpus_root: Path, big_call_model: str, corpus_description: str = ""):
    parts = ["# All document summaries\n"]
    for p in sorted(corpus_root.rglob("*_summary.md")):
        if p.name.startswith("_"):
            continue
        doc_rel = str(p.relative_to(corpus_root)).replace("_summary.md", "")
        parts.append(f"\n## Document: {doc_rel}\n\n{_read(p)}\n")
    user_msg = "\n".join(parts)
    try:
        overview = call_text_llm(build_crosscutting_prompt(corpus_description), user_msg,
                                 model=big_call_model, max_tokens=BIG_CALL_MAX_TOKENS,
                                 stream=True)
    except Exception as ex:
        print(f"  prepare-3 crosscutting failed: {type(ex).__name__}: {ex}",
              file=sys.stderr)
        overview = f"*(crosscutting overview failed: {type(ex).__name__}: {ex})*"
    (corpus_root / "CORPUS_CROSSCUTTING.md").write_text(overview, encoding="utf-8")


def run_prepare_3(corpus_root: str, model: str, big_call_model: str,
                  workers: int, crosscutting: bool, force: bool,
                  only: str | None = None, corpus_description: str = "") -> None:
    root = Path(corpus_root)

    if only in (None, "tree"):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm
        failures: list = []
        print("prepare-3: counting folders...", file=sys.stderr, end=" ", flush=True)
        total = sum(1 for _ in root.rglob("*") if _.is_dir()) + 1
        print(f"{total} found")
        pbar = tqdm(total=total, desc="prepare-3 folders", unit="folder")
        # Parallelize sibling folders at each level. Top-level dirs are siblings.
        subdirs = sorted(c for c in root.iterdir() if c.is_dir())
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {
                pool.submit(summarize_folder, d, root, model, force, failures, pbar,
                            corpus_description): d
                for d in subdirs
            }
            for future in as_completed(futures):
                future.result()
        # Root itself (depends on all subdirs done).
        summarize_folder(root, root, model, force, failures, pbar, corpus_description)
        pbar.close()
        root_summary = root / "_FOLDER_SUMMARY.md"
        if root_summary.exists():
            shutil.copyfile(root_summary, root / "CORPUS_OVERVIEW.md")
        if failures:
            print(f"prepare-3: ⚠ {len(failures)} folder summary/summaries failed "
                  f"(left unwritten; re-run to retry): "
                  f"{', '.join(f[0] for f in failures)}", file=sys.stderr)

    if only in (None, "concat"):
        n = build_all_summaries(root)
        print(f"prepare-3: wrote ALL_SUMMARIES.md ({n} summaries)")

    if only == "crosscutting" or (crosscutting and only is None):
        print("prepare-3: building crosscutting overview (one big call)...")
        build_crosscutting(root, big_call_model, corpus_description)
        print("prepare-3: wrote CORPUS_CROSSCUTTING.md")
