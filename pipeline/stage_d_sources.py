"""Stage D source-document resolution, rendering, and excerpt highlighting.

Builds the `sources/` folder alongside stage-d's article.html: one rendered,
excerpt-highlighted HTML page per document actually cited by this article's
findings, plus the original pre-conversion file for download.
"""
from __future__ import annotations

import html as html_mod
import os
import shutil
import subprocess
import sys
from pathlib import Path as _Path

from pipeline.stage_c_rebuild import find_quote_position

_FAILED_FETCH_PREFIX = "(failed to fetch "
_FETCH_PAGE_SCRIPT = str(_Path(__file__).resolve().parent / "tools" / "fetch_page.py")


def mark_excerpts(text: str, spans: list[tuple[int, int, str, str]]) -> str:
    """Insert HTML-escaped <mark> tags at the given (start, end, finding_id,
    severity) offsets within `text`. Offsets refer to positions in the raw
    (unescaped) `text` -- escaping happens per-segment during this function,
    never before, so offsets are never invalidated by escaping.

    Overlapping spans are split into non-overlapping segments at their
    boundary points, each segment's <mark> carrying the union of finding
    IDs that cover it (data-findings="a,b").
    """
    if not spans:
        return html_mod.escape(text, quote=False)

    # Collect every boundary point (start and end of every span), then walk
    # the text between consecutive boundaries -- each resulting segment is
    # covered by a fixed, unchanging set of spans, so it can be marked once.
    boundaries = sorted({0, len(text), *[s for s, e, _, _ in spans], *[e for s, e, _, _ in spans]})

    parts: list[str] = []
    for seg_start, seg_end in zip(boundaries, boundaries[1:]):
        if seg_start >= seg_end:
            continue
        segment_text = text[seg_start:seg_end]
        covering = [
            (fid, sev) for (s, e, fid, sev) in spans
            if s <= seg_start and seg_end <= e
        ]
        escaped = html_mod.escape(segment_text, quote=False)
        if not covering:
            parts.append(escaped)
            continue
        finding_ids = ",".join(fid for fid, _ in covering)
        # Single covering finding keeps its own anchor id for deep-linking /
        # scroll-to-on-open; a merged (overlap) segment uses the first
        # covering finding's id as the anchor -- any of them is a valid
        # scroll target for that segment.
        anchor_id = f"exc-{covering[0][0]}"
        severity = covering[0][1]
        parts.append(
            f'<mark id="{anchor_id}" data-findings="{finding_ids}" '
            f'data-severity="{severity}">{escaped}</mark>'
        )
    return "".join(parts)


def resolve_cited_sources(
    findings: list[dict], corpus_root: str, web_cache_dir: str,
) -> dict[str, dict]:
    """Map cited findings to their resolvable source documents.

    Returns {key: {"kind": "corpus"|"web", "local_path": str,
                   "original_path": str | None, "excerpts": [(excerpt, finding_id, severity)]}}
    keyed by source_path (corpus documents) or finding_id (web-sourced findings).
    Findings whose source can't be resolved (missing file, failed-fetch
    placeholder, or neither source_path nor source_url set) are skipped.
    """
    resolved: dict[str, dict] = {}

    for f in findings:
        source_path = f.get("source_path")
        source_url = f.get("source_url")
        finding_id = f["finding_id"]
        severity = f["severity"]
        excerpt = f.get("source_excerpt") or ""

        if source_path:
            key = source_path
            local_path = os.path.join(corpus_root, source_path)
            if not os.path.exists(local_path):
                continue
            kind = "corpus"
        elif source_url:
            key = finding_id
            local_path = os.path.join(web_cache_dir, finding_id, "page.md")
            if not os.path.exists(local_path):
                continue
            with open(local_path, encoding="utf-8") as fh:
                content = fh.read()
            if content.startswith(_FAILED_FETCH_PREFIX):
                continue
            kind = "web"
        else:
            continue

        if key not in resolved:
            resolved[key] = {
                "kind": kind, "local_path": local_path,
                "original_path": None, "excerpts": [],
            }
        resolved[key]["excerpts"].append((excerpt, finding_id, severity))

    return resolved


def render_source_document(
    text: str, excerpts: list[tuple[str, str, str]],
) -> tuple[str, list[str]]:
    """Render a source document's raw text to paragraph-wrapped HTML with
    its cited excerpts highlighted. Returns (html, not_found_finding_ids)
    for any excerpt that couldn't be located in the text -- the document
    still renders in full either way.
    """
    spans: list[tuple[int, int, str, str]] = []
    not_found: list[str] = []

    for excerpt, finding_id, severity in excerpts:
        pos = find_quote_position(excerpt, text)
        if pos is None:
            not_found.append(finding_id)
            continue
        start, end = pos
        spans.append((start, end, finding_id, severity))

    marked = mark_excerpts(text, spans)

    paragraphs = marked.split("\n\n")
    blocks = []
    for block in paragraphs:
        block = block.strip()
        if not block:
            continue
        blocks.append(f"<p>{block.replace(chr(10), '<br>')}</p>")
    html = "\n".join(blocks)

    return html, not_found


def _run_fetch_page(url: str, target_id: str, cache_dir: str) -> None:
    """Invoke fetch_page.py as a subprocess, same pattern already used for
    agent-initiated fetches elsewhere in this codebase (absolute script path
    + sys.executable, since a relative path doesn't resolve from an
    arbitrary cwd)."""
    subprocess.run(
        [sys.executable, _FETCH_PAGE_SCRIPT, url, target_id, "--cache-dir", cache_dir],
        capture_output=True, timeout=120,
    )


def backstop_fetch_missing(
    findings: list[dict], corpus_root: str, web_cache_dir: str,
) -> list[tuple[str, str]]:
    """Attempt one fetch for every web-sourced finding that resolve_cited_sources
    couldn't resolve. Returns [(finding_id, source_url), ...] for whatever is
    still unresolved after the attempt.
    """
    still_missing: list[tuple[str, str]] = []
    seen_finding_ids: set[str] = set()

    for f in findings:
        finding_id = f["finding_id"]
        source_url = f.get("source_url")
        source_path = f.get("source_path")
        if source_path or not source_url or finding_id in seen_finding_ids:
            continue
        seen_finding_ids.add(finding_id)

        page_path = os.path.join(web_cache_dir, finding_id, "page.md")
        if os.path.exists(page_path):
            with open(page_path, encoding="utf-8") as fh:
                if not fh.read().startswith(_FAILED_FETCH_PREFIX):
                    continue  # already resolved, nothing to backstop

        _run_fetch_page(source_url, finding_id, web_cache_dir)

        if not os.path.exists(page_path):
            still_missing.append((finding_id, source_url))
            continue
        with open(page_path, encoding="utf-8") as fh:
            if fh.read().startswith(_FAILED_FETCH_PREFIX):
                still_missing.append((finding_id, source_url))

    return still_missing


def write_missing_snapshots_report(output_dir: str, still_missing: list[tuple[str, str]]) -> None:
    """Write MISSING_SNAPSHOTS.md listing web citations that have no archived
    snapshot even after the backstop-fetch attempt. Removes any stale report
    when there's nothing to report, matching prepare_1_convert.py's
    _write_unconverted convention."""
    path = os.path.join(output_dir, "MISSING_SNAPSHOTS.md")
    if not still_missing:
        if os.path.exists(path):
            os.remove(path)
        return
    lines = [
        "# MISSING SNAPSHOTS\n",
        f"**{len(still_missing)} web citation(s) have no archived snapshot** "
        "and could not be fetched. These findings show as text-only in the "
        "article -- no \"Explore the source material\" button.\n\n",
        "| Finding | URL |\n| --- | --- |\n",
    ]
    for finding_id, url in sorted(still_missing):
        lines.append(f"| {finding_id} | {url} |\n")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("".join(lines))


def build_sources_folder(
    findings: list[dict], corpus_root: str, web_cache_dir: str, output_dir: str,
) -> dict[str, str]:
    """Build {output_dir}/sources/ and return {resolve_key: output_relative_html_path}."""
    still_missing = backstop_fetch_missing(findings, corpus_root, web_cache_dir)
    write_missing_snapshots_report(output_dir, still_missing)

    resolved = resolve_cited_sources(findings, corpus_root, web_cache_dir)
    mapping: dict[str, str] = {}

    for key, entry in resolved.items():
        with open(entry["local_path"], encoding="utf-8") as fh:
            text = fh.read()
        html, _not_found = render_source_document(text, entry["excerpts"])

        if entry["kind"] == "corpus":
            rel_html = os.path.join("sources", os.path.splitext(key)[0] + ".html")
            out_html_path = os.path.join(output_dir, rel_html)
            os.makedirs(os.path.dirname(out_html_path), exist_ok=True)
            with open(out_html_path, "w", encoding="utf-8") as fh:
                fh.write(html)

            # Original pre-conversion file: strip the ".md" sidecar suffix.
            if key.endswith(".md"):
                original_source = os.path.join(corpus_root, key[:-len(".md")])
                if os.path.exists(original_source):
                    original_dest = os.path.join(
                        output_dir, "sources", os.path.dirname(key), os.path.basename(key)[:-len(".md")],
                    )
                    os.makedirs(os.path.dirname(original_dest), exist_ok=True)
                    shutil.copy2(original_source, original_dest)
        else:  # web
            rel_html = f"sources/web/{key}.html"
            out_html_path = os.path.join(output_dir, rel_html)
            os.makedirs(os.path.dirname(out_html_path), exist_ok=True)
            with open(out_html_path, "w", encoding="utf-8") as fh:
                fh.write(html)
            shutil.copy2(entry["local_path"], os.path.join(output_dir, "sources", "web", f"{key}.md"))

        mapping[key] = rel_html

    return mapping
