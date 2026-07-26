"""Stage D source-document resolution, rendering, and excerpt highlighting.

Builds the `sources/` folder alongside stage-d's article.html: one rendered,
excerpt-highlighted HTML page per document actually cited by this article's
findings, plus the original pre-conversion file for download.
"""
from __future__ import annotations

import html as html_mod
import os

_FAILED_FETCH_PREFIX = "(failed to fetch "


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
