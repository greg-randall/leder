"""Stage D source-document resolution, rendering, and excerpt highlighting.

Builds the `sources/` folder alongside stage-d's article.html: one rendered,
excerpt-highlighted HTML page per document actually cited by this article's
findings, plus the original pre-conversion file for download.
"""
from __future__ import annotations

import html as html_mod


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
