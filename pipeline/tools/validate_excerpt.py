#!/usr/bin/env python3
"""Validate that a candidate text string exists (or nearly exists) in a source file.

Invoked by verification agents via Bash during stage B:
    python3 pipeline/tools/validate_excerpt.py <file_path> "<candidate_text>"

Outputs JSON to stdout. Exit 0 on match, 1 on no match / file not found.
"""
from __future__ import annotations

import difflib
import json
import re
import sys
from pathlib import Path as _Path

# Import find_quote_position from stage C -- same module stage D already imports.
# The script path is pipeline/tools/validate_excerpt.py; the project root is
# three levels up (tools/ -> pipeline/ -> leder/). Add it to sys.path so
# pipeline.stage_c_rebuild resolves when run from any working directory.
_PROJECT_ROOT = str(_Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pipeline.stage_c_rebuild import find_quote_position  # noqa: E402


def _clean(s: str) -> str:
    return re.sub(r'[^a-zA-Z0-9 ]', '', s.lower())


def _chunk_document(text: str, max_chunk: int = 800) -> list[tuple[str, int, int]]:
    """Split text into sentence-boundary chunks, each with its (start, end)
    character offsets. Chunks longer than max_chunk are split further at
    word boundaries."""
    chunks: list[tuple[str, int, int]] = []
    for m in re.finditer(r'[^.!?\n]+[.!?\n]?', text):
        chunk = m.group()
        cs, ce = m.start(), m.end()
        if len(chunk) <= max_chunk:
            chunks.append((chunk, cs, ce))
        else:
            words = chunk.split()
            sub_start = cs
            for i in range(0, len(words), 60):
                sub = ' '.join(words[i:i + 60])
                sub_end = sub_start + len(sub)
                chunks.append((sub, sub_start, sub_end))
                sub_start = sub_end + 1  # +1 for the space between sub-chunks
    return chunks


def _score_chunks(candidate_clean: str, chunks: list[tuple[str, int, int]]) -> list[tuple[int, float]]:
    """Score each chunk by word-overlap with the candidate. Returns [(idx, score), ...]."""
    needle_words = set(candidate_clean.split())
    if not needle_words:
        return [(i, 0.0) for i in range(len(chunks))]
    scores = []
    for i, (chunk_text, _, _) in enumerate(chunks):
        chunk_words = set(_clean(chunk_text).split())
        scores.append((i, len(needle_words & chunk_words)))
    return scores


def main() -> None:
    if len(sys.argv) != 3:
        print(json.dumps({"found": False, "error": "usage: validate_excerpt.py <file_path> <candidate_text>"}))
        sys.exit(1)

    file_path = sys.argv[1]
    candidate = sys.argv[2]

    if not candidate.strip():
        print(json.dumps({"found": False, "error": "candidate_text is empty"}))
        sys.exit(1)

    try:
        text = _Path(file_path).read_text(encoding="utf-8", errors="replace")
    except (FileNotFoundError, OSError):
        print(json.dumps({"found": False, "error": f"file not found: {file_path}"}))
        sys.exit(1)

    # Tier 1: exact substring (case-insensitive)
    idx = text.lower().find(candidate.lower())
    if idx >= 0:
        actual = text[idx:idx + len(candidate)]
        print(json.dumps({
            "found": True, "actual_text": actual,
            "offset": [idx, idx + len(candidate)], "similarity": 1.0,
        }))
        sys.exit(0)

    # Tier 2: chunk-scored Levenshtein on top 3 chunks
    chunks = _chunk_document(text)
    scores = _score_chunks(_clean(candidate), chunks)
    scores.sort(key=lambda x: x[1], reverse=True)
    top_indices = [i for i, _ in scores[:3]]

    best: dict | None = None
    for idx_chunk in top_indices:
        chunk_text, chunk_start, _ = chunks[idx_chunk]
        pos = find_quote_position(candidate, chunk_text)
        if pos is not None:
            cstart, cend = pos
            actual = chunk_text[cstart:cend]
            ratio = difflib.SequenceMatcher(None, _clean(candidate), _clean(actual)).ratio()
            if ratio >= 0.6:
                result = {
                    "found": True,
                    "actual_text": actual,
                    "offset": [chunk_start + cstart, chunk_start + cend],
                    "similarity": round(ratio, 4),
                }
                if best is None or ratio > best["similarity"]:
                    best = result

    if best is not None:
        print(json.dumps(best))
        sys.exit(0)

    # Tier 3: no match
    print(json.dumps({"found": False}))
    sys.exit(1)


if __name__ == "__main__":
    main()
