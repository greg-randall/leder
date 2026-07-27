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
    character offsets. Chunks longer than max_chunk are kept as-is rather
    than sub-split (find_quote_position handles them fine, and any
    sentence-sized passage is still much smaller than the full document).
    Avoiding sub-splitting eliminates synthetic-string offset math errors."""
    chunks: list[tuple[str, int, int]] = []
    for m in re.finditer(r'[^.!?\n]+[.!?\n]?', text):
        chunks.append((m.group(), m.start(), m.end()))
    return chunks


def _score_chunks(candidate_clean: str, chunks: list[tuple[str, int, int]]) -> list[tuple[int, float]]:
    """Score each chunk by word-overlap with the candidate, normalised to
    [0.0, 1.0] as the fraction of needle words present in the chunk.
    Returns [(idx, score), ...]."""
    needle_words = set(candidate_clean.split())
    if not needle_words:
        return [(i, 0.0) for i in range(len(chunks))]
    scores = []
    for i, (chunk_text, _, _) in enumerate(chunks):
        chunk_words = set(_clean(chunk_text).split())
        scores.append((i, len(needle_words & chunk_words) / len(needle_words)))
    return scores


def validate_excerpt(file_path: str, candidate: str) -> dict:
    """Check whether `candidate` appears in `file_path`, exactly or fuzzily.

    Returns {"found": True, "actual_text", "offset": [start, end], "similarity"}
    or {"found": False} / {"found": False, "error": str}. Never exits; never
    raises for a missing file. `actual_text` is always a real substring of the
    file, which is the whole point -- callers use it in place of whatever
    wording they came in with.
    """
    if not candidate.strip():
        return {"found": False, "error": "candidate_text is empty"}

    try:
        text = _Path(file_path).read_text(encoding="utf-8", errors="replace")
    except (FileNotFoundError, OSError):
        return {"found": False, "error": f"file not found: {file_path}"}

    # Tier 1: exact substring (case-insensitive)
    idx = text.lower().find(candidate.lower())
    if idx >= 0:
        return {
            "found": True,
            "actual_text": text[idx:idx + len(candidate)],
            "offset": [idx, idx + len(candidate)],
            "similarity": 1.0,
        }

    # Tier 2: chunk-scored Levenshtein on top 3 chunks
    chunks = _chunk_document(text)
    scores = _score_chunks(_clean(candidate), chunks)
    scores.sort(key=lambda x: x[1], reverse=True)
    top_indices = [i for i, _ in scores[:3]]

    best: dict | None = None
    for idx_chunk in top_indices:
        chunk_text, chunk_start, _ = chunks[idx_chunk]
        pos = find_quote_position(candidate, chunk_text)
        if pos is None:
            continue
        cstart, cend = pos
        actual = chunk_text[cstart:cend]
        ratio = difflib.SequenceMatcher(None, _clean(candidate), _clean(actual)).ratio()
        if ratio >= 0.6 and (best is None or ratio > best["similarity"]):
            best = {
                "found": True,
                "actual_text": actual,
                "offset": [chunk_start + cstart, chunk_start + cend],
                "similarity": round(ratio, 4),
            }

    # Tier 3: no match
    return best if best is not None else {"found": False}


def main() -> None:
    if len(sys.argv) != 3:
        print(json.dumps({
            "found": False,
            "error": "usage: validate_excerpt.py <file_path> <candidate_text>",
        }))
        sys.exit(1)
    result = validate_excerpt(sys.argv[1], sys.argv[2])
    print(json.dumps(result))
    sys.exit(0 if result.get("found") else 1)


if __name__ == "__main__":
    main()
