"""Shared re-flow pass for pipeline-generated conversion output.

Applied to any .md file the pipeline itself writes (subtitle/whisper/OCR/
gap-filler output) -- NEVER to passthrough_text output, which must stay
byte-for-byte identical to its source. Lives in its own module because both
prepare_1_convert.py and prepare_audio.py need it, and prepare_1_convert.py
already imports from prepare_audio.py -- a shared leaf module avoids a
circular import.
"""
from __future__ import annotations

import re

_LONG_LINE_THRESHOLD = 300
_WRAP_WIDTH = 110  # midpoint of the ~100-120 char target range

_SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def _wrap_fixed_width(line: str, width: int = _WRAP_WIDTH) -> str:
    """Wrap a single line at ~width chars on word boundaries only.

    Never splits a word, never inserts/removes non-whitespace -- each
    replaced space becomes a newline (both collapse to ' ' under
    re.sub(r'\\s+', ' ', ...)), preserving the whitespace-only invariant.
    """
    words = line.split(' ')
    wrapped_lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}" if current else word
        if current and len(candidate) > width:
            wrapped_lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        wrapped_lines.append(current)
    return "\n".join(wrapped_lines)


def reflow_pipeline_text(text: str) -> str:
    """Re-flow any line longer than ~300 chars into readable paragraphs.

    Sentence-boundary wrap is tried first (splits after '.'/'?'/'!' followed
    by whitespace and a capital letter -- the same pattern stage-a's
    _chunk_article already uses). If a long line has no such boundary
    (unpunctuated caption text), falls back to fixed-width word-boundary
    wrapping. Lines at or under the threshold pass through untouched, so
    well-structured converter output (MarkItDown, etc.) is never touched.
    """
    lines = text.split('\n')
    result_lines: list[str] = []
    for line in lines:
        if len(line) <= _LONG_LINE_THRESHOLD:
            result_lines.append(line)
            continue
        sentences = _SENTENCE_BOUNDARY.split(line)
        if len(sentences) > 1:
            result_lines.append('\n'.join(sentences))
        else:
            result_lines.append(_wrap_fixed_width(line))
    return '\n'.join(result_lines)
