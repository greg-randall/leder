#!/usr/bin/env python3
"""OCR-quality gate + OpenAI vision escalation for image-based content.

DeepSeek is text-only, so when tesseract returns too little text we escalate
the page/image to OpenAI gpt-4o-mini. The prompt forbids fabrication -- this is
a fact-checking corpus, so invented text would corrupt verification.
"""
from __future__ import annotations

import base64
import io
import os
import re
from pathlib import Path

VISION_PROMPT = """You are transcribing one image from a fact-checking archive.
Accuracy is critical -- downstream users verify factual claims against your
output. NEVER invent, guess, or complete text.

Return BOTH sections:

## Transcription
Transcribe ALL visible text verbatim, preserving structure (headings, lists,
tables as markdown tables, form labels and their values). Keep exact numbers,
dates, names, units, and IDs as written. Anything unreadable -> [illegible]
(never guess). If there is no text -> "(no text)".

## Description
State the image type (letter, form, permit, map, chart, photo, aerial/satellite,
diagram).
- Photographs: describe the scene concretely -- setting/location, infrastructure
  present (tanks, pipes, ponds, wells, irrigation gear, vehicles), visible
  conditions (standing water, discolored/wet soil, dead or stressed vegetation,
  erosion, staining, spills), any people/equipment with visible markings or IDs,
  and the exact text of any signs, labels, or nameplates. Note any timestamp or
  location text burned into the image. Report only what is observably present --
  do NOT speculate about causes or what happened.
- Charts/maps/diagrams: title, axis labels, units, legend, plotted values/ranges,
  locations, and any figures."""

PROVENANCE_BANNER = ("> ⚠️ Text recovered via {model} vision "
                     "(OCR was insufficient) — verify against original.\n\n")

_WORD_RE = re.compile(r"[A-Za-z]{2,}")

# Media types OpenAI accepts directly; anything else is normalized to PNG.
_DIRECT_MEDIA = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                 ".gif": "image/gif", ".webp": "image/webp"}


def count_real_words(text: str) -> int:
    """Count word-like tokens (>=2 letters) as a proxy for meaningful text."""
    return len(_WORD_RE.findall(text or ""))


def needs_vision(ocr_text: str, min_words: int) -> bool:
    return count_real_words(ocr_text) < min_words


def _encode_image(img_path: Path):
    """Return (media_type, base64_str). Normalizes non-direct formats to PNG."""
    img_path = Path(img_path)
    media = _DIRECT_MEDIA.get(img_path.suffix.lower())
    if media is not None:
        data = img_path.read_bytes()
    else:
        from PIL import Image
        buf = io.BytesIO()
        Image.open(img_path).convert("RGB").save(buf, format="PNG")
        data = buf.getvalue()
        media = "image/png"
    return media, base64.b64encode(data).decode("ascii")


def _openai_client():
    from openai import OpenAI
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def vision_extract(img_path: Path, model: str) -> str:
    """Send one image to the vision model. Returns markdown (no banner)."""
    media, b64 = _encode_image(img_path)
    client = _openai_client()
    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": VISION_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:{media};base64,{b64}"}},
            ],
        }],
    )
    return resp.choices[0].message.content or ""
