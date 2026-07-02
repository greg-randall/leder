"""Shared single-shot text completion over the Anthropic-compatible endpoint.

base_url / auth come from ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN, which
cli._setup_provider_env() sets for DeepSeek. Mirrors Stage A's client setup.
"""
from __future__ import annotations

import os

import anthropic


def call_text_llm(system: str, user: str, model: str, max_tokens: int = 4096) -> str:
    """Return the model's text output (all text blocks concatenated)."""
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("DEEPSEEK_API_KEY"),
    )
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0.0,
        system=system,
        messages=[{"role": "user", "content": user}],
        thinking={"type": "disabled"},
    )
    return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
