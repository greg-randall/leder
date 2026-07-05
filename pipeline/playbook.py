"""Playbook: a YAML-defined editorial check (extraction + verification + display)."""
from __future__ import annotations

from dataclasses import dataclass, field

import yaml


@dataclass
class Playbook:
    name: str
    description: str = ""
    extraction_prompt: str = ""
    quality_gate_enabled: bool = False
    quality_gate_prompt: str = ""
    verification_prompt: str = ""
    allowed_tools: list[str] = field(default_factory=lambda: ["Read"])
    display_template: str = ""


def load_playbook(path: str) -> Playbook:
    with open(path) as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raise ValueError(f"Playbook YAML is empty: {path}")

    name = raw.get("name", "")
    extraction = raw.get("extraction") or {}
    verification = raw.get("verification") or {}
    display = raw.get("display") or {}
    qg = extraction.get("quality_gate") or {}

    if not extraction.get("prompt"):
        raise ValueError(f"Playbook '{name}': extraction.prompt is required")
    if not verification.get("prompt"):
        raise ValueError(f"Playbook '{name}': verification.prompt is required")

    return Playbook(
        name=name,
        description=raw.get("description", ""),
        extraction_prompt=extraction["prompt"].strip(),
        quality_gate_enabled=qg.get("enabled", False),
        quality_gate_prompt=(qg.get("prompt") or "").strip(),
        verification_prompt=verification["prompt"].strip(),
        allowed_tools=verification.get("allowed_tools", ["Read"]),
        display_template=(display.get("template") or "").strip(),
    )
