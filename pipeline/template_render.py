"""Display template renderer for playbook finding cards.

Syntax:
  {{field}}         -> str(getattr(finding, field, ""))
  {{metadata.key}}  -> str(finding.metadata.get("key", ""))
  {{severity_badge}}-> "PASS" / "WARNING" / "CRITICAL" (rendered as text;
                       the HTML renderer wraps these in CSS classes)
  {% if field %}...body...{% endif %} -> body included iff the resolved value
                       is truthy (non-empty, non-None, non-zero).
"""
from __future__ import annotations

import re
from pipeline.finding import Finding

_IF_RE = re.compile(r"\{\%\s*if\s+(\w+(?:\.\w+)?)\s*\%\}(.*?)\{\%\s*endif\s*\%\}", re.DOTALL)
_VAR_RE = re.compile(r"\{\{(\w+(?:\.\w+)?)\}\}")


def _resolve(finding: Finding, name: str) -> str:
    if name == "severity_badge":
        return finding.severity.value
    if name.startswith("metadata."):
        val = finding.metadata.get(name[9:], "")
        return val if val is not None else ""
    val = getattr(finding, name, None)
    if val is None:
        return ""
    return str(val)


def render_template(template: str, finding: Finding) -> str:
    # 1. Expand {% if %} blocks first
    def _if_replacer(m):
        field, body = m.group(1), m.group(2)
        return body if _resolve(finding, field) else ""
    out = _IF_RE.sub(_if_replacer, template)
    # 2. Resolve {{ }} variables

    def _var_replacer(m):
        return _resolve(finding, m.group(1))
    return _VAR_RE.sub(_var_replacer, out)
