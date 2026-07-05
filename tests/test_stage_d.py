"""Tests for Stage D HTML output with severity colors."""
from pathlib import Path


def test_severity_colors_in_html(tmp_path):
    """Generated HTML must use severity classes (pass/warning/critical), not verdict classes."""
    from pipeline.stage_d_html import run_stage_d

    md = tmp_path / "sourced.md"
    md.write_text(
        "# Test Article\n\nSome text here.\n\n"
        "---\n\n## Source Card: fc-001\n\n"
        "* **Severity:** PASS\n"
        "* **Check:** fact_check\n"
        "* **Claim:** Test claim text.\n"
    )
    html_path = run_stage_d(str(md), str(tmp_path / "out.html"))

    html = Path(html_path).read_text()
    # Must use severity-based class names
    assert "pass" in html.lower()
    # Old verdict classes should not appear
    assert "--supported" not in html
    # CSS color variables
    assert "--pass" in html or "var(--pass)" in html


def test_critical_severity_in_html(tmp_path):
    """CRITICAL severity gets red styling."""
    from pipeline.stage_d_html import run_stage_d

    md = tmp_path / "sourced.md"
    md.write_text(
        "# Test Article\n\nSome text.\n\n"
        "---\n\n## Source Card: fc-002\n\n"
        "* **Severity:** CRITICAL\n"
        "* **Check:** fact_check\n"
        "* **Claim:** Bad claim.\n"
    )
    html = Path(run_stage_d(str(md), str(tmp_path / "out.html"))).read_text()
    assert "--critical" in html
