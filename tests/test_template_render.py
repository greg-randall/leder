"""Tests for the display template renderer."""
from pipeline.template_render import render_template
from pipeline.finding import Finding, Severity


FAKE = Finding(
    finding_id="fc-001", check_type="fact_check", severity=Severity.PASS,
    target_text="LA-0304 irrigates 165 acres.",
    anchor_text="LA-0304 irrigates 165 acres in Karnes County.",
    agent_summary="The permit confirms.",
    source_excerpt='"shall irrigate 165 acres"',
    source_path="LA-0304/doc.md",
    metadata={"claim_type": "numeric"},
)


def test_simple_field_substitution():
    out = render_template("{{severity_badge}} **{{target_text}}**", FAKE)
    assert "PASS" in out
    assert "LA-0304 irrigates 165 acres." in out


def test_metadata_access():
    out = render_template("Type: {{metadata.claim_type}}", FAKE)
    assert "numeric" in out


def test_if_block_included():
    tmpl = "{% if source_excerpt %}> {{source_excerpt}}{% endif %}"
    out = render_template(tmpl, FAKE)
    assert '"shall irrigate 165 acres"' in out


def test_if_block_excluded():
    f = Finding(finding_id="x", check_type="x", severity=Severity.WARNING,
                target_text="x", anchor_text="x", agent_summary="ok")
    tmpl = "{% if source_excerpt %}> {{source_excerpt}}{% endif %}"
    out = render_template(tmpl, f)
    assert out.strip() == ""


def test_missing_field_is_empty_string():
    out = render_template("{{recommended_action}}", FAKE)
    assert out.strip() == ""


def test_full_display_template():
    tmpl = (
        "{{severity_badge}} **{{target_text}}**\n"
        "{{agent_summary}}\n"
        "{% if source_excerpt %}> \"{{source_excerpt}}\"{% endif %}\n"
        "{% if source_path %}[source]({{source_path}}){% endif %}"
    )
    out = render_template(tmpl, FAKE)
    assert "PASS" in out
    assert "LA-0304/doc.md" in out
