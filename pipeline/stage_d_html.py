"""Stage D: Convert sourced Markdown article to a clean, self-contained HTML page.

Footnotes become clickable superscript links. Source references are styled
as cards at the bottom. The page is fully self-contained — no external CSS.
"""
from __future__ import annotations

import re
import html as html_mod


_STYLE = """<style>
  :root {
    --text: #1a1a1a; --bg: #fefefe; --muted: #666;
    --supported: #2d7d46; --contradicted: #c42b2b; --unsupported: #b08800;
    --source-bg: #f5f5f5; --border: #e0e0e0;
    font-family: Georgia, 'Times New Roman', serif;
    line-height: 1.7; color: var(--text); background: var(--bg);
    max-width: 720px; margin: 0 auto; padding: 2rem 1.5rem;
  }
  h1, h2, h3 { font-family: -apple-system, BlinkMacSystemFont, sans-serif; line-height: 1.3; }
  p { margin: 0 0 1.2em; }
  sup { line-height: 0; }
  a { color: #2563eb; text-decoration: none; }
  a:hover { text-decoration: underline; }
  a.fn { scroll-margin-top: 2em; }
  a.fn-ref { font-size: 0.8em; vertical-align: super; text-decoration: none; }
  a.fn-ref:hover { text-decoration: underline; }
  a.fn-backref { font-size: 0.85em; margin-left: 0.3em; text-decoration: none; }
  hr { border: none; border-top: 1px solid var(--border); margin: 3em 0 1.5em; }
  .sources h2 { margin-bottom: 0.5em; }
  .source { background: var(--source-bg); border: 1px solid var(--border);
    border-radius: 6px; padding: 0.9em 1em; margin: 0.8em 0; }
  .source .badge { display: inline-block; padding: 0.1em 0.5em; border-radius: 4px;
    font-family: -apple-system, sans-serif; font-size: 0.82em; font-weight: 600; margin-right: 0.5em; }
  .badge.supported { background: #d4edda; color: var(--supported); }
  .badge.contradicted { background: #fde8e8; color: var(--contradicted); }
  .badge.unsupported { background: #fff8e1; color: var(--unsupported); }
  .source .claim { font-weight: 600; }
  .source .rationale { display: block; margin: 0.3em 0; color: var(--muted); font-size: 0.94em; }
  .source .src-link { font-size: 0.85em; font-family: monospace; word-break: break-all; }
  .source .flags { font-size: 0.82em; margin-top: 0.3em; }
  .flag-review { color: var(--contradicted); }
  .flag-reconciled { color: var(--unsupported); }
  .unplaced { background: #fff0f0; border: 2px solid var(--contradicted);
    border-radius: 8px; padding: 1em 1.3em; margin-bottom: 2em; }
  .unplaced h2 { margin-top: 0; color: var(--contradicted); }
  .unplaced li { margin-bottom: 0.8em; }
  @media (prefers-color-scheme: dark) {
    :root { --text: #ddd; --bg: #1a1a1a; --muted: #999;
      --supported: #4ade80; --contradicted: #f87171; --unsupported: #facc15;
      --source-bg: #252525; --border: #333; }
  }
</style>"""


def _escape(text: str) -> str:
    return html_mod.escape(text, quote=False)


def convert(article_sourced_md: str, output_html: str) -> None:
    """Convert article-sourced.md to a self-contained HTML page."""
    with open(article_sourced_md, encoding="utf-8") as f:
        md = f.read()

    # Split into article body and footnotes
    parts = md.split("\n---\n\n## Sources\n", 1)
    body = parts[0] if parts else md
    footnotes_raw = parts[1] if len(parts) > 1 else ""

    # Remove unplaced claims block (it appears at top of md)
    unplaced_match = re.match(r'(# ⚠️ UNPLACED CLAIMS.*?\n)(?=\n)', body, re.DOTALL)
    unplaced_html = ""
    if unplaced_match:
        unplaced_html = _render_unplaced(unplaced_match.group(1))
        body = body[unplaced_match.end():]

    # Convert body: markdown to HTML (lightweight conversion)
    html_body = _md_to_html(body)

    # Process footnotes
    sources_html = _render_sources(footnotes_raw)

    # Assemble
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sourced Article</title>
{_STYLE}
</head>
<body>
{unplaced_html}
{html_body}
<hr>
<div class="sources">
{sources_html}
</div>
</body>
</html>"""

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"HTML written → {output_html}")


def _md_to_html(text: str) -> str:
    """Convert basic Markdown to HTML."""
    # Headings
    text = re.sub(r'^#### (.+)$', r'<h4>\1</h4>', text, flags=re.MULTILINE)
    text = re.sub(r'^### (.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.+)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.+)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)

    # Bold and italic
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'<strong><em>\1</em></strong>', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)

    # Footnote references: [^N] in body
    text = re.sub(
        r'\[\^(\d+)\]',
        r'<a href="#fn\1" id="fnref\1" class="fn-ref"><sup>\1</sup></a>',
        text,
    )

    # Paragraphs: blank-line separated blocks
    paragraphs = text.split('\n\n')
    result = []
    for block in paragraphs:
        block = block.strip()
        if not block:
            continue
        if re.match(r'^<(h[1-4]|ul|ol|li|blockquote)', block):
            result.append(block)
        else:
            # Inline newlines become <br> within paragraphs
            block = block.replace('\n', '<br>\n')
            result.append(f'<p>{block}</p>')

    return '\n'.join(result)


def _render_sources(raw: str) -> str:
    """Render the footnotes block as styled source cards."""
    if not raw.strip():
        return ""

    lines = raw.strip().splitlines()
    sources = []
    current = None

    for line in lines:
        m = re.match(r'\[(\^?\d+)\]:\s+\*\*\[(.+?)\]\*\*\s+\[(\w+)\]\s+(.+?)\s+—\s+(.+?)(\s+⚠️.*)?$', line)
        if m:
            fn_id, verdict_raw, proximity, claim, rationale, flags = m.groups()
            # Second line has Source: ...
            current = {
                "id": fn_id.lstrip("^"),
                "verdict": verdict_raw.strip(),
                "proximity": proximity.strip(),
                "claim": claim.strip(),
                "rationale": rationale.strip(),
                "source": "",
                "flags": flags.strip() if flags else "",
            }
            sources.append(current)
        elif line.strip().startswith("Source:") and current:
            current["source"] = line.strip()[7:].strip()

    if not sources:
        return ""

    html = "<h2>Sources</h2>\n"
    for s in sources:
        v = s["verdict"]
        if "Supported" in v:
            vclass = "supported"
        elif "Contradicted" in v:
            vclass = "contradicted"
        else:
            vclass = "unsupported"

        flags_html = ""
        if s["flags"]:
            for flag in s["flags"].split("⚠️"):
                flag = flag.strip()
                if "HUMAN REVIEW" in flag:
                    flags_html += '<span class="flag-review">⚠️ Human review</span> '
                if "RECONCILED" in flag:
                    flags_html += '<span class="flag-reconciled">🔧 Reconciled</span> '

        src_html = ""
        if s["source"] and s["source"] != "none":
            src_html = f'<span class="src-link">{_escape(s["source"])}</span>'

        html += (
            f'<div class="source" id="fn{s["id"]}">'
            f'<span class="badge {vclass}">{_escape(v)}</span>'
            f'<span class="claim">{_escape(s["claim"])}</span>'
            f'<span class="rationale">{_escape(s["rationale"])}</span>'
            f'{src_html}'
            f'<span class="flags">{flags_html}</span>'
            f'<a href="#fnref{s["id"]}" class="fn-backref">↩</a>'
            f'</div>\n'
        )

    return html


def _render_unplaced(text: str) -> str:
    """Render the unplaced claims warning block."""
    body = text.replace("# ⚠️ UNPLACED CLAIMS", "").strip()
    escaped = _escape(body).replace('\n', '<br>\n')
    return f'<div class="unplaced"><h2>⚠️ Unplaced Claims</h2><p>{escaped}</p></div>\n'


def run_stage_d(article_sourced_md: str, output_html: str = "") -> str:
    """Convert article-sourced.md to article-sourced.html."""
    if not output_html:
        output_html = article_sourced_md.replace(".md", ".html")
    convert(article_sourced_md, output_html)
    return output_html
