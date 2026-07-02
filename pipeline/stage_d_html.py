"""Stage D: Convert sourced Markdown article to a clean, self-contained HTML page.

Features:
- Color-coded footnote markers (green=supported, red=contradicted, orange=unsupported)
- Hover highlights the sentence in the verdict color
- Spacing between adjacent footnote numbers
- Source cards at bottom with clickable URLs
"""
from __future__ import annotations

import re
import html as html_mod


_STYLE = """<style>
  :root {
    --text: #1a1a1a; --bg: #fefefe; --muted: #666;
    --supported: #2d7d46; --supported-bg: #d4edda;
    --contradicted: #c42b2b; --contradicted-bg: #fde8e8;
    --unsupported: #b08800; --unsupported-bg: #fff8e1;
    --source-bg: #f5f5f5; --border: #e0e0e0;
  }
  body { font-family: Georgia, 'Times New Roman', serif;
    line-height: 1.7; color: var(--text); background: var(--bg);
    margin: 0; padding: 0; }
  h1, h2, h3 { font-family: -apple-system, BlinkMacSystemFont, sans-serif; line-height: 1.3; }
  p { margin: 0 0 1.2em; }
  a { color: #2563eb; text-decoration: none; }
  a:hover { text-decoration: underline; }
  hr { display: none; }

  /* Footnote reference markers in body text — colored pill badges */
  a.fn-ref {
    font-size: 0.75em; vertical-align: super; text-decoration: none;
    font-weight: 700; line-height: 0; margin: 0 0.12em;
    padding: 0.1em 0.35em; border-radius: 10px;
    color: #fff;
  }
  a.fn-ref:hover { filter: brightness(1.2); transform: scale(1.15); }
  .fn-ref.supported { background: var(--supported); }
  .fn-ref.contradicted { background: var(--contradicted); }
  .fn-ref.unsupported { background: var(--unsupported); }

  /* Source cards at bottom */
  .sources { display: none; }  /* hidden — sidebar shows them all */
  .source { background: var(--source-bg); border: 1px solid var(--border);
    border-left: 3px solid var(--border); border-radius: 6px;
    padding: 0.9em 1em; margin: 0.8em 0; }
  .source.supported { border-left-color: var(--supported); }
  .source.contradicted { border-left-color: var(--contradicted); }
  .source.unsupported { border-left-color: var(--unsupported); }
  .source .badge { display: inline-block; padding: 0.1em 0.5em; border-radius: 4px;
    font-family: -apple-system, sans-serif; font-size: 0.82em; font-weight: 600; margin-right: 0.5em; }
  .badge.supported { background: var(--supported-bg); color: var(--supported); }
  .badge.contradicted { background: var(--contradicted-bg); color: var(--contradicted); }
  .badge.unsupported { background: var(--unsupported-bg); color: var(--unsupported); }
  .source .claim { font-weight: 600; }
  .source .rationale { display: block; margin: 0.3em 0; color: var(--muted); font-size: 0.94em; }
  .source .src-link { font-size: 0.85em; font-family: monospace; word-break: break-all; }
  .source .src-link a { color: var(--muted); }
  .source .flags { font-size: 0.82em; margin-top: 0.3em; }
  .flag-review { color: var(--contradicted); }
  .flag-reconciled { color: var(--unsupported); }

  /* Unplaced claims warning */
  .unplaced { background: #fff0f0; border: 2px solid var(--contradicted);
    borderRadius: 8px; padding: 1em 1.3em; marginBottom: 2em; }
  .unplaced h2 { marginTop: 0; color: var(--contradicted); }
  .unplaced li { marginBottom: 0.8em; }

  /* Layout: article left + comments column right, single page scroll */
  .page { display: flex; justify-content: center; align-items: flex-start;
    gap: 0; max-width: 1200px; margin: 0 auto; padding: 2rem 1rem; }
  .article-col { flex: 0 1 680px; min-width: 0; padding-right: 2.5rem;
    font-size: 1.05em; line-height: 1.7; }
  .sidebar { flex: 0 0 320px; background: var(--bg);
    border-left: 1px solid var(--border); padding: 0 0 1rem 1.2rem; }
  /* Compact cards — expand on click */
  .src-card { background: var(--source-bg); border: 1px solid var(--border);
    border-left: 3px solid var(--border); border-radius: 4px;
    padding: 0.35em 0.5em; margin: 0.25em 0; cursor: pointer;
    transition: box-shadow 0.3s; overflow: hidden; max-height: 2.4em; }
  .src-card.expanded { max-height: none; cursor: default; }
  .src-card.flash { box-shadow: 0 0 0 2px var(--supported); }
  .src-card.supported { border-left-color: var(--supported); }
  .src-card.contradicted { border-left-color: var(--contradicted); }
  .src-card.unsupported { border-left-color: var(--unsupported); }
  .src-card .sc-badge { display: inline-block; padding: 0 0.35em; border-radius: 3px;
    font-family: -apple-system, sans-serif; font-size: 0.72em; font-weight: 700; margin-right: 0.4em; }
  .sc-badge.supported { background: var(--supported-bg); color: var(--supported); }
  .sc-badge.contradicted { background: var(--contradicted-bg); color: var(--contradicted); }
  .sc-badge.unsupported { background: var(--unsupported-bg); color: var(--unsupported); }
  .src-card .sc-claim { font-weight: 600; }
  .src-card .sc-rationale, .src-card .sc-matched, .src-card .sc-source,
  .src-card .sc-flags { display: none; }
  .src-card.expanded .sc-rationale, .src-card.expanded .sc-matched,
  .src-card.expanded .sc-source, .src-card.expanded .sc-flags { display: block; }
  .src-card .sc-rationale { color: var(--muted); margin: 0.3em 0; font-size: 0.85em; }
  .src-card .sc-matched { margin: 0.3em 0; padding: 0.3em 0.5em;
    background: var(--bg); border-left: 2px solid var(--border);
    font-style: italic; font-size: 0.85em; }
  .src-card .sc-source { font-size: 0.8em; font-family: monospace; word-break: break-all; }
  .src-card .sc-source a { color: var(--muted); }
  .src-card .sc-flags { font-size: 0.78em; margin-top: 0.2em; }
  .src-card .sc-num { font-size: 0.7em; color: var(--muted); float: right; }

  @media (max-width: 1000px) {
    .page { flex-direction: column; }
    .article-col { padding-right: 0; }
    .sidebar { display: none; }
  }
  .sidebar h3 { font-family: -apple-system, sans-serif; font-size: 0.9em;
    color: var(--muted); margin: 0 0 0.5rem; position: sticky; top: 0;
    background: var(--bg); padding: 0.3rem 0; z-index: 1; }
  .sidebar .src-card { background: var(--source-bg); border: 1px solid var(--border);
    border-left: 3px solid var(--border); border-radius: 4px;
    padding: 0.6em 0.7em; margin: 0.5em 0; font-size: 0.82em;
    transition: box-shadow 0.3s, border-color 0.3s; cursor: pointer; }
  .sidebar .src-card:hover { background: var(--bg); }
  .sidebar .src-card.flash { box-shadow: 0 0 0 2px var(--supported); }
  .src-card.supported { border-left-color: var(--supported); }
  .src-card.contradicted { border-left-color: var(--contradicted); }
  .src-card.unsupported { border-left-color: var(--unsupported); }
  .src-card .sc-badge { display: inline-block; padding: 0 0.35em; border-radius: 3px;
    font-family: -apple-system, sans-serif; font-size: 0.75em; font-weight: 700; margin-right: 0.4em; }
  .sc-badge.supported { background: var(--supported-bg); color: var(--supported); }
  .sc-badge.contradicted { background: var(--contradicted-bg); color: var(--contradicted); }
  .sc-badge.unsupported { background: var(--unsupported-bg); color: var(--unsupported); }
  .src-card .sc-claim { font-weight: 600; display: block; margin: 0.2em 0; }
  .src-card .sc-rationale { color: var(--muted); display: block; margin: 0.2em 0; }
  .src-card .sc-matched { display: block; margin: 0.3em 0; padding: 0.3em 0.5em;
    background: var(--bg); border-left: 2px solid var(--border);
    font-style: italic; font-size: 0.92em; }
  .src-card .sc-source { font-size: 0.85em; font-family: monospace; word-break: break-all; display: block; }
  .src-card .sc-source a { color: var(--muted); }
  .src-card .sc-flags { font-size: 0.8em; margin-top: 0.2em; display: block; }
  .src-card .sc-num { font-size: 0.7em; color: var(--muted); float: right; }

  @media (max-width: 900px) {
    body { margin-right: 0; }
    .sidebar { display: none; }
  }

  @media (prefers-color-scheme: dark) {
    :root { --text: #ddd; --bg: #1a1a1a; --muted: #999;
      --supported: #4ade80; --supported-bg: #1a3a2a;
      --contradicted: #f87171; --contradicted-bg: #3a1a1a;
      --unsupported: #facc15; --unsupported-bg: #3a351a;
      --source-bg: #252525; --border: #333; }
  }
</style>"""

_SCRIPT = """<script>
// Sidebar: always-visible, click footnote to scroll to the matching card
var sidebar = document.createElement('div');
sidebar.className = 'sidebar';
sidebar.innerHTML = '<h3>Sources</h3><div class="sidebar-cards"></div>';
var page = document.querySelector('.page');
if (page) page.appendChild(sidebar);
var cardsContainer = sidebar.querySelector('.sidebar-cards');

// Build sidebar cards from the source section at the bottom
document.querySelectorAll('.sources .source').forEach(function(src) {
  var card = document.createElement('div');
  var fnId = src.id.replace('fn', '');

  // Parse the source card's content
  var badge = src.querySelector('.badge');
  var claim = src.querySelector('.claim');
  var rationale = src.querySelector('.rationale');
  var srcLink = src.querySelector('.src-link');
  var flags = src.querySelector('.flags');

  var vclass = '';
  if (badge) {
    var bt = badge.textContent.trim();
    if (bt.indexOf('Supported') > -1) vclass = 'supported';
    else if (bt.indexOf('Contradicted') > -1) vclass = 'contradicted';
    else vclass = 'unsupported';
  }
  card.className = 'src-card ' + vclass;
  card.id = 'sc-' + fnId;

  var html = '<span class="sc-num">#' + fnId + '</span>';
  if (badge) html += '<span class="sc-badge ' + vclass + '">' + badge.innerHTML + '</span>';
  if (claim) html += '<span class="sc-claim">' + claim.innerHTML + '</span>';
  if (rationale) html += '<span class="sc-rationale">' + rationale.innerHTML + '</span>';
  if (srcLink) html += '<span class="sc-source">' + srcLink.innerHTML + '</span>';
  if (flags && flags.textContent.trim()) html += '<span class="sc-flags">' + flags.innerHTML + '</span>';

  card.innerHTML = html;
  cardsContainer.appendChild(card);

  // Click card to expand/collapse
  card.addEventListener('click', function(e) {
    card.classList.toggle('expanded');
    if (card.classList.contains('expanded')) {
      // Scroll article to this footnote
      var ref = document.getElementById('fnref' + fnId);
      if (ref) ref.scrollIntoView({behavior: 'smooth', block: 'center'});
    }
  });
});

// Wire up footnote refs: click → expand matching card + scroll page to align
document.querySelectorAll('a.fn-ref').forEach(function(ref) {
  ref.addEventListener('click', function(e) {
    e.preventDefault();
    var fnId = this.id.replace('fnref', '');
    var card = document.getElementById('sc-' + fnId);
    if (card) {
      // Collapse all other cards
      document.querySelectorAll('.src-card.expanded').forEach(function(c) {
        if (c !== card) c.classList.remove('expanded');
      });
      card.classList.add('expanded');
      // Flash briefly
      card.classList.add('flash');
      setTimeout(function() { card.classList.remove('flash'); }, 1500);
    }
  });
});
</script>"""


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

    # Remove unplaced claims block
    unplaced_match = re.match(r'(# ⚠️ UNPLACED CLAIMS.*?\n)(?=\n)', body, re.DOTALL)
    unplaced_html = ""
    if unplaced_match:
        unplaced_html = _render_unplaced(unplaced_match.group(1))
        body = body[unplaced_match.end():]

    # Parse footnote verdicts BEFORE converting body (need fn→verdict map)
    fn_verdicts = _parse_footnote_verdicts(footnotes_raw)

    # Convert body: markdown → HTML with color-coded footnote refs
    html_body = _md_to_html(body, fn_verdicts)

    # Process footnote sources
    sources_html = _render_sources(footnotes_raw)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sourced Article</title>
{_STYLE}
</head>
<body>
<div class="page">
<div class="article-col">
{unplaced_html}
{html_body}
</div>
</div>
<div class="sources">
{sources_html}
</div>
{_SCRIPT}
</body>
</html>"""

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"HTML written → {output_html}")


def _parse_footnote_verdicts(raw: str) -> dict[str, str]:
    """Parse footnote text to extract verdict per footnote number.

    Returns dict: {"1": "supported", "2": "contradicted", ...}
    """
    verdicts = {}
    for line in raw.splitlines():
        m = re.match(r'\[(\^?\d+)\]:\s+\*\*\[(.+?)\]\*\*', line)
        if m:
            fn_id = m.group(1).lstrip("^")
            verdict_raw = m.group(2).strip()
            if "Supported" in verdict_raw:
                verdicts[fn_id] = "supported"
            elif "Contradicted" in verdict_raw:
                verdicts[fn_id] = "contradicted"
            else:
                verdicts[fn_id] = "unsupported"
    return verdicts


def _md_to_html(text: str, fn_verdicts: dict[str, str]) -> str:
    """Convert basic Markdown to HTML with color-coded footnote references."""
    # Headings
    text = re.sub(r'^#### (.+)$', r'<h4>\1</h4>', text, flags=re.MULTILINE)
    text = re.sub(r'^### (.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.+)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.+)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)

    # Bold and italic
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'<strong><em>\1</em></strong>', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)

    # Footnote references: [^N] → color-coded superscript link
    text = re.sub(
        r'\[\^(\d+)\](?:\[\^(\d+)\])?',  # handle adjacent [^1][^2]
        lambda m: _render_fn_refs(m, fn_verdicts),
        text,
    )

    # Paragraphs
    paragraphs = text.split('\n\n')
    result = []
    for block in paragraphs:
        block = block.strip()
        if not block:
            continue
        if re.match(r'^<(h[1-4]|ul|ol|li|blockquote)', block):
            result.append(block)
        else:
            block = block.replace('\n', '<br>\n')
            result.append(f'<p>{block}</p>')

    return '\n'.join(result)


def _render_fn_refs(m: re.Match, fn_verdicts: dict[str, str]) -> str:
    """Render one or two adjacent footnote references with spacing and colors."""
    refs = []
    for i in range(1, 3):
        n = m.group(i)
        if not n:
            break
        v = fn_verdicts.get(n, "unsupported")
        refs.append(
            f'<a href="#fn{n}" id="fnref{n}" class="fn-ref {v}" '
            f'title="Footnote {n} — {v}">{n}</a>'
        )
    return " ".join(refs)


def _render_sources(raw: str) -> str:
    """Render the footnotes block as styled source cards with clickable links."""
    if not raw.strip():
        return ""

    lines = raw.strip().splitlines()
    sources = []
    current = None

    for line in lines:
        m = re.match(r'\[(\^?\d+)\]:\s+\*\*\[(.+?)\]\*\*\s+\[(\w+)\]\s+(.+?)\s+—\s+(.+?)(\s+⚠️.*)?$', line)
        if m:
            fn_id, verdict_raw, proximity, claim, rationale, flags = m.groups()
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
        elif line.strip().startswith("Matched:") and current:
            current["matched"] = line.strip()[8:].strip().strip('"')
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

        # Render source path/URL as clickable link
        source_text = s["source"]
        if source_text and source_text != "none":
            # Check if it's a URL
            if re.match(r'https?://', source_text):
                source_html = (
                    f'<span class="src-link">'
                    f'<a href="{_escape(source_text)}" target="_blank" rel="noopener">'
                    f'{_escape(source_text)}</a></span>'
                )
            else:
                source_html = f'<span class="src-link">{_escape(source_text)}</span>'
        else:
            source_html = ""

        matched_html = ""
        if s.get("matched"):
            matched_html = (
                f'<div style="margin:0.4em 0;padding:0.4em 0.7em;background:var(--bg);'
                f'border-left:3px solid var(--{vclass});font-style:italic;font-size:0.9em">'
                f'{_escape(s["matched"])}</div>'
            )

        html += (
            f'<div class="source {vclass}" id="fn{s["id"]}">'
            f'<span class="badge {vclass}">{_escape(v)}</span>'
            f'<span class="claim">{_escape(s["claim"])}</span>'
            f'{matched_html}'
            f'<span class="rationale">{_escape(s["rationale"])}</span>'
            f'{source_html}'
            f'<span class="flags">{flags_html}</span>'
            f'<a href="#fnref{s["id"]}" class="fn-backref" style="font-size:0.85em;margin-left:0.3em;text-decoration:none">↩</a>'
            f'</div>\n'
        )

    return html


def _render_unplaced(text: str) -> str:
    """Render the unplaced claims warning block as structured cards."""
    body = text.replace("# ⚠️ UNPLACED CLAIMS", "").strip()
    lines = body.splitlines()

    # Parse the intro paragraph
    intro_lines = []
    claim_blocks = []
    current_block = None

    for line in lines:
        if line.startswith("- **"):
            if current_block:
                claim_blocks.append(current_block)
            current_block = {"header": line.lstrip("- ").strip()}
        elif current_block is not None:
            line_s = line.strip()
            if line_s.startswith("Verdict:"):
                current_block["verdict"] = line_s.replace("Verdict:", "").strip()
            elif line_s.startswith("Source:"):
                current_block["source"] = line_s.replace("Source:", "").strip()
            elif line_s.startswith("Rationale:"):
                current_block["rationale"] = line_s.replace("Rationale:", "").strip()
            elif line_s.startswith("Failing quote:"):
                current_block["quote"] = line_s.replace("Failing quote:", "").strip()
            elif line_s:
                # Continuation of previous field
                last_key = [k for k in ["rationale", "quote", "source"] if k in current_block]
                if last_key:
                    current_block[last_key[-1]] += " " + line_s
        else:
            intro_lines.append(line)

    if current_block:
        claim_blocks.append(current_block)

    html = '<div class="unplaced"><h2>⚠️ Unplaced Claims</h2>\n'
    html += '<p>' + _escape('\n'.join(intro_lines)).replace('\n', '<br>\n') + '</p>\n'

    for cb in claim_blocks:
        v = cb.get("verdict", "")
        if "supported" in v.lower():
            vclass = "supported"
        elif "contradicted" in v.lower():
            vclass = "contradicted"
        else:
            vclass = "unsupported"

        html += '<div class="source unplaced" style="margin:0.8em 0">\n'
        html += f'<span class="badge {vclass}">{_escape(v)}</span>\n'
        html += f'<span class="claim">{_escape(cb.get("header", ""))}</span>\n'
        if cb.get("rationale"):
            html += f'<span class="rationale">{_escape(cb["rationale"][:500])}</span>\n'
        if cb.get("quote"):
            html += f'<div style="margin-top:0.3em;padding:0.3em 0.6em;background:var(--source-bg);border-left:3px solid var(--border);font-style:italic;font-size:0.9em">{_escape(cb["quote"][:400])}</div>\n'
        if cb.get("source") and cb["source"] != "none":
            src = cb["source"]
            if re.match(r'https?://', src):
                html += f'<span class="src-link"><a href="{_escape(src)}" target="_blank" rel="noopener">{_escape(src)}</a></span>\n'
            else:
                html += f'<span class="src-link">{_escape(src)}</span>\n'
        html += '</div>\n'

    html += '</div>\n'
    return html


def run_stage_d(article_sourced_md: str, output_html: str = "") -> str:
    """Convert article-sourced.md to article-sourced.html."""
    if not output_html:
        output_html = article_sourced_md.replace(".md", ".html")
    convert(article_sourced_md, output_html)
    return output_html
