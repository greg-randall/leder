"""Stage D: Convert sourced Markdown article to a clean HTML page with Bootstrap.

Features:
- Bootstrap grid layout (CDN) — article left, verification sidebar right
- Color-coded pill badges in body text (green/red/orange)
- Compact sidebar cards expand on click
- Dark mode support
"""
from __future__ import annotations

import re
import html as html_mod


_STYLE = """<style>
  :root {
    --supported: #2d7d46; --contradicted: #c42b2b; --unsupported: #b08800;
    --source-bg: #f8f9fa;
  }
  body { font-family: Georgia, 'Times New Roman', serif; }
  h1, h2, h3 { font-family: -apple-system, BlinkMacSystemFont, sans-serif; }
  p { margin-bottom: 1.2rem; }

  /* Footnote pills in body text */
  a.fn-ref {
    font-size: 0.75em; vertical-align: super; text-decoration: none;
    font-weight: 700; line-height: 0; margin: 0 0.12em;
    padding: 0.1em 0.35em; border-radius: 10px; color: #fff;
  }
  a.fn-ref:hover { filter: brightness(1.2); transform: scale(1.15); }
  .fn-ref.supported { background: var(--supported); }
  .fn-ref.contradicted { background: var(--contradicted); }
  .fn-ref.unsupported { background: var(--unsupported); }

  /* Sidebar cards */
  .src-card {
    background: var(--source-bg); border: 1px solid #dee2e6;
    border-left: 3px solid #dee2e6; border-radius: 4px;
    padding: 0.4em 0.6em; margin: 0.3em 0; cursor: pointer;
    overflow: hidden; max-height: 3.8em; transition: max-height 0.3s;
  }
  .src-card.expanded { max-height: 60em; cursor: default; }
  .src-card.flash { box-shadow: 0 0 0 3px rgba(45,125,70,0.4); }
  .src-card.supported { border-left-color: var(--supported); }
  .src-card.contradicted { border-left-color: var(--contradicted); }
  .src-card.unsupported { border-left-color: var(--unsupported); }
  .src-card .sc-claim { font-weight: 600; display: inline; font-size: 0.9em; }
  .src-card .sc-rationale, .src-card .sc-matched, .src-card .sc-source,
  .src-card .sc-flags { display: none; }
  .src-card.expanded .sc-rationale, .src-card.expanded .sc-matched,
  .src-card.expanded .sc-source, .src-card.expanded .sc-flags { display: block; }
  .src-card .sc-rationale { color: #6c757d; margin: 0.3em 0; font-size: 0.85em; }
  .src-card .sc-matched { margin: 0.3em 0; padding: 0.3em 0.5em;
    background: #fff; border-left: 2px solid #dee2e6;
    font-style: italic; font-size: 0.85em; }
  .src-card .sc-source { font-size: 0.8em; font-family: monospace; word-break: break-all; }
  .src-card .sc-source a { color: #6c757d; }
  .src-card .sc-flags { font-size: 0.78em; margin-top: 0.2em; }
  .src-card .sc-num { font-size: 0.85em; font-weight: 700; display: inline;
    padding: 0.1em 0.45em; border-radius: 3px;
    margin-right: 0.3em; line-height: 1.6; }
  .src-card.supported .sc-num { background: var(--supported); color: #fff; }
  .src-card.contradicted .sc-num { background: var(--contradicted); color: #fff; }
  .src-card.unsupported .sc-num { background: var(--unsupported); color: #fff; }

  /* Unplaced claims */
  .unplaced { background: #fff0f0; border: 2px solid var(--contradicted);
    border-radius: 8px; padding: 1em 1.3em; margin-bottom: 2em; }
  .unplaced h2 { margin-top: 0; color: var(--contradicted); }

  @media (prefers-color-scheme: dark) {
    :root {
      --supported: #4ade80; --contradicted: #f87171; --unsupported: #facc15;
      --source-bg: #212529;
    }
    body { background: #1a1a1a; color: #ddd; }
    .src-card .sc-matched { background: #2d2d2d; }
    .src-card .sc-source a { color: #aaa; }
    .src-card .sc-rationale { color: #aaa; }
    .unplaced { background: #2d1a1a; }
  }
</style>"""

_SCRIPT = """<script>
(function() {
  var cardsContainer = document.querySelector('.sidebar-cards');
  if (!cardsContainer) return;

  // Build sidebar cards from the hidden source section
  document.querySelectorAll('.sources .source').forEach(function(src) {
    var card = document.createElement('div');
    var fnId = src.id.replace('fn', '');

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

    var html = '<span class="sc-num ' + vclass + '">#' + fnId + '</span>';
    if (claim) html += '<span class="sc-claim">' + claim.innerHTML + '</span>';
    if (rationale) html += '<span class="sc-rationale">' + rationale.innerHTML + '</span>';
    if (srcLink) html += '<span class="sc-source">' + srcLink.innerHTML + '</span>';
    if (flags && flags.textContent.trim()) html += '<span class="sc-flags">' + flags.innerHTML + '</span>';

    card.innerHTML = html;
    cardsContainer.appendChild(card);

    // Click card to expand/collapse
    card.addEventListener('click', function() {
      card.classList.toggle('expanded');
    });
  });

  // Wire up footnote refs: click in article -> expand matching card
  document.querySelectorAll('a.fn-ref').forEach(function(ref) {
    ref.addEventListener('click', function(e) {
      e.preventDefault();
      var fnId = this.id.replace('fnref', '');
      var card = document.getElementById('sc-' + fnId);
      if (card) {
        // Collapse all others, expand this one
        document.querySelectorAll('.src-card.expanded').forEach(function(c) {
          if (c !== card) c.classList.remove('expanded');
        });
        card.classList.add('expanded');
        card.classList.add('flash');
        card.scrollIntoView({behavior: 'smooth', block: 'nearest'});
        setTimeout(function() { card.classList.remove('flash'); }, 2000);
      }
    });
  });
})();
</script>"""


def _escape(text: str) -> str:
    return html_mod.escape(text, quote=False)


def convert(article_sourced_md: str, output_html: str) -> None:
    with open(article_sourced_md, encoding="utf-8") as f:
        md = f.read()

    parts = md.split("\n---\n\n## Sources\n", 1)
    body = parts[0] if parts else md
    footnotes_raw = parts[1] if len(parts) > 1 else ""

    unplaced_match = re.match(r'(# ⚠️ UNPLACED CLAIMS.*?\n)(?=\n)', body, re.DOTALL)
    unplaced_html = ""
    if unplaced_match:
        unplaced_html = _render_unplaced(unplaced_match.group(1))
        body = body[unplaced_match.end():]

    fn_verdicts = _parse_footnote_verdicts(footnotes_raw)
    html_body = _md_to_html(body, fn_verdicts)
    sources_html = _render_sources(footnotes_raw)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sourced Article</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
{_STYLE}
</head>
<body>
<div class="container-fluid" style="max-width:1400px">
<div class="row">
<div class="col-lg-8">
{unplaced_html}
{html_body}
</div>
<div class="col-lg-4">
<div class="sidebar-cards mt-3"></div>
</div>
</div>
</div>
<div class="sources" style="display:none">{sources_html}</div>
{_SCRIPT}
</body>
</html>"""

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"HTML written → {output_html}")


def _parse_footnote_verdicts(raw: str) -> dict[str, str]:
    verdicts = {}
    for line in raw.splitlines():
        m = re.match(r'\[(\^?\d+)\]:\s+\*\*\[(.+?)\]\*\*', line)
        if m:
            fn_id = m.group(1).lstrip("^")
            v = m.group(2).strip()
            if "Supported" in v: verdicts[fn_id] = "supported"
            elif "Contradicted" in v: verdicts[fn_id] = "contradicted"
            else: verdicts[fn_id] = "unsupported"
    return verdicts


def _md_to_html(text: str, fn_verdicts: dict[str, str]) -> str:
    text = re.sub(r'^#### (.+)$', r'<h4>\1</h4>', text, flags=re.MULTILINE)
    text = re.sub(r'^### (.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.+)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.+)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'<strong><em>\1</em></strong>', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)

    def _fn_ref(m: re.Match) -> str:
        refs = []
        for i in range(1, 3):
            n = m.group(i)
            if not n: break
            v = fn_verdicts.get(n, "unsupported")
            refs.append(
                f'<a href="#fn{n}" id="fnref{n}" class="fn-ref {v}" '
                f'title="Footnote {n} — {v}">{n}</a>'
            )
        return " ".join(refs)

    text = re.sub(r'\[\^(\d+)\](?:\[\^(\d+)\])?', _fn_ref, text)

    paragraphs = text.split('\n\n')
    result = []
    for block in paragraphs:
        block = block.strip()
        if not block: continue
        if re.match(r'^<(h[1-4]|ul|ol|li)', block):
            result.append(block)
        else:
            result.append(f'<p>{block.replace(chr(10), "<br>")}</p>')
    return '\n'.join(result)


def _render_sources(raw: str) -> str:
    """Build HTML source cards (hidden) for the JS sidebar to read from."""
    if not raw.strip():
        return ""
    lines = raw.strip().splitlines()
    cards = []
    cur = None
    for line in lines:
        m = re.match(r'\[(\^?\d+)\]:\s+\*\*\[(.+?)\]\*\*\s+\[(\w+)\]\s+(.+?)\s+—\s+(.+?)(\s+⚠️.*)?$', line)
        if m:
            fn_id, verdict_raw, proximity, claim, rationale, flags = m.groups()
            cur = {"id": fn_id.lstrip("^"), "verdict": verdict_raw.strip(),
                   "claim": claim.strip(), "rationale": rationale.strip(),
                   "matched": "", "source": "", "flags": flags.strip() if flags else ""}
            cards.append(cur)
        elif line.strip().startswith("Matched:") and cur:
            cur["matched"] = line.strip()[8:].strip().strip('"')
        elif line.strip().startswith("Source:") and cur:
            cur["source"] = line.strip()[7:].strip()
    html_parts = []
    for c in cards:
        v = c["verdict"]
        vclass = "supported" if "Supported" in v else ("contradicted" if "Contradicted" in v else "unsupported")
        html_parts.append(
            f'<div class="source {vclass}" id="fn{c["id"]}">'
            f'<span class="badge {vclass}">{_escape(v)}</span>'
            f'<span class="claim">{_escape(c["claim"])}</span>'
            f'<span class="rationale">{_escape(c["rationale"])}</span>'
            f'<span class="src-link">{_escape(c["source"])}</span>'
            f'<span class="flags">{_escape(c["flags"])}</span>'
            f'</div>'
        )
    return "\n".join(html_parts)


def _render_unplaced(text: str) -> str:
    body = text.replace("# ⚠️ UNPLACED CLAIMS", "").strip()
    lines = body.splitlines()
    intro_lines = []
    claim_blocks = []
    current_block = None

    for line in lines:
        if line.startswith("- **"):
            if current_block: claim_blocks.append(current_block)
            current_block = {"header": line.lstrip("- ").strip()}
        elif current_block is not None:
            s = line.strip()
            if s.startswith("Verdict:"): current_block["verdict"] = s.replace("Verdict:", "").strip()
            elif s.startswith("Source:"): current_block["source"] = s.replace("Source:", "").strip()
            elif s.startswith("Rationale:"): current_block["rationale"] = s.replace("Rationale:", "").strip()
            elif s.startswith("Failing quote:"): current_block["quote"] = s.replace("Failing quote:", "").strip()
            elif s:
                for k in ["rationale", "quote", "source"]:
                    if k in current_block:
                        current_block[k] += " " + s
                        break
        else:
            intro_lines.append(line)

    if current_block: claim_blocks.append(current_block)

    html = '<div class="unplaced"><h2>⚠️ Unplaced Claims</h2>\n'
    html += '<p>' + _escape('\n'.join(intro_lines)).replace('\n', '<br>\n') + '</p>\n'

    for cb in claim_blocks:
        v = cb.get("verdict", "")
        vclass = "supported" if "supported" in v.lower() else ("contradicted" if "contradicted" in v.lower() else "unsupported")
        html += f'<div class="src-card {vclass} expanded" style="margin:0.5em 0">\n'
        html += f'<span class="sc-num {vclass}">⚠️</span>\n'
        html += f'<span class="sc-claim">{_escape(cb.get("header", ""))}</span>\n'
        if cb.get("rationale"):
            html += f'<span class="sc-rationale">{_escape(cb["rationale"][:500])}</span>\n'
        if cb.get("quote"):
            html += f'<span class="sc-matched">{_escape(cb["quote"][:400])}</span>\n'
        if cb.get("source") and cb["source"] != "none":
            src = cb["source"]
            if re.match(r'https?://', src):
                html += f'<span class="sc-source"><a href="{_escape(src)}" target="_blank">{_escape(src)}</a></span>\n'
            else:
                html += f'<span class="sc-source">{_escape(src)}</span>\n'
        html += '</div>\n'

    html += '</div>\n'
    return html


def run_stage_d(article_sourced_md: str, output_html: str = "") -> str:
    if not output_html:
        output_html = article_sourced_md.replace(".md", ".html")
    convert(article_sourced_md, output_html)
    return output_html
