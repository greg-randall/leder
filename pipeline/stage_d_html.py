"""Stage D: Convert sourced Markdown article to a clean HTML page with Bootstrap.

Features:
- Bootstrap grid layout (CDN) — article left, verification sidebar right
- Color-coded pill badges in body text (green/red/orange)
- Compact sidebar cards expand on click
- Dark mode support
"""
from __future__ import annotations

import html as html_mod
import json
import os
import re

from pipeline.stage_d_sources import build_sources_folder


_STYLE = """<style>
  :root {
    --pass: #2d7d46; --warning: #b08800; --critical: #c42b2b;
    --source-bg: #f8f9fa;
  }
  body { font-family: Georgia, 'Times New Roman', serif; padding: 2em 0 0 0; }
  h1, h2, h3 { font-family: -apple-system, BlinkMacSystemFont, sans-serif; }
  p { margin-bottom: 1.2rem; }

  /* Footnote pills in body text */
  a.fn-ref {
    font-size: 0.75em; vertical-align: super; text-decoration: none;
    font-weight: 700; line-height: 0; margin: 0 0.12em;
    padding: 0.1em 0.35em; border-radius: 10px; color: #fff;
  }
  a.fn-ref:hover { filter: brightness(1.2); transform: scale(1.15); }
  .fn-ref.pass { background: var(--pass); }
  .fn-ref.warning { background: var(--warning); }
  .fn-ref.critical { background: var(--critical); }

  /* Sidebar cards */
  /* Smooth scrolling everywhere */
  html { scroll-behavior: smooth; }
  .sidebar-col { position: sticky; top: 1rem; max-height: calc(100vh - 2rem);
    overflow-y: auto; scroll-behavior: smooth; }

  .src-card {
    background: var(--source-bg); border: 1px solid #dee2e6;
    border-left: 3px solid #dee2e6; border-radius: 4px;
    padding: 0.4em 0.6em; margin: 0.3em 0; cursor: pointer;
    overflow: hidden; max-height: 3.8em;
    transition: max-height 0.5s cubic-bezier(0.4, 0, 0.2, 1),
                box-shadow 0.4s ease;
  }
  a.fn-ref { transition: transform 0.2s ease, filter 0.2s ease; }
  .src-card.expanded { max-height: 60em; cursor: default; }
  .src-card.flash { box-shadow: 0 0 0 3px rgba(45,125,70,0.4); }
  .src-card.pass { border-left-color: var(--pass); }
  .src-card.warning { border-left-color: var(--warning); }
  .src-card.critical { border-left-color: var(--critical); }
  .src-card .sc-claim { font-weight: 600; display: inline; font-size: 0.9em; }
  .src-card .sc-rationale, .src-card .sc-matched, .src-card .sc-source,
  .src-card .sc-flags, .src-card .sc-recommendation, .src-card .sc-hr { display: none; }
  .src-card.expanded .sc-rationale, .src-card.expanded .sc-matched,
  .src-card.expanded .sc-source, .src-card.expanded .sc-flags,
  .src-card.expanded .sc-recommendation, .src-card.expanded .sc-hr { display: block; }
  .src-card .sc-rationale { color: #6c757d; margin: 0.3em 0; font-size: 0.85em; }
  .src-card .sc-matched { margin: 0.3em 0; padding: 0.3em 0.5em;
    background: #fff; border-left: 2px solid #dee2e6;
    font-style: italic; font-size: 0.85em; }
  .src-card .sc-matched::before { content: "Source text: "; font-weight: 600; font-style: normal; }
  .src-card .sc-hr { border: none; border-top: 1px solid #dee2e6; margin: 0.5em 0; }
  .src-card .sc-recommendation { margin: 0.3em 0; font-size: 0.85em; }
  .src-card .sc-recommendation::before { content: "Recommendation: "; font-weight: 600; }
  .src-card .sc-source { font-size: 0.8em; font-family: monospace; word-break: break-all; }
  .src-card .sc-source a { color: #6c757d; }
  .src-card .sc-flags { font-size: 0.78em; margin-top: 0.2em; }
  .src-card .sc-num { font-size: 0.85em; font-weight: 700; display: inline;
    padding: 0.1em 0.45em; border-radius: 3px;
    margin-right: 0.3em; line-height: 1.6; }
  .src-card.pass .sc-num { background: var(--pass); color: #fff; }
  .src-card.warning .sc-num { background: var(--warning); color: #fff; }
  .src-card.critical .sc-num { background: var(--critical); color: #fff; }

  /* Unplaced claims */
  .unplaced { background: #fff0f0; border: 2px solid var(--critical);
    border-radius: 8px; padding: 1em 1.3em; margin-bottom: 2em; }
  .unplaced h2 { margin-top: 0; color: var(--critical); }

  @media (prefers-color-scheme: dark) {
    :root {
      --pass: #4ade80; --critical: #f87171; --warning: #facc15;
      --source-bg: #212529;
    }
    body { background: #1a1a1a; color: #ddd; }
    .src-card .sc-matched { background: #2d2d2d; }
    .src-card .sc-hr { border-top-color: #444; }
    .src-card .sc-source a { color: #aaa; }
    .src-card .sc-rationale { color: #aaa; }
    .sc-toggle:hover { background: #ddd !important; color: #222 !important; }
    .unplaced { background: #2d1a1a; }
  }

  /* Source document modal */
  .src-modal-overlay {
    display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.6);
    z-index: 1000; align-items: center; justify-content: center;
  }
  .src-modal-overlay.open { display: flex; }
  .src-modal {
    background: #fff; width: 94vw; height: 92vh; border-radius: 8px;
    display: flex; flex-direction: column; overflow: hidden;
    box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    position: relative;
  }
  .src-modal-close {
    position: absolute; top: 0.4rem; left: 0.6rem; font-size: 1.8rem;
    background: none; border: none; cursor: pointer; color: #555; z-index: 1;
  }
  .src-modal-doc {
    flex: 2; overflow-y: auto; padding: 2rem 3rem; font-family: Georgia, serif;
  }
  .src-modal-doc mark {
    background: #fff3b0; padding: 0.05em 0.15em; scroll-margin-top: 2rem;
  }
  .src-modal-doc mark.active { background: #ffd23f; box-shadow: 0 0 0 2px #b08800; }
  .src-modal-summary-banner {
    background: #fff3cd; border: 1px solid #ffe69c; color: #664d03;
    padding: 0.6em 1em; border-radius: 4px; margin-bottom: 1rem; font-weight: 600;
  }
  .src-modal-info {
    flex: 1; overflow-y: auto; border-top: 2px solid #dee2e6; padding: 1rem 3rem;
    background: #f8f9fa; font-family: -apple-system, sans-serif; font-size: 0.9em;
  }
  .src-modal-info .simi-col { display: inline-block; vertical-align: top; width: 32%; margin-right: 1%; }
  .src-modal-info h4 { font-size: 0.85em; text-transform: uppercase; color: #888; margin-bottom: 0.3em; }
  .src-explore-btn {
    display: block; margin-top: 0.5em; font-size: 0.8em; padding: 0.3em 0.7em;
    background: #f0f0f0; border: 1px solid #ccc; border-radius: 4px; cursor: pointer;
  }
  .src-explore-btn:hover { background: #e0e0e0; }

  @media (prefers-color-scheme: dark) {
    .src-modal { background: #1a1a1a; color: #ddd; }
    .src-modal-info { background: #212529; border-top-color: #444; }
    .src-modal-doc mark { background: #4a3f0a; color: #fff3b0; }
    .src-modal-doc mark.active { background: #6b5b0f; box-shadow: 0 0 0 2px #facc15; }
  }
</style>"""

_SCRIPT = """<script>
// Eased smooth-scroll helper
function smoothScrollTo(el, target, duration) {
  var start = el.scrollTop;
  var change = target - start;
  var startTime = performance.now();
  function ease(t) { return t < 0.5 ? 2*t*t : -1+(4-2*t)*t; } // easeInOutQuad
  function animate(now) {
    var elapsed = now - startTime;
    var progress = Math.min(elapsed / duration, 1);
    el.scrollTop = start + change * ease(progress);
    if (progress < 1) requestAnimationFrame(animate);
  }
  requestAnimationFrame(animate);
}

(function() {
  var cardsContainer = document.querySelector('.sidebar-cards');
  if (!cardsContainer) return;

  // Build sidebar cards from the hidden source section
  document.querySelectorAll('.sources .source').forEach(function(src) {
    var card = document.createElement('div');
    var fnId = src.id.replace('fn', '');
    var findingId = src.getAttribute('data-finding-id') || fnId;

    var badge = src.querySelector('.badge');
    var claim = src.querySelector('.claim');
    var rationale = src.querySelector('.rationale');
    var matched = src.querySelector('.matched');
    var recommendation = src.querySelector('.recommendation');
    var srcLink = src.querySelector('.src-link');
    var flags = src.querySelector('.flags');

    var sev = src.getAttribute('data-severity');
    var vclass = 'warning';
    if (sev === 'PASS') vclass = 'pass';
    else if (sev === 'CRITICAL') vclass = 'critical';
    card.className = 'src-card ' + vclass;
    card.id = 'sc-' + fnId;

    var html = '<span class="sc-num ' + vclass + '">#' + fnId + '</span>';
    if (claim) html += '<span class="sc-claim">' + claim.innerHTML + '</span>';
    if (rationale) html += '<span class="sc-rationale">' + rationale.innerHTML + '</span>';
    if (matched && matched.textContent.trim()) html += '<span class="sc-matched">' + matched.innerHTML + '</span>';
    if (srcLink) html += '<span class="sc-source">' + srcLink.innerHTML + '</span>';
    if (flags && flags.textContent.trim()) html += '<span class="sc-flags">' + flags.innerHTML + '</span>';
    if (recommendation && recommendation.textContent.trim()) {
      html += '<hr class="sc-hr"><span class="sc-recommendation">' + recommendation.innerHTML + '</span>';
    }

    card.innerHTML = html;
    cardsContainer.appendChild(card);

    // Add explicit toggle button so text selection doesn't collapse the card
    var toggleBtn = document.createElement('button');
    toggleBtn.className = 'sc-toggle';
    toggleBtn.innerHTML = '▶';
    toggleBtn.title = 'Expand / collapse';
    toggleBtn.setAttribute('aria-label', 'Expand card');
    toggleBtn.style.cssText = (
      'position:absolute;top:6px;right:6px;' +
      'width:30px;height:30px;line-height:30px;text-align:center;' +
      'font-size:14px;color:#555;' +
      'background:#f0f0f0;border:none;border-radius:50%;' +
      'cursor:pointer;padding:0;user-select:none;'
    );
    card.style.position = 'relative';
    card.appendChild(toggleBtn);

    // Only toggle on button click, not anywhere on the card
    toggleBtn.addEventListener('click', function(e) {
      e.stopPropagation();
      card.classList.toggle('expanded');
      var expanded = card.classList.contains('expanded');
      this.innerHTML = expanded ? '▼' : '▶';
      this.setAttribute('aria-label', expanded ? 'Collapse card' : 'Expand card');
    });

    var sourceHtml = src.getAttribute('data-source-html');
    if (sourceHtml) {
      var exploreBtn = document.createElement('button');
      exploreBtn.className = 'src-explore-btn';
      exploreBtn.textContent = 'Explore the source material';
      exploreBtn.addEventListener('click', function(e) {
        e.stopPropagation();
        openSourceModal(findingId, sourceHtml, src);
      });
      card.appendChild(exploreBtn);
    }
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
        // Scroll after expansion animation finishes (0.5s transition)
        card.addEventListener('transitionend', function onEnd() {
          card.removeEventListener('transitionend', onEnd);
          card.scrollIntoView({behavior: 'smooth', block: 'start'});
        });
        setTimeout(function() { card.classList.remove('flash'); }, 2500);
      }
    });
  });
})();
</script>
<script>
var _explorableFindingIds = [];
var _currentModalFindingId = null;
document.querySelectorAll('.sources .source[data-source-html]').forEach(function(src) {
  var fid = src.getAttribute('data-finding-id');
  if (fid) _explorableFindingIds.push(fid);
});

function openSourceModal(fnId, sourceHtml, sourceDiv) {
  _currentModalFindingId = fnId;
  console.log('openSourceModal: findingId=' + fnId + ', source=' + sourceHtml);
  var overlay = document.getElementById('srcModalOverlay');
  var docPane = document.getElementById('srcModalDoc');
  var infoPane = document.getElementById('srcModalInfo');

  var isSummary = sourceDiv.getAttribute('data-is-summary') === 'true';
  var claim = sourceDiv.querySelector('.claim');
  var rationale = sourceDiv.querySelector('.rationale');
  var recommendation = sourceDiv.querySelector('.recommendation');
  var context = sourceDiv.querySelector('.context');
  var srcLink = sourceDiv.querySelector('.src-link');
  var sev = sourceDiv.getAttribute('data-severity');

  var originalHref = sourceHtml.replace(/\\.html$/, '');
  infoPane.innerHTML =
    '<div class="simi-col"><h4>Finding</h4>' +
    '<strong>' + sev + '</strong><br>' + (rationale ? rationale.innerHTML : '') +
    (recommendation && recommendation.textContent.trim() ? '<br><em>' + recommendation.innerHTML + '</em>' : '') +
    '</div>' +
    '<div class="simi-col"><h4>Article context</h4>' +
    (context ? context.innerHTML : '(no surrounding context captured)') +
    '</div>' +
    '<div class="simi-col"><h4>Original document</h4>' +
    '<a href="' + originalHref + '" download>Download original file</a><br>' +
    '<span style="color:#888;font-size:0.85em">' + (srcLink ? srcLink.innerHTML : '') + '</span>' +
    '</div>';

  docPane.innerHTML = '<p>Loading…</p>';
  overlay.classList.add('open');
  document.body.style.overflow = 'hidden';
  window.location.hash = 'exc-' + fnId;

  fetch(sourceHtml).then(function(r) { return r.text(); }).then(function(text) {
    var banner = isSummary
      ? '<div class="src-modal-summary-banner">⚠ This is a summary, not the primary source.</div>'
      : '';
    docPane.innerHTML = banner + text;
    console.log('Modal loaded. Looking for #exc-' + fnId);
    var target = docPane.querySelector('#exc-' + fnId);
    // Apply .active to ALL segments for this finding -- merged/split
    // segments carrying this finding's ID in data-findings must also highlight.
    docPane.querySelectorAll('mark[data-findings]').forEach(function(el) {
      if (el.dataset.findings.split(',').indexOf(fnId) !== -1) {
        el.classList.add('active');
        // Use the first data-findings match as the scroll target if the
        // exact id match wasn't found (overlap-at-top case).
        if (!target) { target = el; }
      }
    });
    if (target) {
      console.log('Found target, scrolling...');
      if (!target.classList.contains('active')) {
        target.classList.add('active');
      }
      target.scrollIntoView({block: 'center'});
    } else {
      console.log('Target not found. Available exc- IDs:');
      docPane.querySelectorAll('[id^="exc-"]').forEach(function(el) {
        console.log('  ' + el.id);
      });
    }
  }).catch(function(err) {
    console.log('Fetch failed: ' + err);
    docPane.innerHTML = '<p>Could not load the source document.</p>';
  });
}

function closeSourceModal() {
  document.getElementById('srcModalOverlay').classList.remove('open');
  document.body.style.overflow = '';
  if (window.location.hash.indexOf('exc-') === 1) {
    history.replaceState(null, '', window.location.pathname);
  }
}

function navigateModal(direction) {
  if (_currentModalFindingId === null || _explorableFindingIds.length === 0) return;
  var idx = _explorableFindingIds.indexOf(_currentModalFindingId);
  if (idx === -1) return;
  var nextIdx = (idx + direction + _explorableFindingIds.length) % _explorableFindingIds.length;
  var nextId = _explorableFindingIds[nextIdx];
  var nextSourceDiv = document.getElementById('fn' + nextId);
  var nextSourceHtml = nextSourceDiv.getAttribute('data-source-html');
  openSourceModal(nextId, nextSourceHtml, nextSourceDiv);
}

document.getElementById('srcModalClose').addEventListener('click', closeSourceModal);
document.getElementById('srcModalOverlay').addEventListener('click', function(e) {
  if (e.target === this) closeSourceModal();
});

document.addEventListener('keydown', function(e) {
  var overlay = document.getElementById('srcModalOverlay');
  if (!overlay.classList.contains('open')) return;
  var tag = (e.target.tagName || '').toLowerCase();
  if (tag === 'input' || tag === 'textarea') return;

  if (e.key === 'Escape') {
    closeSourceModal();
  } else if (e.key === 'n') {
    navigateModal(1);
  } else if (e.key === 'p') {
    navigateModal(-1);
  }
});

(function() {
  var hash = window.location.hash;
  if (hash.indexOf('#exc-') !== 0) return;
  var fnId = hash.slice('#exc-'.length);
  console.log('Deep-link: looking for data-finding-id=' + fnId);
  var sourceDiv = document.querySelector('.source[data-finding-id="' + fnId + '"]');
  if (sourceDiv && sourceDiv.getAttribute('data-source-html')) {
    console.log('Found source div, opening modal...');
    openSourceModal(fnId, sourceDiv.getAttribute('data-source-html'), sourceDiv);
  } else {
    console.log('No explorable source div found for ' + fnId);
  }
})();
</script>
"""


def _escape(text: str) -> str:
    return html_mod.escape(text, quote=False)


def convert(article_sourced_md: str, findings_path: str, output_dir: str,
            corpus_root: str, web_cache_dir: str,
            highlight_margin: int = 10) -> None:
    with open(article_sourced_md, encoding="utf-8") as f:
        md = f.read()

    os.makedirs(output_dir, exist_ok=True)
    with open(findings_path, encoding="utf-8") as f:
        findings_doc = json.load(f)
    source_key_map = build_sources_folder(
        findings_doc.get("findings", []), corpus_root, web_cache_dir, output_dir,
        highlight_margin=highlight_margin,
    )
    # _render_sources parses the "Source:" line stage-c already wrote into the
    # sourced markdown, verbatim -- and stage_c_rebuild.py wraps corpus
    # source_path values in backticks there (source_ref = f"`{claim.source_path}`"),
    # while web source_url values are written raw, unwrapped. The lookup keys
    # built here must match that exact text, backticks included for corpus paths,
    # not build_sources_folder's own (backtick-free) keys.
    source_map: dict[str, str] = {}
    for f in findings_doc.get("findings", []):
        source_path = f.get("source_path")
        source_url = f.get("source_url")
        finding_id = f.get("finding_id", "")
        # Normalize source_path the same way resolve_cited_sources does.
        normalized = None
        if source_path:
            sp = source_path
            if os.path.isabs(sp):
                corpus_abs = os.path.abspath(corpus_root)
                if sp.startswith(corpus_abs + os.sep):
                    sp = os.path.relpath(sp, corpus_abs)
            for prefix in ("corpus/", "./"):
                if sp.startswith(prefix):
                    sp = sp[len(prefix):]
                    break
            if sp in source_key_map:
                normalized = sp
        # Determine the output-relative HTML path for this finding.
        html_path = None
        if normalized:
            html_path = source_key_map[normalized]
        elif source_url and finding_id in source_key_map:
            html_path = source_key_map[finding_id]
        if not html_path:
            continue
        # Map every prefix variant that might appear in the Source: line.
        for variant in {source_path, f"./{source_path}", f"corpus/{source_path}"}:
            if variant:
                source_map[f"`{variant}`"] = html_path
        if source_url:
            source_map[source_url] = html_path
    findings_list = findings_doc.get("findings", [])

    parts = md.split("\n---\n\n## Sources\n", 1)
    body = parts[0] if parts else md
    footnotes_raw = parts[1] if len(parts) > 1 else ""

    # Enrich each finding with surrounding article context (a few sentences
    # around the target_text, with the target itself underlined).
    _enrich_contexts(findings_list, body)

    unplaced_match = re.match(r'(# ⚠️ UNPLACED CLAIMS.*?)\n---\n\n', body, re.DOTALL)
    unplaced_html = ""
    if unplaced_match:
        unplaced_html = _render_unplaced(unplaced_match.group(1))
        body = body[unplaced_match.end():]

    fn_verdicts = _parse_footnote_verdicts(footnotes_raw)
    html_body = _md_to_html(body, fn_verdicts)
    # Stage C's footnote-number -> finding_id map, written beside the sourced
    # markdown. Absent for markdown produced before the sidecar existed.
    footnote_map: dict[str, str] = {}
    sidecar_path = os.path.splitext(article_sourced_md)[0] + ".footnotes.json"
    if os.path.exists(sidecar_path):
        with open(sidecar_path, encoding="utf-8") as f:
            footnote_map = json.load(f)

    sources_html = _render_sources(footnotes_raw, source_map, findings_list, footnote_map)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sourced Article</title>
<link href="https://unpkg.com/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
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
<div class="sidebar-col">
<div class="sidebar-cards mt-3"></div>
</div>
</div>
</div>
</div>
<div class="sources" style="display:none">{sources_html}</div>
<div class="src-modal-overlay" id="srcModalOverlay">
  <div class="src-modal">
    <button class="src-modal-close" id="srcModalClose" aria-label="Close">×</button>
    <div class="src-modal-doc" id="srcModalDoc"></div>
    <div class="src-modal-info" id="srcModalInfo"></div>
  </div>
</div>
{_SCRIPT}
</body>
</html>"""

    article_path = os.path.join(output_dir, "article.html")
    with open(article_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"HTML written → {article_path}")


def _parse_footnote_verdicts(raw: str) -> dict[str, str]:
    verdicts = {}
    for line in raw.splitlines():
        m = re.match(r'\[(\^?\d+)\]:\s+\*\*\[(.+?)\]\*\*', line)
        if m:
            fn_id = m.group(1).lstrip("^")
            v = m.group(2).strip()
            # New format: "✓ fact_check", "✗ fact_check", "? fact_check"
            if v.startswith("✓"):
                verdicts[fn_id] = "pass"
            elif v.startswith("✗"):
                verdicts[fn_id] = "critical"
            elif v.upper() in ("PASS", "SUPPORTED"):
                verdicts[fn_id] = "pass"
            elif v.upper() in ("CRITICAL", "CONTRADICTED"):
                verdicts[fn_id] = "critical"
            else:
                verdicts[fn_id] = "warning"
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
            if not n:
                break
            v = fn_verdicts.get(n, "warning")
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
        if not block:
            continue
        if re.match(r'^<(h[1-4]|ul|ol|li)', block):
            result.append(block)
        else:
            result.append(f'<p>{block.replace(chr(10), "<br>")}</p>')
    return '\n'.join(result)


# Same markers Stage B uses to detect a cited source that's one of our own
# generated rollup/summary files rather than a primary source document.
_SUMMARY_SOURCE_MARKERS = (
    "_summary", "_FOLDER_SUMMARY", "CORPUS_OVERVIEW", "CORPUS_ROLLUP",
    "ALL_SUMMARIES", "INDEX.md",
)


def _is_generated_summary_source(path: str) -> bool:
    if not path:
        return False
    return any(m in path for m in _SUMMARY_SOURCE_MARKERS)


def _enrich_contexts(findings: list[dict], article_body: str) -> None:
    """For each finding with a target_text, find its surrounding paragraph
    in the article body and store a few sentences of context (with the
    closest-matching sentence underlined) in finding['_context_html'].

    Uses word-overlap scoring rather than exact substring match because
    stage B's LLM often rephrases target_text (e.g. spelling out a pronoun
    as a full company name), so the target rarely appears verbatim.
    """
    import re as _re

    def _words(s: str) -> set[str]:
        return set(_re.sub(r'[^a-zA-Z0-9 ]', '', _re.sub(r'\[\^?\d+\]', '', s.lower())).split())

    # Strip markdown headers and unplaced block from body for searching.
    clean_body = _re.sub(r'^# .*\n', '', article_body, flags=_re.MULTILINE)
    clean_body = _re.sub(r'# ⚠️ UNPLACED CLAIMS.*?\n---\n\n', '', clean_body, flags=_re.DOTALL)
    paragraphs = [p.strip() for p in clean_body.split('\n\n') if p.strip()]

    for f in findings:
        target = f.get('target_text', '').strip()
        if not target:
            continue
        target_words = _words(target)
        if not target_words:
            continue

        # Find the paragraph with the highest word overlap.
        best_para = ''
        best_score = 0
        for p in paragraphs:
            score = len(target_words & _words(p))
            if score > best_score:
                best_score = score
                best_para = p
        if best_score < max(2, len(target_words) * 0.25):
            continue

        # Split the best paragraph into sentences, find the one with
        # the highest word overlap to the target.
        sentences = _re.split(r'(?<=[.!?])\s+', best_para)
        best_sent_idx = 0
        best_sent_score = 0
        for i, s in enumerate(sentences):
            score = len(target_words & _words(s))
            if score > best_sent_score:
                best_sent_score = score
                best_sent_idx = i

        # Window: 2 sentences before, best, 2 after.
        start = max(0, best_sent_idx - 2)
        end = min(len(sentences), best_sent_idx + 3)
        window = sentences[start:end]
        if best_sent_score >= 2:
            window[best_sent_idx - start] = f'<u>{window[best_sent_idx - start]}</u>'
        f['_context_html'] = ' '.join(window)


def _render_sources(raw: str, source_map: dict[str, str] | None = None,
                    findings_list: list[dict] | None = None,
                    footnote_map: dict[str, str] | None = None) -> str:
    """Build HTML source cards (hidden) for the JS sidebar to read from."""
    if not raw.strip():
        return ""
    source_map = source_map or {}
    findings_list = findings_list or []
    footnote_map = footnote_map or {}
    lines = raw.strip().splitlines()
    cards = []
    cur = None
    for line in lines:
        m = re.match(r'\[(\^?\d+)\]:\s+\*\*\[(.+?)\]\*\*\s+\[(\w+)\]\s+(.+?)\s+—\s+(.+?)(\s+⚠️.*)?$', line)
        if m:
            fn_id, verdict_raw, proximity, claim, rationale, flags = m.groups()
            cur = {"id": fn_id.lstrip("^"), "verdict": verdict_raw.strip(),
                   "claim": claim.strip(), "rationale": rationale.strip(),
                   "matched": "", "recommendation": "", "source": "",
                   "flags": flags.strip() if flags else ""}
            cards.append(cur)
        elif line.strip().startswith("Matched:") and cur:
            cur["matched"] = line.strip()[8:].strip().strip('"')
        elif line.strip().startswith("Recommendation:") and cur:
            raw = line.strip()[len("Recommendation:"):].strip()
            if raw.lower() in ("null", "none", ""):
                raw = ""
            cur["recommendation"] = raw
        elif line.strip().startswith("Source:") and cur:
            cur["source"] = line.strip()[7:].strip()
            if _is_generated_summary_source(cur["source"]):
                cur["flags"] += " ⚠️ Summary source (not a primary document)"
    html_parts = []
    for c in cards:
        raw = c["verdict"]
        # New format: "✓ fact_check", "✗ fact_check", "? fact_check"
        if raw.startswith("✓"):
            severity = "PASS"
        elif raw.startswith("✗"):
            severity = "CRITICAL"
        elif raw.upper() in ("PASS", "SUPPORTED"):
            severity = "PASS"
        elif raw.upper() in ("CRITICAL", "CONTRADICTED"):
            severity = "CRITICAL"
        else:
            severity = "WARNING"
        vclass = severity.lower()
        source_html = source_map.get(c["source"], "")
        extra_attrs = f' data-source-html="{_escape(source_html)}"' if source_html else ""
        if _is_generated_summary_source(c["source"]):
            extra_attrs += ' data-is-summary="true"'
        # Map footnote number to the actual finding for scroll-target matching.
        # Stage C numbers footnotes by document position, so the positional
        # index into findings.json is only a fallback for sourced markdown
        # written before the sidecar map existed.
        finding: dict = {}
        finding_id = footnote_map.get(c["id"], "")
        if finding_id:
            finding = next(
                (f for f in findings_list if f.get("finding_id") == finding_id), {})
        else:
            try:
                idx = int(c["id"]) - 1
            except ValueError:
                idx = -1
            if 0 <= idx < len(findings_list):
                finding = findings_list[idx]
                finding_id = finding.get("finding_id", "")
        if finding_id:
            extra_attrs += f' data-finding-id="{_escape(finding_id)}"'
        raw_context = finding.get("_context_html") or finding.get("context") or finding.get("target_text", "")
        context_html = raw_context  # _context_html is pre-escaped, plain text is safe as-is
        rec = _escape(c["recommendation"])
        html_parts.append(
            f'<div class="source {vclass}" id="fn{c["id"]}" data-severity="{severity}"{extra_attrs}>'
            f'<span class="badge {vclass}">{_escape(raw)}</span>'
            f'<span class="claim">{_escape(c["claim"])}</span>'
            f'<span class="rationale">{_escape(c["rationale"])}</span>'
            f'<span class="matched">{_escape(c["matched"])}</span>'
            + (f'<span class="recommendation">{rec}</span>' if rec else '')
            + f'<span class="context">{context_html if finding.get("_context_html") else _escape(context_html)}</span>'
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
            if current_block:
                claim_blocks.append(current_block)
            current_block = {"header": line.lstrip("- ").strip()}
        elif current_block is not None:
            s = line.strip()
            if s.startswith("Verdict:"):
                current_block["verdict"] = s.replace("Verdict:", "").strip()
            elif s.startswith("Source:"):
                current_block["source"] = s.replace("Source:", "").strip()
            elif s.startswith("Rationale:"):
                current_block["rationale"] = s.replace("Rationale:", "").strip()
            elif s.startswith("Failing quote:"):
                current_block["quote"] = s.replace("Failing quote:", "").strip()
            elif s:
                for k in ["rationale", "quote", "source"]:
                    if k in current_block:
                        current_block[k] += " " + s
                        break
        else:
            intro_lines.append(line)

    if current_block:
        claim_blocks.append(current_block)

    html = '<div class="unplaced"><h2>⚠️ Unplaced Claims</h2>\n'
    html += '<p>' + _escape('\n'.join(intro_lines)).replace('\n', '<br>\n') + '</p>\n'

    for cb in claim_blocks:
        v = cb.get("verdict", "").upper()
        if v in ("PASS", "SUPPORTED"):
            vclass = "pass"
        elif v in ("CRITICAL", "CONTRADICTED"):
            vclass = "critical"
        else:
            vclass = "warning"
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


def run_stage_d(article_sourced_md: str, findings_path: str, output_dir: str,
                corpus_root: str, web_cache_dir: str,
                highlight_margin: int = 10) -> str:
    convert(article_sourced_md, findings_path, output_dir, corpus_root, web_cache_dir,
            highlight_margin=highlight_margin)
    return os.path.join(output_dir, "article.html")
