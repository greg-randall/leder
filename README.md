# L.E.D.E.R.: Linguistic Engine for Draft Evaluation & Review

A YAML-driven editorial review pipeline. Extract claims, verify facts, check quotes, catch math errors. Whatever your draft needs. One codebase, one `config.yaml`, as many editorial checks as you care to write.

## 1. What it is and why

You have a long article making factual claims, and a folder of source documents that either back those claims up or knock them down. Checking every claim by hand is slow and you miss things. LEDER does the grunt work: it reads your article, pulls out the claims, dispatches real AI agents to search your documents (or the web) for evidence, then rebuilds the article with footnotes linked to sources. You review the finished product and make the final call.

It handles a mix of local files (permits, reports, emails, spreadsheets, anything that converts to markdown) and web sources. Swap the article and corpus and it works the same way.

And because every check is a YAML playbook, not hardcoded Python, adding a new editorial review (Math Check, Quote Precision, Right of Reply) means writing a few prompts, picking some tools, and writing a display template. The rest is configuration.

## 2. Quickstart

### What you need

- Python 3.10+
- [ripgrep](https://github.com/BurntSushi/ripgrep) (`sudo apt install ripgrep`)
- A DeepSeek API key (Anthropic keys also work. The pipeline auto-configures for DeepSeek.)

### Install

```bash
cd leder
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API key:
#   DEEPSEEK_API_KEY=sk-...
```

Check that everything is in place:

```bash
python3 -m pipeline.cli check
```

### Set up your corpus

Drop source documents (PDFs, DOCs, images, spreadsheets) into `corpus/`. Organize them however you like — by operator, by document type, by date. The pipeline preserves your folder structure.

Then build the search index:

```bash
python3 -m pipeline.cli prepare     # convert → summarize → rollup (all three)
```

Or step by step:

```bash
python3 -m pipeline.cli prepare-1   # convert raw files to markdown
python3 -m pipeline.cli prepare-2   # per-document LLM summaries
python3 -m pipeline.cli prepare-3   # recursive folder summaries + crosscutting overview
```

Run these whenever your source documents change. They're not part of `all` — Stage B will refuse to run if the corpus isn't prepped (use `--force-run` to bypass).

### Run the pipeline

```bash
python3 -m pipeline.cli all
```

Runs startup validation, then all five stages. About 15 minutes for a 5,000-word article against 300 documents on DeepSeek V4 Pro.

One stage at a time:

```bash
python3 -m pipeline.cli stage-a     # extract claims from article
python3 -m pipeline.cli stage-b     # verify claims against corpus + web
python3 -m pipeline.cli stage-c     # rebuild article with footnotes
python3 -m pipeline.cli stage-d     # HTML article + source viewer (output folder)
python3 -m pipeline.cli stage-e     # .docx with Word comments
```

Stage A prints `Wrote 219 targets → targets.json`. Stage B prints `Done: 132 findings → findings.json`. Stage C prints `Mechanical match: 254/254 placed, 0 unmatched`. Stage D writes an output folder containing `article.html` (a Bootstrap HTML page with a sidebar of severity-colored source cards) and a `sources/` folder with highlighted original documents linked via a full-page modal viewer. Stage E writes a .docx you can upload to Google Docs with all comments intact.

### Config

The pipeline reads `config.yaml` from the project root. Defaults are sensible — you only need to change `corpus.root` if your documents live elsewhere.

```yaml
corpus:
  root: "corpus/"
  description: >
    Transcripts of Waskom, Texas city council meetings (2024-2026)...

prepare:
  source_root: "corpus/"
  convert:
    file_timeout: 300       # seconds before a hung per-file conversion is abandoned
    min_content_bytes: 100  # minimum chars to accept a conversion as meaningful
  audio:
    enabled: true
    model: "medium"         # faster-whisper model size
    device: "auto"          # auto = GPU if available else CPU
  vision_fallback:
    enabled: true
    model: "gpt-4o-mini"
    min_image_dim: 130      # skip OCR/vision on images smaller than this
    max_pages_per_doc: 30
    dedup_images: true      # skip perceptually duplicate images

stage_a:
  model: "deepseek-v4-pro"

stage_b:
  model: "deepseek-v4-pro"
  concurrency: 32
  timeout: 600
  max_turns: 60

stage_c:
  quote_match_method: "normalized"

stage_d:
  highlight_margin: 10     # words before/after matched region in source viewer

playbooks:
  dir: "pipelines/"
  active: ["fact_check"]
```

How each file type gets converted (prepare-1):

- Already-plain-text formats (`.csv/.tsv/.txt/.json/.yaml/.log`, source code, etc. — the full list is `prepare.text_native_extensions` in `config.yaml`): copied through **verbatim** into their `.md` sidecar, not run through MarkItDown. They're already readable, so converting them only reformats/bloats them — a CSV would become a giant markdown pipe table. The sidecar is a byte-for-byte twin of the source. (`.html`/`.xml` are deliberately excluded — their converters genuinely improve them.)
- Digital PDFs, DOCX, XLSX, PPTX, HTML, EPUB, ZIP: [MarkItDown](https://github.com/microsoft/markitdown) (`pip install markitdown[all]`), which preserves tables and structure. Spreadsheet formats (`.xlsx/.ods`) get embedded table newlines collapsed so pipe tables render correctly. PDF acceptance is page-aware: MarkItDown's output is only kept when it has at least ~200 extracted characters per page (via PyMuPDF); a PDF with one digital cover page and nine scanned pages no longer passes on the strength of that single page alone.
- Scanned or image-only PDFs: MarkItDown has no OCR, so prepare-1 falls back to rasterizing pages via [PyMuPDF](https://pypi.org/project/PyMuPDF/) (`pip install pymupdf` — no external poppler/pdftoppm binary needed) and OCR'ing each page via [pytesseract](https://pypi.org/project/pytesseract/) (`pip install pytesseract`, wraps the `tesseract` binary). Pages whose OCR is too thin or garbled (spellcheck-based detection — if <50% of tokens are real words) get escalated to gpt-4o-mini vision (capped at `max_pages_per_doc`). The page-aware gate (above) means a mostly-scanned PDF with one digital page falls through to OCR correctly, rather than being accepted on that one page's text alone.
- Image files (`.png/.jpg/.tif/.tiff/.gif/.bmp/.webp`, and `.heic/.heif` via the `pillow-heif` plugin): local tesseract OCR via pytesseract. Thin or garbled results escalate to vision. The vision prompt is anti-fabrication: transcribe verbatim, mark `[illegible]`, never guess. Tiny images (email signatures, social icons — configurable via `min_image_dim`, default 130px) and perceptual duplicates (phash distance ≤ 4) are skipped and replaced with a stub naming the surviving file. Images that produce zero OCR text and can't be recovered by vision are now reported as genuine failures in `UNCONVERTED.md` (not silently succeeding with an empty stub).
- Email (`.eml`, `.msg`): routed directly to dedicated extractors — not through MarkItDown. Attachments and nested/forwarded emails are extracted to `{stem}_attachments/` with sequential numbering. A second pass auto-converts those extracted files in the same prepare-1 run — no need to run it twice. Nested `.eml` files are saved so they're picked up on the second pass too. HTML-only email bodies are converted to markdown via MarkItDown instead of landing as raw tag soup; when both `text/plain` and `text/html` parts exist, the plain-text part is preferred (avoiding near-duplicate body text).
- Audio (`.wav/.mp3/.m4a/.mp4/.flac/.ogg/.aac/.wma`): local faster-whisper (`pip install faster-whisper`). GPU first with an automatic subprocess health check. Falls back to CPU with a warning if cuDNN is missing. An optional `prepare.audio.vocabulary` config list (names, organizations, places) is passed as faster-whisper's `initial_prompt` to bias decoding toward correct proper-noun spellings.
- Subtitles (`.srt`, `.vtt`): [pycaption](https://pycaption.readthedocs.io/) (`pip install pycaption`) parses cues, then prepare-1 drops cue numbers/timestamps and collapses consecutive duplicate lines (rolling captions) into flowing transcript text — the pipeline only needs to confirm a quote is in the source, not when it was said. Chosen over webvtt-py after finding it silently drops cues containing a lone-space filler line (a pattern YouTube's auto-captions use), which loses real spoken content with no error; a small header-normalization shim handles yt-dlp's extra `Kind:`/`Language:` metadata lines, which pycaption is otherwise stricter about than webvtt-py. The output gets a factual processing header and is re-flowed at sentence/word boundaries so no line exceeds ~300 characters (some converters produce 30,000+ character single-line transcripts that break grep-based search downstream).
- Gap-fillers for the formats MarkItDown lacks: `.doc/.ppt/.odt/.ods/.odp` (LibreOffice), `.rtf` (pandoc → LibreOffice fallback), `.xml`.

Set `OPENAI_API_KEY` for the vision escalation. Files that can't be converted (and images that produce zero OCR text with no vision recovery) are listed in `UNCONVERTED.md` (loud banner + nonzero exit). Files recovered via OCR, vision, whisper, or a gap-filler are listed in `NEEDS_REVIEW.md` with how each was recovered. Replacing a source file now causes prepare-1 to re-convert it even without `--force` (detected via mtime comparison), so re-downloading or updating a corpus document doesn't leave a stale sidecar. A per-file timeout (`prepare.convert.file_timeout`, default 300s) abandons hung converters and continues the rest of the batch.

### Adding editorial checks (playbooks)

Checks live in `pipelines/` as YAML files. Add one to `playbooks.active` in `config.yaml` and it runs with the pipeline. Each playbook has three parts: an extraction prompt (what to pull out of the article), a verification prompt (how to judge it), and a display template (how findings render in the HTML sidebar and Word doc).

```yaml
# pipelines/fact_check.yaml
name: "Fact Check"
description: "Extract verifiable factual claims and verify against source documents."

extraction:
  prompt: |
    Extract every verifiable factual claim from the article below.
    Article:
    ---
    {{article_text}}
    ---
  quality_gate:
    enabled: true
    prompt: |
      Existing extracted claims:
      {{existing_claims}}
      Identify any factual claims that were MISSED.

verification:
  prompt: |
    You are a fact-checker verifying claims against a corpus of documents.
    ## Article Summary
    {{article_summary}}
    ## Claim to Verify
    {{target_text}}
    ## Article Context
    {{context}}
    [check-specific evaluation criteria, severity rubric, output instructions...]
  allowed_tools: [Read, Bash, WebSearch, WebFetch]

display:
  template: |
    {{severity_badge}} **{{target_text}}**
    {{agent_summary}}
    {% if source_excerpt %}> "{{source_excerpt}}"{% endif %}
    {% if source_path %}[source]({{source_path}}){% endif %}
```

The runner provides `{{article_text}}`, `{{existing_claims}}`, `{{article_summary}}`, `{{target_text}}`, and `{{context}}`. Playbooks reference them freely.

**Important:** The verification prompt only needs to include check-specific evaluation criteria. Generic rules — the tiered search strategy, the confidence rubric with hard caps, the mandatory `validate_excerpt` step, the corroboration principles, and the sandbox instructions — are injected automatically by `pipeline/prompts.py` and must NOT be duplicated in the playbook YAML (doing so would repeat them and bloat the prompt).

If you already have the current pipeline working with fact-checking: nothing breaks. The v1 `active` list contains only `fact_check`, which achieves full parity with the old hardcoded pipeline. The old CLI flags (`--claims`) still work. New playbooks get added one at a time by adding to `active`.

## 3. How it works

### Stage 1: Extraction

The runner loads each active playbook, chunks the article at paragraph boundaries, and dispatches each chunk with the playbook's extraction prompt. A shared system prompt (from `pipeline/prompts.py`) enforces rules for claim granularity, standalone wording, context injection, attribution framing, and anchor-text uniqueness. A quality gate (per playbook, optional) re-reads the full article to catch cross-chunk misses. Output: `targets.json`, with every target tagged by its originating playbook.

### Stage 2: Verification

For each target, the runner resolves its playbook, builds a verification prompt by prepending the generic rules block (tiered search strategy, confidence rubric with hard caps, mandatory `validate_excerpt` step, corroboration principles, sandbox instructions) to the playbook's check-specific prompt, and spawns a Claude Agent SDK agent with the playbook's allowed tools plus two in-process MCP tools (`validate_excerpt` and `fetch_page` — see below). The agent returns a structured `FindingOutput` (severity, agent_summary, source_path, source_excerpt, source_excerpt_offset, source_excerpt_similarity, confidence, human_review, recommended_action, metadata). Incremental writes survive crashes. Output: `findings.json`.

Agents run in parallel with configurable concurrency (`stage_b.concurrency`, default 32). The agent loop is async throughout.

#### In-process MCP tools

The verification agent has access to two custom tools, implemented as in-process MCP servers (`pipeline/agent_tools.py`) rather than as CLI scripts the agent invokes via Bash:

- **`validate_excerpt`** — A mandatory verification step. Before reporting any `source_excerpt`, the agent must call this tool with the source path and candidate text. The tool:
  1. Checks for an exact case-insensitive substring match (returns `actual_text`, offsets, similarity=1.0).
  2. If no exact match, chunk-scores the document by word overlap and runs sliding-window Levenshtein on the top 3 chunks (returns `actual_text` — always a real substring of the file — with offsets and similarity).
  3. If nothing matches, returns `{"found": false}`.

  The agent must use the returned `actual_text` as `source_excerpt` and the returned `offset` as `source_excerpt_offset` — never its own wording. If the tool returns `found: false`, the agent may try a different candidate, lower confidence and flag for human review, or report the finding as unverifiable. It may NOT fabricate an excerpt the tool didn't confirm.

- **`fetch_page`** — Fetches a web page and caches it for the audit trail. Four tiers: jina.ai (fast, free markdown) → obscura (headless browser for bot-protected pages) → playwright (real headless browser, last resort) → archive.is (paywall bypass via Camoufox, opt-in). Degrades through tiers rather than raising on failure. Detects paywall previews and warns the agent. Saves output atomically to `web_cache/{target_id}/page.md`.

Both tools enforce path containment: `validate_excerpt` cannot read outside the corpus root; `fetch_page`'s cache location is captured in a closure so the agent can't redirect it.

#### Filesystem sandbox

In addition to the in-process tools, agents have access to `Read`, `Grep`, and `Glob` — but these are policed by a `can_use_tool` callback (`corpus_only_permission` in `agent_tools.py`). Every path argument is resolved against the corpus root via `resolve_within()`, which follows symlinks and collapses `..` before checking containment. Glob-style pattern arguments (`pattern`, `glob`) are string-checked for `..`, absolute prefixes, `~`, and brace/bracket/backslash syntax. Tools not in the spec are denied outright. The agent's working directory IS the corpus root — agents use relative paths only.

#### Excerpt gate

After the agent returns, a code-side excerpt gate (in `stage_b_verify.py`) validates every `source_excerpt` against the claimed source file before accepting the finding:

- **Exact match** (literal substring check, case-insensitive): accepted as-is, `excerpt_status = "exact"`.
- **Repeated-prefix strip**: if the excerpt is the real text prefixed with a garbled partial repeat (a known LLM artifact), the prefix is stripped and the remainder is checked again. If it matches, accepted with `excerpt_status = "repaired"`.
- **Not found**: the excerpt is dropped (set to `None`), `source_excerpt_offset` and `source_excerpt_similarity` are cleared, and `excerpt_status = "not_found"`. The finding survives — its severity judgment and agent summary are still valid — but the sidebar won't show a highlighted source quote.
- **No source_path**: `excerpt_status = "unchecked"` (web-only findings skip this gate entirely).

Excerpts flagged as repaired or not-found log the escape for manual review.

#### Web cache

`fetch_page` saves fetched pages to `web_cache/{target_id}/page.md`. After Stage B finishes, a backfill step scans for findings with a `source_url` but no cached page, and fetches them via obscura, then Jina, then curl.

### Stage C: Rebuild

Each finding's anchor_text is matched to the article via sliding-window Levenshtein distance (handles smart quotes, ellipses, slight paraphrasing). When an anchor text matches more than one position in the article, the finding's `context` field (the surrounding paragraph the claim was extracted from) is used to disambiguate by comparing the letter-only text around each candidate position. Findings with the same anchor text merge into one footnote with all badges visible, even when they come from different playbooks. Severity-colored markers go at word boundaries. Footnotes are numbered by document position (left-to-right through the article). Output: `article-sourced.md`.

### Stage D: HTML

The sourced markdown becomes an output folder with `article.html` and a `sources/` subfolder (built by `pipeline/stage_d_sources.py`). `article.html` is a Bootstrap 5 page: footnote pills are green (PASS), yellow (WARNING), or red (CRITICAL). A sticky sidebar on the right shows every source card. The header is the check_type. The body is rendered through the playbook's display template. Click a pill in the article and the matching sidebar card opens and scrolls into view.

The `sources/` folder contains one highlighted HTML page per cited source document, with the finding's excerpt marked in yellow. Excerpt offsets come directly from the agent-verified `source_excerpt_offset` — no more fuzzy matching during Stage D. Multiple findings that cite the same document get overlapping `<mark>` spans; when two findings' excerpts overlap, the shared segment carries both finding IDs (`data-findings="a,b"`) so clicking either pill activates the matching region. Each sidebar card has an "Explore the source material" button that opens a full-page modal showing the source document (top ~2/3) with the excerpt auto-scrolled into view, plus the finding's details, article context, and a download link for the original file (bottom ~1/3). Press `n`/`p` to navigate between findings, Escape to close, and `article.html#exc-{finding_id}` links deep-link straight to a specific finding's source view.

Because the source viewer loads documents via `fetch()`, the output folder must be served over HTTP (e.g. `python3 -m http.server` from inside it, or any static host) — opening `article.html` directly via a `file://` URL will show the article but the "Explore the source material" modal won't be able to load source documents due to browser security restrictions on local file access.

Output: `article-html/article.html` + `article-html/sources/`.

### Stage E: Word document

Each finding turns into a Word comment anchored to its text. The comment includes a `[check_type] [severity]` prefix, the agent summary, source path, and recommended action. Upload to Google Drive, open with Google Docs, and all comments show up in the sidebar. Output: `article-sourced.docx`.

## 4. Technical detail

### Architecture

```
article.md
    │
    ▼
┌────────────────────────────────────────────────────┐
│ STAGE 1: Extraction (playbook-driven)              │
│   Load playbooks from YAML                         │
│   Chunk article (~300 words)                       │
│   Each chunk → LLM + playbook extraction prompt    │
│   Shared system prompt (prompts.py)                │
│   Quality gate → full-article re-read              │
│   Output: targets.json (tagged with playbook)      │
└────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────┐
│ STAGE 2: Verification (playbook-driven)            │
│   For each target: resolve playbook → prompt+tools │
│   Generic rules injected (prompts.py)              │
│   Claude Agent SDK. One real agent per target.     │
│   In-process MCP tools:                            │
│     validate_excerpt  (mandatory, 3-tier matching) │
│     fetch_page        (jina→obscura→pw→archive.is) │
│   Filesystem sandbox (corpus_only_permission)      │
│   Structured FindingOutput schema                  │
│   Excerpt gate: literal-check, repair, or drop     │
│   Incremental writes (crash-resistant)             │
│   Async concurrency (configurable, default 32)     │
│   Output: findings.json                            │
└────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────┐
│ STAGE C: Rebuild                                   │
│   Sliding-window Levenshtein for anchor_text       │
│   Context disambiguation for duplicate anchors     │
│   → article position mapping                       │
│   Severity-colored badges (PASS/WARNING/CRITICAL)  │
│   Multi-playbook merging at same anchor            │
│   Footnotes numbered by document position          │
│   Output: article-sourced.md                       │
└────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────┐
│ STAGE D: HTML                                      │
│   Markdown → Bootstrap 5 HTML                      │
│   Color-coded pills (green/yellow/red)             │
│   Sticky sidebar with playbook display templates   │
│   Sources sub-folder (stage_d_sources.py):         │
│     Agent-verified excerpt offsets (no fuzzy)      │
│     Overlapping multi-finding <mark> spans          │
│     Full-page modal source viewer (fetch-based)    │
│   Output: article-html/article.html + sources/     │
└────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────┐
│ STAGE E: Word document                             │
│   Parses sourced markdown + findings JSON          │
│   Word comments: [check_type] [severity] ...       │
│   Output: article-sourced.docx                     │
└────────────────────────────────────────────────────┘
```

### Files

```
leder/
├── .env.example                    # API key template
├── .env                            # Your API keys (gitignored)
├── article.md                      # Input article
├── article-sourced.md              # Stage C output
├── article-sourced.footnotes.json  # Stage C footnote manifest
├── article-html/                   # Stage D output (folder)
├── article-sourced.docx            # Stage E output
├── targets.json                    # Stage 1 output
├── findings.json                   # Stage 2 output
├── config.yaml                     # Pipeline configuration
├── example.config.yaml             # Annotated config reference
├── corpus/                         # Local document corpus + web_cache
├── debug/                          # Agent transcripts (--debug mode)
├── pipelines/                      # Playbook YAML files
│   └── fact_check.yaml             #   Fact Check playbook (v1)
├── requirements.txt
├── pipeline/
│   ├── cli.py                      # Entry point, .env loading, subcommands
│   ├── config.py                   # Dataclass-based config loader
│   ├── models.py                   # Legacy Claim model (backward compat)
│   ├── finding.py                  # Target, Finding, FindingsDocument, Severity
│   ├── playbook.py                 # Playbook dataclass + YAML loader
│   ├── llm.py                      # Shared single-shot text completion
│   ├── prompts.py                  # Shared prompt rulesets (extraction + verification)
│   ├── template_render.py          # Display template engine
│   ├── agent_tools.py              # In-process MCP tools + filesystem sandbox
│   ├── startup_check.py            # Prerequisite validation
│   ├── stage_a_extract.py          # Generic extraction runner
│   ├── stage_b_verify.py           # Generic verification runner (async)
│   ├── stage_c_rebuild.py          # Footnote insertion + context disambiguation
│   ├── stage_d_html.py             # Bootstrap HTML with sidebar
│   ├── stage_d_sources.py          # Source document highlighting + rendering
│   ├── stage_e_docx.py             # .docx with Word comments
│   ├── prepare_1_convert.py        # Corpus prep: MarkItDown + OCR + vision
│   ├── prepare_2_summarize.py      # Corpus prep: per-document LLM summaries
│   ├── prepare_3_rollup.py         # Corpus prep: recursive folder rollup
│   ├── prepare_ocr.py              # OCR + vision fallback for scanned PDFs/images
│   ├── prepare_audio.py            # Local faster-whisper audio transcription
│   ├── prepare_vision.py           # Word-count gate + vision escalation
│   ├── prepare_reflow.py           # Line reflow for converter output
│   └── tools/
│       ├── fetch_page.py           # Multi-tier web fetcher (jina/obscura/pw/archive.is)
│       └── validate_excerpt.py     # Mechanical excerpt verification (3-tier matching)
└── tests/
```

### Key files

`pipeline/finding.py` is the data contract. `Finding` carries `finding_id`, `check_type`, `severity` (PASS, WARNING, CRITICAL), `target_text`, `anchor_text`, `agent_summary`, `recommended_action`, `source_path`, `source_url`, `source_excerpt`, `source_excerpt_offset` (character positions in the source document), `source_excerpt_similarity` (1.0 for exact, lower for fuzzy), `excerpt_status` (exact, repaired, not_found, or unchecked), `confidence` (one of exactly 0.95, 0.8, 0.6, 0.4, or 0.2 — snapped to the nearest band on ingest), `human_review`, and `metadata` (attribution_status, corpus_contradicted_by_external, etc.). `Target` carries `target_text`, `anchor_text`, `playbook`, `context`, and `claim_type`. `FindingsDocument` wraps article metadata and the finding list with `to_json()` and `from_json()`.

`pipeline/playbook.py` defines the Playbook dataclass and `load_playbook(path)`. A playbook is a YAML file with `extraction.prompt`, `extraction.quality_gate`, `verification.prompt`, `verification.allowed_tools`, and a `display.template`.

`pipeline/prompts.py` holds shared prompt rulesets injected into every playbook. `build_extraction_system_prompt()` returns the system prompt for Stage A (claim granularity, standalone wording, context injection, attribution framing, anchor-text uniqueness, claim_type definitions). `build_verification_rules_block()` returns the generic verification rules prepended to the playbook's own prompt (tiered search strategy, `validate_excerpt` mandate, sandbox instructions, corroboration principles, confidence rubric with five bands and hard caps, date handling).

`pipeline/agent_tools.py` builds the per-claim in-process MCP server that provides `validate_excerpt` and `fetch_page` to the verification agent. Also implements `corpus_only_permission()` — the `can_use_tool` callback that confines Read/Grep/Glob to the corpus root via `resolve_within()` path containment (symlink-aware, tilde-denied, pattern-escape-checked). `_TOOL_SPEC` maps each allowed tool to its path argument and pattern arguments so the callback knows what to police.

`pipeline/tools/validate_excerpt.py` is the core of the `validate_excerpt` tool. Three tiers: exact case-insensitive substring → chunk-scored Levenshtein on the top 3 chunks (using `find_quote_position` from stage C) → `{"found": false}`. Always returns `actual_text` as a real substring of the file — callers use it in place of whatever wording they came in with.

`pipeline/tools/fetch_page.py` is the multi-tier web fetcher. Four tiers in order: jina.ai (fast, free, clean markdown) → obscura (headless browser for bot-protected pages) → playwright (real headless browser, last resort) → archive.is (Camoufox-driven paywall bypass, opt-in). Each tier degrades gracefully rather than raising. Paywall signals in the first 500 characters trigger a warning. Output is written atomically via a same-directory temp file + `os.replace`.

`pipeline/stage_a_extract.py` chunks the article and dispatches extraction prompts per playbook with a shared system prompt from `prompts.py`. `playbook_names` is required.

`pipeline/stage_b_verify.py` spawns Claude Agent SDK agents with playbook-specific prompts and tools, plus the generic verification rules block from `prompts.py`. The `FindingOutput` pydantic model enforces structured output. An excerpt gate validates every returned `source_excerpt` against the claimed source file (literal match → keep; repeated-prefix strip → repair; not found → drop). Agents run with async concurrency.

`pipeline/stage_c_rebuild.py` inserts footnotes into markdown via sliding-window Levenshtein. When an anchor matches more than one position, `_best_match_by_context()` disambiguates using the finding's context field. Footnotes are numbered by document position. Severity maps to badge colors: PASS green, WARNING yellow, CRITICAL red.

`pipeline/stage_d_sources.py` builds the `sources/` folder: resolves each finding's source path, marks excerpts in the document using agent-verified `source_excerpt_offset` values, handles overlapping multi-finding spans with combined `data-findings` attributes, and writes rendered HTML with excerpt highlighting. The old fuzzy matcher is deleted — offsets are trusted directly from the agent.

`pipeline/prepare_1_convert.py` is the corpus-prep converter. Routing: images go to tesseract (with HEIC/HEIF support via `pillow-heif`) then vision if thin or garbled (spellcheck-based detection), with tiny/duplicate images skipped and zero-text images reported as failures; audio goes to local faster-whisper (optionally vocabulary-seeded via `prepare.audio.vocabulary`); `.eml`/`.msg` go to dedicated extractors (HTML-only email bodies convert to markdown, plain-text parts preferred when both exist) with attachment extraction and an auto second pass; `.srt`/`.vtt` go to a pycaption based transcript extractor (output gets a processing header and is sentence/word-boundary re-flowed); PDF goes to MarkItDown first with page-aware OCR fallback (≥ 200 chars/page threshold); already-text formats (`prepare.text_native_extensions`) are copied through verbatim; all converter output is re-flowed so no line exceeds ~300 chars (passthrough_text excluded); everything else goes to MarkItDown, then gap-fillers, then UNCONVERTED.md. A source file newer than its `.md` sidecar is always re-converted (mtime check).

### Provider setup

The pipeline detects DeepSeek from `DEEPSEEK_API_KEY` in `.env` and configures `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_MODEL`, and related variables. Stage B agents use the Claude Agent SDK, which spawns Claude Code CLI processes. These inherit the environment and use DeepSeek through the Anthropic-compatible endpoint.

### Debug mode

Stage B supports `--debug N` to randomly sample N claims and save full agent transcripts:

```bash
python3 -m pipeline.cli stage-b --targets targets.json --debug 10
```

Writes `debug/{claim_id}.log` (text output) and `debug/{claim_id}.jsonl` (full message stream with every tool call and result).

### Resume and deduplication

Stage B skips targets that already have a finding in incremental output. Stage A loads playbooks once and caches them. Stage C deduplicates by `(target_text, check_type, anchor_text)`, so findings that share a target_text/check_type but were extracted from different anchors (e.g. fan-out clones of a near-duplicate target) survive as separate footnotes instead of collapsing. Findings from different playbooks on the same anchor merge into one footnote with all badges visible.

## Future TODOs

Ideas worth pursuing but not yet planned or scheduled:

- **Article-seeded whisper vocabulary.** Instead of a hand-curated `prepare.audio.vocabulary` in config, extract proper names from the article (which is converted first — fast, since it's text/markdown) and use them as the `initial_prompt` for faster-whisper. The article is always available at prepare-1 time (unlike `CORPUS_CROSSCUTTING.md`, which isn't built until prepare-2), and a dumb regex for consecutive capitalized words would already help without adding an ML dependency. Could be gated behind `vocabulary_source: "article"` in config so pure-audio corpora with no article still work.

- **Speaker diarization for audio.** Use a diarization model (e.g. pyannote.audio, which requires a Hugging Face token and `torch`) to label who-said-what in multi-speaker transcripts like meeting recordings. Currently prepare-1 treats all transcribed speech as one undifferentiated voice, which loses attribution information (e.g. "was this the operator or a commissioner?"). Adds a real ML dependency and HF token-gating step. **Extreme caution:** wrong speaker labels are worse than no labels at all in a fact-checking context — misattributing a quote to the wrong person can fabricate evidence rather than clarify it. Any implementation must make attribution uncertainty visible (e.g. "likely Speaker A" vs. definite labels) and treat diarization output as a hint, not ground truth.

- **Suspicious-HTML-email fallback.** Deeply-nested layout-table-heavy emails (common in templates that use tables for CSS positioning rather than real tabular data) likely produce messy, verbose markdown even after the new MarkItDown HTML-conversion path. A lightweight detection step — e.g. output that's mostly pipe tables with little readable prose — could flag those for manual review in `NEEDS_REVIEW.md` rather than silently shipping garbage downstream.
