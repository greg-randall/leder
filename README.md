# L.E.D.E.R.: Linguistic Engine for Draft Evaluation & Review

A YAML-driven editorial review pipeline. Extract claims, verify facts, check quotes, catch math errors. Whatever your draft needs. One codebase, one `config.yaml`, as many editorial checks as you care to write.

## 1. What it is and why

You have a long article making factual claims, and a folder of source documents that either back those claims up or knock them down. Checking every claim by hand is slow and you miss things. LEDER does the grunt work: it reads your article, pulls out the claims, dispatches real AI agents to search your documents (or the web) for evidence, and rebuilds the article with footnotes linked to sources. You review the finished product and make the final call.

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

### Run it

The pipeline reads `config.yaml` from the project root. This file controls which models to use, how many agents run at once, timeouts, turn limits, which playbooks are active, and corpus-prep settings. The defaults are sensible.

```yaml
article:
  path: "article.md"

corpus:
  root: "corpus/"

playbooks:
  dir: "pipelines/"
  active: ["fact_check"]

stage_a:
  model: "deepseek-v4-pro"
  quality_gate: true

stage_b:
  model: "deepseek-v4-pro"
  concurrency: 32
  timeout: 600
  max_turns: 60

stage_c:
  quote_match_method: "normalized"

prepare:
  source_root: "raw-source-docs/"
  convert_workers: 8
  ocr_images: true
  vision_fallback:
    enabled: true
    model: "gpt-4o-mini"
    min_words: 20
    max_pages_per_doc: 30
  audio:
    enabled: true
    model: "medium"
    device: "auto"
  summarize:
    model: "deepseek-v4-flash"
    workers: 50
  rollup:
    model: "deepseek-v4-pro"
    big_call_model: "deepseek-v4-pro[1m]"
    workers: 12
    crosscutting: true
```

Full pipeline:

```bash
python3 -m pipeline.cli all
```

Runs startup validation, then all stages. About 15 minutes for a 5,000-word article against 300 documents on DeepSeek V4 Flash.

One stage at a time:

```bash
python3 -m pipeline.cli stage-a     # extract claims via playbook prompts
python3 -m pipeline.cli stage-b     # verify claims against corpus + web
python3 -m pipeline.cli stage-c     # rebuild article with footnotes
python3 -m pipeline.cli stage-d     # HTML with color-coded source cards
python3 -m pipeline.cli stage-e     # .docx with Word comments
```

Stage A prints `Wrote 219 targets → targets.json`. Stage B prints `Done: 132 findings → findings.json`. Stage C prints `Mechanical match: 254/254 placed, 0 unmatched`. Stage D writes an HTML page with a sidebar of severity-colored source cards. Stage E writes a .docx you can upload to Google Docs with all comments intact.

### Corpus prep (`prepare-1/2/3`)

The pipeline reads a corpus of summarized markdown from `corpus.root`. Build it from a folder of raw source documents with the `prepare` commands (run once when your source docs change; NOT part of `all`):

```bash
python3 -m pipeline.cli prepare-1   # convert raw files to markdown
python3 -m pipeline.cli prepare-2   # per-document LLM summaries
python3 -m pipeline.cli prepare-3   # recursive folder summaries + crosscutting overview
python3 -m pipeline.cli prepare     # all three in order
```

How each file type gets converted (prepare-1):

- Digital PDFs, DOCX, XLSX, PPTX, HTML, MSG, EPUB, ZIP, CSV, JSON, TXT: [MarkItDown](https://github.com/microsoft/markitdown) (`pip install markitdown[all]`), which preserves tables and structure.
- Scanned or image-only PDFs: MarkItDown has no OCR, so prepare-1 falls back to local tesseract page by page. Pages whose OCR is too thin get escalated to gpt-4o-mini vision (capped at `max_pages_per_doc`).
- Image files (`.png/.jpg/.tif/.tiff/.gif/.bmp/.webp`): local tesseract OCR. Thin results escalate to vision. The vision prompt is anti-fabrication: transcribe verbatim, mark `[illegible]`, never guess.
- Audio (`.wav/.mp3/.m4a/.mp4/.flac/.ogg/.aac/.wma`): local faster-whisper (`pip install faster-whisper`). GPU first with an automatic subprocess health check. Falls back to CPU with a warning if cuDNN is missing.
- Gap-fillers for the formats MarkItDown lacks: `.eml`, `.doc/.ppt/.odt/.ods/.odp` (LibreOffice), `.rtf` (pandoc to LibreOffice), `.tsv`, `.xml`.

Set `OPENAI_API_KEY` for the vision escalation. Files that can't be converted are listed in `UNCONVERTED.md` (loud banner + nonzero exit). Files recovered via OCR, vision, whisper, or a gap-filler are listed in `NEEDS_REVIEW.md` with how each was recovered.

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
    [tiered search strategy, evaluation rubric, output instructions...]
  allowed_tools: [Read, Bash, WebSearch, WebFetch]

display:
  template: |
    {{severity_badge}} **{{target_text}}**
    {{agent_summary}}
    {% if source_excerpt %}> "{{source_excerpt}}"{% endif %}
    {% if source_path %}[source]({{source_path}}){% endif %}
```

The runner provides `{{article_text}}`, `{{existing_claims}}`, `{{article_summary}}`, `{{target_text}}`, and `{{context}}`. Playbooks reference them freely.

If you already have the current pipeline working with fact-checking: nothing breaks. The v1 `active` list contains only `fact_check`, which achieves full parity with the old hardcoded pipeline. The old CLI flags (`--claims`) still work. New playbooks get added one at a time by adding to `active`.

## 3. How it works

### Stage 1: Extraction

The runner loads each active playbook, chunks the article at paragraph boundaries, and dispatches each chunk with the playbook's extraction prompt. A quality gate (per playbook, optional) re-reads the full article to catch cross-chunk misses. Output: `targets.json`, with every target tagged by its originating playbook.

### Stage 2: Verification

For each target, the runner looks up its playbook, injects `{{article_summary}}`, `{{target_text}}`, and `{{context}}` into the verification prompt, and spawns a Claude Agent SDK agent with the playbook's allowed tools. The agent returns a unified Finding (severity, agent_summary, source_path, source_excerpt, recommended_action, confidence, metadata). Incremental writes survive crashes. Output: `findings.json`.

### Stage C: Rebuild

Each finding's anchor_text is matched to the article via sliding-window Levenshtein distance (handles smart quotes, ellipses, slight paraphrasing). Findings with the same anchor text merge into one footnote with all badges visible, even when they come from different playbooks. Severity-colored markers go at word boundaries. Output: `article-sourced.md`.

### Stage D: HTML

The sourced markdown becomes a self-contained HTML page using Bootstrap 5. Footnote pills are green (PASS), yellow (WARNING), or red (CRITICAL). A sticky sidebar on the right shows every source card. The header is the check_type. The body is rendered through the playbook's display template. Click a pill in the article and the matching sidebar card opens and scrolls into view. Output: `article-sourced.html`.

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
│   Quality gate → full-article re-read              │
│   Output: targets.json (tagged with playbook)      │
└────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────┐
│ STAGE 2: Verification (playbook-driven)            │
│   For each target: resolve playbook → prompt+tools │
│   Claude Agent SDK. One real agent per target.     │
│   Tools: per-playbook (Read, Bash, WebSearch, ...) │
│   Unified FindingOutput schema                     │
│   Incremental writes (crash-resistant)             │
│   Output: findings.json                            │
└────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────┐
│ STAGE C: Rebuild                                   │
│   Sliding-window Levenshtein for anchor_text       │
│   → article position mapping                       │
│   Severity-colored badges (PASS/WARNING/CRITICAL)  │
│   Multi-playbook merging at same anchor            │
│   Output: article-sourced.md                       │
└────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────┐
│ STAGE D: HTML                                      │
│   Markdown → Bootstrap 5 HTML                      │
│   Color-coded pills (green/yellow/red)             │
│   Sticky sidebar with playbook display templates   │
│   Output: article-sourced.html                     │
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
├── article.md                      # Input article
├── article-sourced.md              # Stage C output
├── article-sourced.html            # Stage D output
├── article-sourced.docx            # Stage E output
├── targets.json                    # Stage 1 output
├── findings.json                   # Stage 2 output
├── corpus/                         # Local document corpus + web_cache
├── debug/                          # Agent transcripts (--debug mode)
├── pipelines/                      # Playbook YAML files
│   └── fact_check.yaml             #   Fact Check playbook (v1)
├── requirements.txt
├── pipeline/
│   ├── cli.py                      # Entry point, .env loading, subcommands
│   ├── config.yaml                 # Model, concurrency, playbook, prepare settings
│   ├── config.py                   # Dataclass-based config loader
│   ├── models.py                   # Claim, ClaimsDocument (backward compat)
│   ├── finding.py                  # Finding, Target, FindingsDocument, Severity
│   ├── playbook.py                 # Playbook dataclass + YAML loader
│   ├── template_render.py          # Display template engine
│   ├── startup_check.py            # Prerequisite validation
│   ├── stage_a_extract.py          # Generic extraction runner
│   ├── stage_b_verify.py           # Generic verification runner
│   ├── stage_c_rebuild.py          # Footnote insertion (findings.json + claims.json)
│   ├── stage_d_html.py             # Bootstrap HTML with sidebar
│   ├── stage_e_docx.py             # .docx with Word comments
│   ├── prepare_1_convert.py        # Corpus prep: MarkItDown + OCR + vision
│   ├── prepare_2_summarize.py      # Corpus prep: per-document LLM summaries
│   ├── prepare_3_rollup.py         # Corpus prep: recursive folder rollup
│   ├── prepare_ocr.py              # OCR + vision fallback for scanned PDFs/images
│   ├── prepare_audio.py            # Local faster-whisper audio transcription
│   ├── prepare_vision.py           # Word-count gate + vision escalation
│   └── tools/
│       └── search_corpus.py
└── tests/
```

### Key files

`pipeline/finding.py` is the data contract. `Finding` carries `finding_id`, `check_type`, `severity` (PASS, WARNING, CRITICAL), `target_text`, `anchor_text`, `agent_summary`, `recommended_action`, `source_path`, `source_url`, `source_excerpt`, `confidence`, `human_review`, and `metadata`. `FindingsDocument` wraps article metadata and the finding list with `to_json()` and `from_json()`.

`pipeline/playbook.py` defines the Playbook dataclass and `load_playbook(path)`. A playbook is a YAML file with `extraction.prompt`, `extraction.quality_gate`, `verification.prompt`, `verification.allowed_tools`, and a `display.template`.

`pipeline/stage_a_extract.py` chunks the article and dispatches extraction prompts per playbook. When `playbook_names` is provided, it uses the generic path. When absent, the old hardcoded fact-check path runs for backward compat.

`pipeline/stage_b_verify.py` spawns Claude Agent SDK agents with playbook-specific prompts and tools. The `FindingOutput` pydantic model enforces structured output. `VerdictOutput` is kept for backward compat.

`pipeline/stage_c_rebuild.py` accepts either `findings.json` (new) or `claims.json` (old) and produces the same footnoted markdown. Severity maps to badge colors: PASS green, WARNING yellow, CRITICAL red.

`pipeline/prepare_1_convert.py` is the corpus-prep converter. Routing: images go to tesseract then vision if thin; audio goes to local faster-whisper; PDF goes to MarkItDown first with OCR fallback if empty; everything else goes to MarkItDown, then gap-fillers, then UNCONVERTED.md.

### Provider setup

The pipeline detects DeepSeek from `DEEPSEEK_API_KEY` in `.env` and configures `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_MODEL`, and related variables. Stage 2 agents use the Claude Agent SDK, which spawns Claude Code CLI processes. These inherit the environment and use DeepSeek through the Anthropic-compatible endpoint.

### Backward compatibility

The old `claims.json` format still works. The `--claims` flag on `stage-b` and `stage-c` is accepted (routed internally). Old `Claim` and `ClaimsDocument` classes in `models.py` are untouched. The old hardcoded fact-check path in `stage_a` and `stage_b` still runs when no playbook config is provided.

### Debug mode

Stage B supports `--debug N` to randomly sample N claims and save full agent transcripts:

```bash
python3 -m pipeline.cli stage-b --targets targets.json --debug 10
```

Writes `debug/{claim_id}.log` (text output) and `debug/{claim_id}.jsonl` (full message stream with every tool call and result).

### Web cache

Agents save web-fetched pages to `web_cache/{target_id}/`. After Stage B finishes, a backfill step scans for findings with a `source_url` but no cached page, and fetches them via obscura, then Jina, then curl.

### Resume and deduplication

Stage B skips targets that already have a finding in incremental output. Stage A loads playbooks once and caches them. Stage C deduplicates by `(target_text, check_type)`. Findings from different playbooks on the same anchor merge into one footnote with all badges visible.
