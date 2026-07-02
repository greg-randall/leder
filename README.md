# Source-Linking Pipeline

## 1. What it is and why

You have a long article making factual claims, and a folder of source documents that either support or contradict those claims. Verifying every claim by hand is slow, error-prone, and exhausting. This pipeline automates the heavy lifting: it reads your article, pulls out every factual claim, dispatches real AI agents to search your document corpus for evidence, and rebuilds the article with color-coded footnotes linked to sources. The human reviews the finished product — spot-checking sources, investigating anything flagged for review, and making the final call.

It's built for journalism workflows where the source material is a mix of local files (permits, reports, emails, spreadsheets — anything convertible to markdown) and web sources. The pipeline is reusable: swap the article and corpus, and it works the same way.

## 2. Quickstart

### Prerequisites

- Python 3.10+
- [ripgrep](https://github.com/BurntSushi/ripgrep) (`sudo apt install ripgrep`)
- A DeepSeek API key (the pipeline auto-configures for DeepSeek via Anthropic-compatible endpoint; Anthropic keys also work)

### Install

```bash
cd g-journalism-run
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API key:
#   DEEPSEEK_API_KEY=sk-...
```

Verify everything works:

```bash
python3 -m pipeline.cli check
```

### Run the full pipeline

```bash
python3 -m pipeline.cli all
```

This runs startup validation, then all five stages in sequence. With a ~5,000-word article and ~300 documents, expect roughly 15 minutes on DeepSeek V4 Flash.

### Run a single stage

```bash
python3 -m pipeline.cli stage-a --article my_article.md --output claims.json
python3 -m pipeline.cli stage-b --claims claims.json --output claims-verified.json
python3 -m pipeline.cli stage-c --article my_article.md --claims claims-verified.json --output article-sourced.md
python3 -m pipeline.cli stage-d --input article-sourced.md --output article-sourced.html
python3 -m pipeline.cli stage-e --article article-sourced.md --claims claims-verified.json --output article-sourced.docx
```

### What success looks like

Stage A prints `Wrote 219 claims → claims.json`. Stage B prints `Done: 132 ✓ / 4 ✗ / 10 ?`. Stage C prints `Mechanical match: 254/254 placed, 0 unmatched`. Stage D writes an HTML file with a sidebar of color-coded source cards. Stage E writes a .docx you can upload to Google Docs with all comments intact.

## 3. How it works, in plain terms

**Stage A — Decomposition.** The article is split into chunks of ~300 words (at paragraph boundaries, then sentences if needed) and sent to an LLM. The LLM extracts every factual claim — standalone, context-injected statements like "LA-0304 irrigates 165 acres in Karnes County via sprinkler" rather than bare phrases like "165 acres." A quality-gate second pass catches anything missed. Output: `claims.json`.

**Stage B — Verification.** Each claim gets a real Claude Code agent — an autonomous AI with access to your computer's filesystem and the web. Agents search the local document corpus using a tiered strategy: they start with project-level summaries to identify which case folder is relevant, drill into case-level overviews to find the right document category, then read the original documents to verify the claim. If nothing exists locally (e.g., a claim about a statute or a news event), they search the web. Each agent outputs a structured verdict: supported, contradicted, or unsupported, with a rationale and a verbatim source excerpt. Claims are processed in parallel (default 32 at a time). Results are written incrementally so a crash never loses progress.

**Stage C — Rebuild.** The verified claims are matched back to their positions in the original article. Each claim carries a `source_quote` — the verbatim text from the article — which is matched against the article using a sliding-window Levenshtein distance to handle smart quotes, ellipses, and paraphrasing. Footnote markers `[^N]` are inserted at word boundaries and snapped to the end of the matched text. Output: `article-sourced.md`.

**Stage D — HTML.** The sourced markdown is converted to a self-contained HTML page using Bootstrap 5. Footnote pills are color-coded (green = supported, red = contradicted, orange = unsupported). A sticky sidebar on the right shows every source card — compact by default, expand on click. Click a footnote pill in the article, the matching sidebar card expands and scrolls into view. Output: `article-sourced.html`.

**Stage E — Word document.** The sourced article is converted to a .docx file with each footnote becoming a Word comment anchored to the relevant text. Comments include the verdict, claim, rationale, and source path. Upload to Google Drive, open with Google Docs, and all comments appear in the sidebar for collaborative editing. Output: `article-sourced.docx`.

## 4. The technical detail

### Architecture

```
article.md
    │
    ▼
┌──────────────────────────────────────────────────┐
│ Stage A: extract_claims()                         │
│   LLM (DeepSeek V4 Pro) + structured output       │
│   Chunks article (~300 words), extracts in parallel│
│   Quality gate: second pass catches missed claims  │
│   Output: claims.json                              │
└──────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────┐
│ Stage B: verify_claim() × N (parallel, async)     │
│   Claude Agent SDK — one real agent per claim      │
│   Tools: Bash (ripgrep), Read, WebSearch, WebFetch │
│   Tiered search: summaries → originals → web       │
│   Structured output via json_schema                │
│   Incremental writes (crash-resistant)             │
│   Output: claims.json (enriched with verdicts)     │
└──────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────┐
│ Stage C: insert_footnote_markers()                │
│   Sliding-window Levenshtein for source_quote      │
│   → article position mapping                       │
│   Word-boundary snapping for footnote markers      │
│   Deduplicates claims with same text               │
│   Output: article-sourced.md                       │
└──────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────┐
│ Stage D: convert()                                │
│   Markdown → Bootstrap 5 HTML                      │
│   Color-coded pill badges (green/red/orange)       │
│   Sticky sidebar with expandable source cards      │
│   Output: article-sourced.html                     │
└──────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────┐
│ Stage E: convert()                                │
│   Parses sourced markdown + claims JSON            │
│   python-docx 1.2.0 native comment API             │
│   Anchors comments to text runs                    │
│   Output: article-sourced.docx                     │
└──────────────────────────────────────────────────┘
```

### File structure

```
g-journalism-run/
├── .env.example                    # API key template
├── article.md                      # Input article
├── article-sourced.md              # Stage C output
├── article-sourced.html            # Stage D output
├── article-sourced.docx            # Stage E output
├── claims.json                     # Intermediate (Stage A → B → C)
├── web_cache/                      # Fetched page snapshots (per claim)
├── debug/                          # Agent transcripts (--debug mode)
├── source-docs-and-summaries/      # Local document corpus
├── requirements.txt
├── pipeline/
│   ├── cli.py                      # Entry point, .env loading, subcommands
│   ├── config.yaml                 # Model, concurrency, timeout settings
│   ├── config.py                   # Dataclass-based config loader
│   ├── models.py                   # Claim, ClaimsDocument, enums
│   ├── startup_check.py            # Prerequisite validation (ripgrep, etc.)
│   ├── stage_a_extract.py          # Claim extraction with chunking
│   ├── stage_b_verify.py           # Agent-based verification (Claude Agent SDK)
│   ├── stage_c_rebuild.py          # Footnote insertion with fuzzy matching
│   ├── stage_d_html.py             # Bootstrap HTML with sidebar
│   ├── stage_e_docx.py             # .docx with Word comments
│   └── tools/
│       └── __init__.py
└── tests/
```

### Key files explained

**`pipeline/models.py`** — The data contract. `Claim` is a dataclass with Stage A fields (`claim_text`, `source_quote`, `claim_type`, `context`), Stage B fields (`verdict`, `source_proximity`, `source_path`, `source_url`, `rationale`, `source_excerpt`, `human_review`, `confidence`), and Stage C fields (`reconciled`). `ClaimsDocument` wraps the article metadata + claim list with `to_json()`/`from_json()`. Every stage reads and writes this format.

**`pipeline/stage_b_verify.py`** — The most complex stage. Uses `claude-agent-sdk`'s `query()` to spawn one Claude Code process per claim. Each agent receives the system prompt (tiered search strategy, evaluation criteria, output schema), the surrounding paragraph from the article, and the claim text. `VerdictOutput` is a pydantic model passed as `output_format` — the SDK validates the JSON before we see it. `asyncio.Semaphore` controls concurrency. `_write_incremental` saves partial results after every claim. Resume support: already-verified claims are skipped on re-run.

**`pipeline/stage_c_rebuild.py`** — `find_quote_position()` strips everything except letters and spaces, then tries exact match. If that fails, it slides a window across the article computing Levenshtein distance via `difflib.SequenceMatcher`, picking the best match above 60% similarity. `_letters_pos_to_original()` maps positions back to the original article using a parallel walk. `insert_footnote_markers()` snaps insertion points to word boundaries.

**`pipeline/cli.py`** — `_load_dotenv()` reads `.env` from the project root. `_setup_provider_env()` auto-detects DeepSeek and sets `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_MODEL`, etc. `_resolve_output_path()` offers overwrite/timestamp/quit on file conflicts. All five stages are wired as subcommands, plus `all` and `check`.

### Configuration

```yaml
# pipeline/config.yaml
stage_a:
  model: "deepseek-v4-pro"     # Quality extraction needs a strong model
  quality_gate: true

stage_b:
  model: "deepseek-v4-flash"   # Fast/cheap — verification is search + JSON
  concurrency: 32              # Parallel agents (disk I/O is the real cap)
  timeout: 600                 # Per-agent timeout in seconds
  max_turns: 60                # Max tool-calling turns per agent

stage_c:
  quote_match_method: "levenshtein"  # Fuzzy matching for source_quote → position
```

### Provider setup

The pipeline auto-detects DeepSeek from `DEEPSEEK_API_KEY` in `.env` and configures all `ANTHROPIC_*` environment variables. Stages A, B, and C all use DeepSeek models (configurable per stage). Stage B agents use the Claude Agent SDK which spawns Claude Code CLI processes — these inherit the environment and use DeepSeek through the Anthropic-compatible endpoint.

### Debug mode

Stage B supports `--debug N` to randomly sample N claims and save full agent transcripts:

```bash
python3 -m pipeline.cli stage-b --claims claims.json --debug 10
```

This writes `debug/{claim_id}.log` (human-readable text output) and `debug/{claim_id}.jsonl` (full message stream including every tool call and result).

### Web cache

Agents are instructed to save web-fetched pages to `web_cache/{claim_id}/`. After Stage B completes, a backfill step scans for claims with `source_url` but no cached page, and fetches missing pages via obscura → Jina → curl.

### Resume and deduplication

Stage B skips claims that already have a `supported` or `contradicted` verdict. Stage A deduplicates claims with identical text BEFORE dispatching agents (no point verifying the same text twice). Stage C deduplicates by claim text in the output: if all copies agree on verdict, keeps the median-length rationale; if they disagree, keeps one representative per verdict.
