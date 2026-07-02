# Source-Linking Pipeline

## 1. What it is and why

You have a long article making factual claims, and a folder of source documents that either back those claims up or knock them down. Checking every claim by hand is slow and you miss things. This pipeline does the grunt work: it reads your article, pulls out the claims, dispatches real AI agents to search your documents (or the web) for evidence, and rebuilds the article with footnotes linked to sources. You review the finished product and make the final call.

It handles a mix of local files (permits, reports, emails, spreadsheets. Anything that converts to markdown.) and web sources. Swap the article and corpus and it works the same way.

## 2. Quickstart

### What you need

- Python 3.10+
- [ripgrep](https://github.com/BurntSushi/ripgrep) (`sudo apt install ripgrep`)
- A DeepSeek API key (Anthropic keys also work. The pipeline auto-configures for DeepSeek.)

### Install

```bash
cd g-journalism-run
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

The pipeline reads `config.yaml` from the project root. This file controls which models to use, how many agents run at once, timeouts, and turn limits. The defaults are sensible. You will want to point `article.path` at your own article and `corpus.root` at your own document folder. Here is the full file:

```yaml
article:
  path: "article.md"

corpus:
  root: "source-docs-and-summaries/"

stage_a:
  model: "deepseek-v4-pro"
  quality_gate: true

stage_b:
  model: "deepseek-v4-flash"
  concurrency: 32
  timeout: 600
  max_turns: 60

stage_c:
  quote_match_method: "levenshtein"
```

Full pipeline:

```bash
python3 -m pipeline.cli all
```

Runs startup validation, then all five stages. A ~5,000-word article against ~300 documents takes about 15 minutes on DeepSeek V4 Flash.

One stage at a time:

```bash
python3 -m pipeline.cli stage-a --article my_article.md --output claims.json
python3 -m pipeline.cli stage-b --claims claims.json --output claims-verified.json
python3 -m pipeline.cli stage-c --article my_article.md --claims claims-verified.json --output article-sourced.md
python3 -m pipeline.cli stage-d --input article-sourced.md --output article-sourced.html
python3 -m pipeline.cli stage-e --article article-sourced.md --claims claims-verified.json --output article-sourced.docx
```

What success looks like:

Stage A prints `Wrote 219 claims → claims.json`. Stage B prints `Done: 132 ✓ / 4 ✗ / 10 ?`. Stage C prints `Mechanical match: 254/254 placed, 0 unmatched`. Stage D writes an HTML page with a sidebar of color-coded source cards. Stage E writes a .docx you can upload to Google Docs with all comments intact.

## 3. How it works

### Stage A: Decomposition

The article is split into chunks of about 300 words, at paragraph boundaries, then sentence breaks if a chunk is too long. Each chunk goes to an LLM which extracts factual claims as standalone statements. "LA-0304 irrigates 165 acres in Karnes County via sprinkler," not "165 acres." A second pass catches anything the first pass missed. Output: `claims.json`.

### Stage B: Verification

Each claim gets a real Claude Code agent. The agent has access to your filesystem and the web. It searches the local corpus top-down: project-level summaries tell it which case folder to look in, case-level overviews point to the right document category, then it reads the original documents and checks the claim. If the claim is about a statute or a news event that isn't in the local files, the agent searches the web. Each agent returns a verdict (supported, contradicted, or unsupported), a rationale, and a verbatim excerpt from the source. Claims run in parallel, 32 at a time by default. Results save after each claim finishes, so a crash doesn't lose progress.

### Stage C: Rebuild

Each claim carries a `source_quote`, the exact text from the article. Stage C matches these quotes back to their positions. It strips everything but letters and spaces, tries an exact match first, then falls back to a sliding-window Levenshtein distance that handles smart quotes, ellipses, and slight paraphrasing. Footnote markers go at word boundaries. Output: `article-sourced.md`.

### Stage D: HTML

The sourced markdown becomes a self-contained HTML page using Bootstrap 5. Footnote pills are green for supported, red for contradicted, orange for unsupported. A sticky sidebar on the right shows every source card. Cards start compact. Click to expand. Click a pill in the article and the matching sidebar card opens and scrolls into view. Output: `article-sourced.html`.

### Stage E: Word document

The sourced article becomes a .docx. Each footnote turns into a Word comment anchored to its text. Comments contain the verdict, claim, rationale, and source path. Upload to Google Drive, open with Google Docs, and all comments show up in the sidebar. Output: `article-sourced.docx`.

## 4. Technical detail

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
│   Claude Agent SDK. One real agent per claim.      │
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

### Files

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

### Key files

**`pipeline/models.py`** is the data contract. `Claim` is a dataclass with Stage A fields (`claim_text`, `source_quote`, `claim_type`, `context`), Stage B fields (`verdict`, `source_proximity`, `source_path`, `source_url`, `rationale`, `source_excerpt`, `human_review`, `confidence`), and a Stage C field (`reconciled`). `ClaimsDocument` wraps article metadata and the claim list with `to_json()` and `from_json()`. Every stage reads and writes this format.

**`pipeline/stage_b_verify.py`** is the most complex stage. It uses `claude-agent-sdk`'s `query()` to spawn one Claude Code process per claim. Each agent gets the system prompt (tiered search strategy, evaluation criteria, output schema), the surrounding paragraph from the article, and the claim text. `VerdictOutput` is a pydantic model passed as `output_format`. The SDK validates the JSON before we touch it. `asyncio.Semaphore` controls concurrency. `_write_incremental` saves partial results after every claim. Already-verified claims are skipped on re-run.

**`pipeline/stage_c_rebuild.py`** uses `find_quote_position()` to strip everything but letters and spaces, try an exact match, then fall back to a sliding-window Levenshtein distance via `difflib.SequenceMatcher`. It picks the best match above 60% similarity. `_letters_pos_to_original()` maps positions back to the original article. `insert_footnote_markers()` snaps insertion points to word boundaries.

**`pipeline/cli.py`** loads `.env` from the project root, auto-detects DeepSeek, and sets the `ANTHROPIC_*` environment variables. It offers overwrite, timestamp, or quit on file conflicts. All five stages are wired as subcommands, plus `all` and `check`.

### Provider setup

The pipeline detects DeepSeek from `DEEPSEEK_API_KEY` in `.env` and configures `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_MODEL`, and related variables. Stage B agents use the Claude Agent SDK, which spawns Claude Code CLI processes. These inherit the environment and use DeepSeek through the Anthropic-compatible endpoint.

### Debug mode

Stage B supports `--debug N` to randomly sample N claims and save full agent transcripts:

```bash
python3 -m pipeline.cli stage-b --claims claims.json --debug 10
```

Writes `debug/{claim_id}.log` (text output) and `debug/{claim_id}.jsonl` (full message stream with every tool call and result).

### Web cache

Agents save web-fetched pages to `web_cache/{claim_id}/`. After Stage B finishes, a backfill step scans for claims with a `source_url` but no cached page, and fetches them via obscura, then Jina, then curl.

### Resume and deduplication

Stage B skips claims that already have a `supported` or `contradicted` verdict. Stage A deduplicates identical claim text before dispatching agents. Stage C deduplicates by claim text in the output: if all copies agree on verdict, it keeps the median-length rationale. If they disagree, it keeps one representative per verdict.
