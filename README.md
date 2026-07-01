# Source-Linking Pipeline

A reusable, fully automated pipeline that takes a Markdown article and a local document corpus, extracts every verifiable factual claim, researches each claim against the corpus (with web fallback), and rebuilds the article with numbered footnotes linking every claim to its source, verdict, and rationale.

The human reviews the finished product — verifying that all expected claims were caught, spot-checking sources, and investigating anything flagged for human review.

## How It Works

```
article.md
    │
    ▼
Stage A: DECOMPOSITION
    LLM extracts every factual claim → claims.json
    Each claim is standalone, context-injected, and anchored to the article
    by a verbatim source_quote for mechanical footnote placement.
    │
    ▼
Stage B: VERIFICATION (parallel, one REAL agent per claim)
    Each claim gets a full Claude Code agent via the Claude Agent SDK.
    Agents autonomously search, read, and evaluate:
      1. Project-level summaries → route to the right section
      2. Group-level summaries   → route to the right files
      3. File-level summaries     → identify the exact original document
      4. Original documents       → verify the claim against primary sources
      5. Web search               → for claims not in the local corpus

    Output: claims.json enriched with verdict, source, proximity, rationale.
    │
    ▼
Stage C: REBUILD
    Mechanical footnote insertion + LLM reconciliation for quotes that
    fail to match. Produces article-sourced.md with numbered [^1] footnotes.
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure your API key

Copy the example env file and add your key:

```bash
cp .env.example .env
# Edit .env with your API key
```

The pipeline auto-detects your provider. With DeepSeek (recommended — cheap, high throughput):

```
# .env
DEEPSEEK_API_KEY=sk-...
```

Or use Anthropic directly:

```
# .env
ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Verify everything works

```bash
python3 -m pipeline.cli check
```

## Prerequisites

- Python 3.10+
- [ripgrep](https://github.com/BurntSushi/ripgrep) (`sudo apt install ripgrep`)
- A DeepSeek or Anthropic API key (set in `.env`)
- The [Claude Agent SDK](https://pypi.org/project/claude-agent-sdk/) bundles the Claude Code CLI — no separate Claude Code installation needed

### Local Corpus

The pipeline expects a local corpus converted to markdown and summarized at three levels:

| Level | Purpose | Example |
|-------|---------|---------|
| Project-level | Cross-cutting patterns | `CORPUS_OVERVIEW.md` |
| Group-level | One per logical grouping | `LA-0304_Rickaway Energy/_CASE_OVERVIEW.md` |
| File-level | One per source document | `.../Reports/2021_Q3_summary.md` |

Summaries are a **search index**, not evidence. The agent uses them to find the right original document, then verifies against that original. For the current RRC permit files project, this summarization is complete.

## Usage

### Full Pipeline

```bash
python3 -m pipeline.cli all
```

This runs startup validation, then Stage A → Stage B → Stage C in sequence.

### Individual Stages

```bash
# Extract claims from an article
python3 -m pipeline.cli stage-a --article my_article.md

# Verify claims against the corpus
python3 -m pipeline.cli stage-b --claims claims.json

# Rebuild article with footnotes
python3 -m pipeline.cli stage-c --article my_article.md --claims claims.json
```

### Configuration

All settings are in `pipeline/config.yaml`:

```yaml
article:
  path: "article.md"

corpus:
  root: "source-docs-and-summaries/"

stage_a:
  model: "deepseek-v4-pro[1m]"   # Quality extraction needs a strong model
  quality_gate: true              # Second pass to catch missed claims

stage_b:
  model: "deepseek-v4-flash"     # Fast/cheap — verification is search + JSON
  concurrency: 32                # Parallel agents (disk I/O is the real cap)

stage_c:
  quote_match_method: "normalized"
```

### Model selection

| Stage | Default | Why |
|-------|---------|-----|
| A — extraction | `deepseek-v4-pro[1m]` | Decomposing prose into atomic claims needs precision |
| B — verification | `deepseek-v4-flash` | Search files, read documents, output JSON — straightforward |
| C — reconciliation | Reuses stage A model | Matching broken quotes to article text |

## Output Format

### article-sourced.md

The output article has `[^1]` footnote markers inline and a `## Sources` block at the end:

```markdown
LA-0304 was originally permitted in 2001 to Koch Midstream Services.[^1]

---

## Sources

[^1]: **[✓ Supported]** [Original] LA-0304 was originally permitted in 2001
to Koch Midstream Services. — The 2014 permit amendment states the original
authorization was issued to Koch Midstream Services in 2001.
    Source: `LA-0304_Rickaway Energy/FinalActions/2014/20141031_LF-0304.pdf.md`
```

### Footnote Legend

| Badge | Meaning |
|-------|---------|
| ✓ Supported | Source confirms the claim |
| ✗ Contradicted | Source directly contradicts the claim |
| ? Unsupported | No source found or source is silent |
| [Original] | Primary document (permit, report, statute, email) |
| [Derived] | Summary or secondary source (flagged for review) |
| [Unverifiable] | No source could be found |
| ⚠️ HUMAN REVIEW | Claim needs manual verification |
| 🔧 RECONCILED | Quote was repaired by LLM — check placement |

### Unplaced Claims

Claims that fail both mechanical matching and LLM reconciliation appear in a **⚠️ UNPLACED CLAIMS** block at the **top** of the output file. These are verified claims that couldn't be located in the article text — they require manual placement.

## Project Structure

```
g-journalism-run/
├── .env.example                        # API key template
├── article.md                          # Input article
├── article-sourced.md                  # Output (Stage C)
├── claims.json                         # Intermediate data (Stage A → B → C)
├── source-docs-and-summaries/          # Local corpus
├── pipeline/                           # Pipeline code
│   ├── cli.py                          # Entry point + .env loading
│   ├── config.yaml                     # Configuration
│   ├── config.py                       # Config loader
│   ├── models.py                       # Data model (claims.json schema)
│   ├── startup_check.py                # Prerequisite validation
│   ├── stage_a_extract.py              # Stage A — claim extraction
│   ├── stage_b_verify.py               # Stage B — verification (Claude Agent SDK)
│   ├── stage_c_rebuild.py              # Stage C — article rebuild
│   └── tools/
│       └── __init__.py
├── tests/                              # Test suite
└── requirements.txt
```
