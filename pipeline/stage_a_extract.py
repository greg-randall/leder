"""Stage A: Extract verifiable claims from a Markdown article using LLM structured output."""
from __future__ import annotations

import concurrent.futures
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

from pipeline.models import Claim, ClaimsDocument, Article, Corpus


EXTRACTION_SYSTEM_PROMPT = """You are an expert fact-checker preparing an article for source verification.

Your goal is to extract every verifiable factual claim from the article.

Rules for claims:
1. STANDALONE: Every claim must be fully intelligible without the article context.
   - BAD: "The application fee is $50."
   - GOOD: "LA-0304's initial permit application fee was $50."
   - BAD: "It was later transferred."
   - GOOD: "LA-0304 was transferred from Hawthorn Energy Partners to Rickaway Energy Corp in 2025."

2. CONTEXT INJECTION: Inject the specific subject into every claim. Never use generic
   terms like "the permit," "the operator," or "the facility" unless qualified.

3. GRANULARITY: The smallest assertion that a single source document could verify
   or refute. Split compound sentences. Merge fragments that would share a source.

4. NO OPINIONS: Skip subjective statements, rhetorical questions, predictions,
   statements of intent ("this article will..."), and pure narrative framing.

5. PRESERVE PRECISION: Keep all numbers, dates, proper nouns, URLs, and technical
   terms exactly as written. Never paraphrase numerical data.

6. EXHAUSTIVE: Extract every distinct factual claim. Do not summarize or skip.
   For lists and comparative statements, extract each item as a separate claim.

7. SOURCE QUOTE: For each claim, provide the EXACT verbatim substring from the
   article that contains it. This quote will be used to mechanically locate where
   to place the footnote. The quote must be unique enough to match in the article text."""


QUALITY_GATE_SYSTEM_PROMPT = """You are reviewing a fact-checking extraction job. An article was
processed and claims were extracted. Your job: read the article again alongside
the list of extracted claims, and identify any factual claims that were MISSED.

Output only claims that are NOT already in the existing list. If every factual
statement was captured, return an empty list.

Apply the same rules: standalone, context-injected, no opinions, preserve precision,
exhaustive. For each missed claim, also provide the source_quote (verbatim from article)."""


def build_extraction_prompt(article_text: str) -> str:
    return f"""Extract every verifiable factual claim from the article below.

Article:
---
{article_text}
---"""


def build_quality_gate_prompt(article_text: str, existing_claims: list[str]) -> str:
    claims_list = "\n".join(f"- {c}" for c in existing_claims)
    return f"""Existing extracted claims:
{claims_list}

Article:
---
{article_text}
---

Identify any factual claims that were MISSED from the extraction above. Only return claims NOT already listed."""


def _call_llm_structured(system: str, user: str, model: str, tool_schema: dict) -> dict:
    """Call an LLM with structured output (function calling). Returns the parsed tool call args."""
    import anthropic

    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("DEEPSEEK_API_KEY"),
    )
    response = client.messages.create(
        model=model,
        max_tokens=8192,
        system=system,
        messages=[{"role": "user", "content": user}],
        tools=[tool_schema],
        tool_choice={"type": "auto"},
        thinking={"type": "disabled"},
    )
    for block in response.content:
        if block.type == "tool_use":
            return block.input
    # Fallback: model returned text instead of calling the tool.
    for block in response.content:
        if block.type == "text" and block.text:
            import re as _re
            match = _re.search(r'\{.*"claims".*\}', block.text, _re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
    raise RuntimeError("LLM did not call the expected tool and no JSON fallback found")


_EXTRACTION_TOOL = {
    "name": "extract_claims",
    "description": "Extract verifiable factual claims from the article.",
    "input_schema": {
        "type": "object",
        "properties": {
            "article_title": {"type": "string", "description": "A short title for the article."},
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim_text": {"type": "string", "description": "Standalone, context-injected factual claim."},
                        "source_quote": {"type": "string", "description": "Verbatim excerpt from the article containing this claim."},
                        "claim_type": {
                            "type": "string",
                            "enum": ["numeric", "attribution", "legal", "generalization"],
                        },
                    },
                    "required": ["claim_text", "source_quote", "claim_type"],
                },
            },
        },
        "required": ["article_title", "claims"],
    },
}

_MISSED_CLAIMS_TOOL = {
    "name": "report_missed_claims",
    "description": "Report any factual claims that were missed during extraction.",
    "input_schema": {
        "type": "object",
        "properties": {
            "missed_claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim_text": {"type": "string"},
                        "source_quote": {"type": "string"},
                        "claim_type": {
                            "type": "string",
                            "enum": ["numeric", "attribution", "legal", "generalization"],
                        },
                    },
                    "required": ["claim_text", "source_quote", "claim_type"],
                },
            },
        },
        "required": ["missed_claims"],
    },
}


def extract_claims(article_text: str, model: str = "claude-sonnet-5", quality_gate: bool = True,
                   article_path: str = "article.md") -> ClaimsDocument:
    """Extract claims from article text using LLM with structured output."""
    t0 = time.time()

    print(f"Sending article to LLM ({model}) for claim extraction...", file=sys.stderr)
    result = _call_llm_structured(
        system=EXTRACTION_SYSTEM_PROMPT,
        user=build_extraction_prompt(article_text),
        model=model,
        tool_schema=_EXTRACTION_TOOL,
    )
    t1 = time.time()

    claims = []
    for i, c in enumerate(result.get("claims", [])):
        claims.append(Claim(
            claim_id=f"c{i+1:04d}",
            claim_text=c["claim_text"],
            source_quote=c["source_quote"],
            claim_type=c["claim_type"],
        ))

    article_title = result.get("article_title", "Untitled")
    print(f"First pass: {len(claims)} claims extracted ({t1 - t0:.1f}s)", file=sys.stderr)

    if quality_gate:
        print(f"Running quality gate ({model})...", file=sys.stderr)
        existing_texts = [c.claim_text for c in claims]
        missed = _call_llm_structured(
            system=QUALITY_GATE_SYSTEM_PROMPT,
            user=build_quality_gate_prompt(article_text, existing_texts),
            model=model,
            tool_schema=_MISSED_CLAIMS_TOOL,
        )
        t2 = time.time()
        start_idx = len(claims)
        missed_count = len(missed.get("missed_claims", []))
        for j, mc in enumerate(missed.get("missed_claims", [])):
            claims.append(Claim(
                claim_id=f"c{start_idx + j + 1:04d}",
                claim_text=mc["claim_text"],
                source_quote=mc["source_quote"],
                claim_type=mc["claim_type"],
            ))
        print(f"Quality gate: {missed_count} missed claims found ({t2 - t1:.1f}s)", file=sys.stderr)

    print(f"Total: {len(claims)} claims in {time.time() - t0:.1f}s", file=sys.stderr)

    return ClaimsDocument(
        article=Article(
            path=article_path,
            title=article_title,
            generated_at=datetime.now(timezone.utc).isoformat(),
        ),
        corpus=Corpus(root="", project=""),
        claims=claims,
    )


def _chunk_article(text: str, max_words: int = 300) -> list[str]:
    """Split article into chunks suitable for claim extraction.

    Tries progressively finer delimiters until every chunk is under max_words.
    """
    # Delimiter cascade: double-newline → sentence → semicolon → comma → space
    delimiters = [
        ("double newline", r'\n\n'),
        ("sentence", r'(?<=[.!?])\s+'),
        ("semicolon", r';\s*'),
        ("comma", r',\s+'),
        ("space", r'\s+'),
    ]

    chunks = [text]

    for name, pattern in delimiters:
        if all(len(c.split()) <= max_words for c in chunks):
            break
        new_chunks = []
        for c in chunks:
            if len(c.split()) <= max_words:
                new_chunks.append(c)
            else:
                parts = re.split(pattern, c)
                new_chunks.extend(p for p in parts if p.strip())
        chunks = new_chunks

    # Final safeguard: hard-split any remaining oversized chunks
    final = []
    for c in chunks:
        words_list = c.split()
        while len(words_list) > max_words:
            final.append(" ".join(words_list[:max_words]))
            words_list = words_list[max_words:]
        if words_list:
            final.append(" ".join(words_list))
    return final


def run_stage_a(
    article_path: str,
    output_path: str,
    corpus_root: str,
    project_name: str,
    model: str = "claude-sonnet-5",
    quality_gate: bool = True,
) -> ClaimsDocument:
    """Read article, extract claims, write claims.json. Returns the ClaimsDocument."""
    if not os.path.exists(article_path):
        raise FileNotFoundError(f"Article not found: {article_path}")

    with open(article_path, encoding="utf-8") as f:
        article_text = f.read()

    if not article_text.strip():
        raise ValueError("Article is empty")

    words = len(article_text.split())
    print(f"Article: {article_path} ({words} words, {len(article_text)} chars)", file=sys.stderr)
    print(f"Model: {model}", file=sys.stderr)

    chunks = _chunk_article(article_text)
    if len(chunks) == 1:
        print(f"Single chunk ({words} words)", file=sys.stderr)
        doc = extract_claims(article_text, model=model, quality_gate=quality_gate, article_path=article_path)
    else:
        total = len(chunks)
        sizes = ", ".join(str(len(c.split())) for c in chunks)
        concurrency = min(len(chunks), 32)
        print(f"Chunked into {total} chunks (word counts: {sizes})", file=sys.stderr)
        print(f"Processing {concurrency} at a time...", file=sys.stderr)

        all_claims = []
        t0 = time.time()

        def _extract_chunk(i_chunk):
            i, chunk = i_chunk
            cw = len(chunk.split())
            chunk_doc = extract_claims(chunk, model=model, quality_gate=False, article_path=article_path)
            return (i, cw, chunk_doc.claims)

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(_extract_chunk, (i, c)): i for i, c in enumerate(chunks)}
            for future in concurrent.futures.as_completed(futures):
                i, cw, claims = future.result()
                all_claims.extend(claims)
                print(f"  Chunk {i+1}/{total} ({cw} words) → {len(claims)} claims", file=sys.stderr)

        # Renumber claims sequentially
        for j, claim in enumerate(all_claims):
            claim.claim_id = f"c{j+1:04d}"

        # Run quality gate on the full article with all existing claims
        existing_texts = [c.claim_text for c in all_claims]
        print(f"Running quality gate on full article ({model})...", file=sys.stderr)
        missed = _call_llm_structured(
            system=QUALITY_GATE_SYSTEM_PROMPT,
            user=build_quality_gate_prompt(article_text, existing_texts),
            model=model,
            tool_schema=_MISSED_CLAIMS_TOOL,
        )
        missed_count = len(missed.get("missed_claims", []))
        start_idx = len(all_claims)
        for j, mc in enumerate(missed.get("missed_claims", [])):
            all_claims.append(Claim(
                claim_id=f"c{start_idx + j + 1:04d}",
                claim_text=mc["claim_text"],
                source_quote=mc["source_quote"],
                claim_type=mc["claim_type"],
            ))
        print(f"Quality gate: {missed_count} missed claims found ({time.time() - t0:.1f}s total)", file=sys.stderr)

        doc = ClaimsDocument(
            article=Article(
                path=article_path,
                title="",
                generated_at=datetime.now(timezone.utc).isoformat(),
            ),
            corpus=Corpus(root="", project=""),
            claims=all_claims,
        )

    doc.corpus = Corpus(root=corpus_root, project=project_name)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(doc.to_json())

    print(f"Wrote {len(doc.claims)} claims -> {output_path}")
    return doc
