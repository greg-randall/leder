"""Stage A: Extract verifiable claims from a Markdown article using LLM structured output."""
from __future__ import annotations

import concurrent.futures
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

from pipeline.finding import Target, FindingsDocument
from pipeline.playbook import load_playbook
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
    raise RuntimeError(
        f"LLM did not call the expected tool. "
        f"Text response preview: {block.text[:200] if block.text else 'empty'}"
    )


_EXTRACTION_TOOL = {
    "name": "extract_claims",
    "description": "Extract verifiable factual claims from the article.",
    "input_schema": {
        "type": "object",
        "properties": {
            "article_title": {"type": "string", "description": "A short title for the article."},
            "article_summary": {"type": "string", "description": "One to two sentences summarizing what the article is about, for context when fact-checking individual claims."},
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


def _extraction_tool_for(playbook):
    """Build the structured-output tool schema for a playbook's extraction pass.
    The schema is fixed (every playbook extracts Target-shaped objects); only the
    description varies by playbook name."""
    return {
        "name": "extract_targets",
        "description": f"Extract targets for the '{playbook.name}' check from the article text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "article_title": {"type": "string",
                                  "description": "A short title for the article."},
                "article_summary": {"type": "string",
                                    "description": "One to two sentences summarizing the article."},
                "targets": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "target_text": {"type": "string",
                                            "description": "Standalone, context-injected statement."},
                            "anchor_text": {"type": "string",
                                            "description": "Verbatim excerpt from article."},
                            "claim_type": {"type": "string",
                                           "enum": ["numeric", "attribution", "legal",
                                                    "generalization"]},
                        },
                        "required": ["target_text", "anchor_text"],
                    },
                },
            },
            "required": ["targets"],
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


def _extract_targets_from_text(text: str, model: str, playbook,
                               tool_schema: dict,
                               quality_gate: bool = False,
                               existing_target_texts=None):
    """Run a playbook's extraction prompt against a block of text.
    Returns (targets, article_title, article_summary).
    """
    import anthropic
    import re as _re

    existing = existing_target_texts or []
    if quality_gate and existing:
        user_prompt = (playbook.quality_gate_prompt
                       .replace("{{existing_claims}}", "\n".join(f"- {c}" for c in existing))
                       .replace("{{article_text}}", text))
    else:
        user_prompt = playbook.extraction_prompt.replace("{{article_text}}", text)

    _leftover = _re.findall(r"\{\{.*?\}\}", user_prompt)
    if _leftover:
        print(f"  stage-a: unsubstituted placeholders in user prompt: {_leftover}",
              file=sys.stderr)

    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("DEEPSEEK_API_KEY"),
    )
    response = client.messages.create(
        model=model,
        max_tokens=8192,
        system=playbook.extraction_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        tools=[tool_schema],
        tool_choice={"type": "auto"},
        thinking={"type": "disabled"},
    )
    for block in response.content:
        if block.type == "tool_use":
            args = block.input
            targets = [Target(
                target_text=t["target_text"],
                anchor_text=t["anchor_text"],
                playbook=playbook.name,
                claim_type=t.get("claim_type"),
            ) for t in args.get("targets", [])]
            return targets, args.get("article_title", ""), args.get("article_summary", "")
    # Fallback: try to parse JSON from text response
    for block in response.content:
        if block.type == "text" and block.text:
            match = _re.search(r'\{.*"targets".*\}', block.text, _re.DOTALL)
            if match:
                try:
                    args = json.loads(match.group())
                    targets = [Target(
                        target_text=t["target_text"],
                        anchor_text=t["anchor_text"],
                        playbook=playbook.name,
                        claim_type=t.get("claim_type"),
                    ) for t in args.get("targets", [])]
                    return targets, args.get("article_title", ""), args.get("article_summary", "")
                except (json.JSONDecodeError, KeyError):
                    pass
    raise RuntimeError("LLM did not call the extraction tool or return parseable JSON.")


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
            context=_find_paragraph(article_text, c["source_quote"]),
        ))

    article_title = result.get("article_title", "Untitled")
    article_summary = result.get("article_summary", "")
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
                context=_find_paragraph(article_text, mc["source_quote"]),
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
        article_summary=article_summary,
    )


def _find_paragraph(text: str, quote: str, context_chars: int = 1500) -> str:
    """Find the paragraph containing `quote` and return surrounding context.

    Searches for the quote in the text (normalized), then expands to the
    enclosing paragraph boundaries. Truncates to ~context_chars centered
    on the quote if the paragraph is very long.
    """
    n_text = re.sub(r'\s+', ' ', text).lower()
    n_quote = re.sub(r'\s+', ' ', quote).lower().strip()
    idx = n_text.find(n_quote)
    if idx == -1:
        return ""

    # Expand to paragraph boundaries (double newlines)
    start = text.rfind('\n\n', 0, max(0, idx - 1000))
    if start == -1:
        start = 0
    else:
        start += 2
    end = text.find('\n\n', idx + len(quote))
    if end == -1:
        end = len(text)

    para = text[start:end].strip()
    if len(para) <= context_chars:
        return para
    # Truncate around the quote
    qi = para.lower().find(n_quote)
    if qi == -1:
        return para[:context_chars] + "..."
    half = context_chars // 2
    left = max(0, qi - half)
    right = min(len(para), qi + len(quote) + half)
    result = para[left:right].strip()
    if left > 0:
        result = "..." + result
    if right < len(para):
        result = result + "..."
    return result


def _chunk_article(text: str, target_words: int = 300, max_words: int = 1000) -> list[str]:
    """Split article into chunks of ~target_words, never exceeding max_words.

    1. Split on double-newlines (paragraph boundaries).
    2. Merge adjacent small chunks until each is at least target_words,
       without exceeding max_words.
    3. Any chunk still over max_words is split on sentences.
    4. Any chunk STILL over max_words is hard-split.
    """
    # Step 1: split on paragraph boundaries
    chunks = [c.strip() for c in text.split('\n\n') if c.strip()]

    # Step 2: merge small chunks with neighbors
    merged = []
    buf = ""
    for c in chunks:
        combined = (buf + "\n\n" + c).strip() if buf else c
        if len(combined.split()) <= max_words:
            buf = combined
        else:
            if buf:
                merged.append(buf)
            buf = c
    if buf:
        merged.append(buf)

    # Step 2b: merge forward — any chunk under target_words gets merged
    # with following chunks until it reaches target_words or would exceed max_words
    final = []
    i = 0
    while i < len(merged):
        c = merged[i]
        cw = len(c.split())
        # Keep merging subsequent chunks while under target_words AND combined stays under max_words
        while cw < target_words and i + 1 < len(merged):
            next_cw = len(merged[i + 1].split())
            if cw + next_cw <= max_words:
                c = c + "\n\n" + merged[i + 1]
                cw = cw + next_cw
                i += 1
            else:
                break
        final.append(c)
        i += 1

    # Step 3: split oversized chunks on sentences
    sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    result = []
    for c in final:
        if len(c.split()) <= max_words:
            result.append(c)
        else:
            parts = sentence_pattern.split(c)
            for p in parts:
                if p.strip():
                    result.append(p.strip())

    # Step 4: hard-split any remaining oversized chunks
    oversized = []
    for c in result:
        if len(c.split()) > max_words:
            words_list = c.split()
            while len(words_list) > max_words:
                oversized.append(" ".join(words_list[:max_words]))
                words_list = words_list[max_words:]
            if words_list:
                oversized.append(" ".join(words_list))
        else:
            oversized.append(c)

    return [c for c in oversized if len(c.split()) >= 10]  # Drop tiny fragments


def _save_partial_claims(output_path: str, results_by_idx: dict, total_chunks: int, article_path: str) -> None:
    """Write partial claims.json as chunks complete."""
    all_claims = []
    for i in range(total_chunks):
        all_claims.extend(results_by_idx.get(i, []))
    for j, claim in enumerate(all_claims):
        claim.claim_id = f"c{j+1:04d}"
    data = {
        "article": {"path": article_path, "title": "", "generated_at": ""},
        "corpus": {"root": "", "project": ""},
        "claims": [
            {
                "claim_id": c.claim_id, "claim_text": c.claim_text,
                "source_quote": c.source_quote, "claim_type": c.claim_type,
                "context": c.context,
                "verdict": c.verdict, "source_proximity": c.source_proximity,
                "source_path": c.source_path, "source_url": c.source_url,
                "rationale": c.rationale, "source_excerpt": c.source_excerpt,
                "human_review": c.human_review,
                "confidence": c.confidence, "reconciled": c.reconciled,
            }
            for c in all_claims
        ],
    }
    tmp = output_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, output_path)


def run_stage_a(
    article_path: str,
    output_path: str,
    corpus_root: str,
    project_name: str,
    model: str = "claude-sonnet-5",
    quality_gate: bool = True,
    playbook_dir: str = "pipelines/",
    playbook_names: list[str] | None = None,
) -> ClaimsDocument | FindingsDocument | None:
    """Read article, extract claims, write claims.json. Returns the ClaimsDocument.

    When playbook_names is provided, uses the generic playbook-driven path
    and writes targets.json. Otherwise falls back to the old hardcoded
    fact-check extraction path (backward compatible).
    """
    # --- Generic playbook path (new, used when playbook_names is provided) ---
    if playbook_names:
        from pathlib import Path as _Path

        article_text = _Path(article_path).read_text(encoding="utf-8")
        chunks = _chunk_article(article_text)

        playbooks = []
        for name in playbook_names:
            yaml_path = _Path(playbook_dir) / f"{name}.yaml"
            if yaml_path.exists():
                playbooks.append(load_playbook(str(yaml_path)))

        if not playbooks:
            raise ValueError(f"No playbooks found in {playbook_dir} matching {playbook_names}")

        all_targets = []
        article_title = ""
        article_summary = ""

        for pb in playbooks:
            tool_schema = _extraction_tool_for(pb)
            # Warn about unknown placeholders in the playbook's extraction prompt
            _unsub = re.findall(r"\{\{(?!article_text)(?!existing_claims)\w+\}\}", pb.extraction_prompt)
            if _unsub:
                print(f"  stage-a: unknown placeholders in '{pb.name}' extraction prompt: {_unsub}",
                      file=sys.stderr)
            pb_targets = []

            # Phase 1: chunk-based extraction
            for chunk in chunks:
                targets, title, summary = _extract_targets_from_text(
                    chunk, model, pb, tool_schema, quality_gate=False)
                pb_targets.extend(targets)
                if title:
                    article_title = title
                if summary:
                    article_summary = summary

            # Phase 2: quality gate (full-article re-read)
            if quality_gate and pb.quality_gate_enabled:
                print(f"  Quality gate: {pb.name} ({model})...", file=sys.stderr)
                existing_texts = [t.target_text for t in pb_targets]
                missed, _, _ = _extract_targets_from_text(
                    article_text, model, pb, tool_schema,
                    quality_gate=True, existing_target_texts=existing_texts)
                pb_targets.extend(missed)

            all_targets.extend(pb_targets)

        # Write targets.json
        doc = {
            "article_file": article_path,
            "article_title": article_title,
            "article_summary": article_summary,
            "targets": [t.to_dict() for t in all_targets],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
        print(f"Wrote {len(all_targets)} targets -> {output_path}", file=sys.stderr)
        return FindingsDocument(
            article_file=article_path,
            article_summary=article_summary,
            findings=[],
        )

    # --- Old hardcoded path (backward compat) ---
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
        print(f"Chunked into {total} chunks ({concurrency} concurrent)", file=sys.stderr)

        from tqdm import tqdm

        all_claims = []
        t0 = time.time()

        def _extract_chunk(i: int, chunk: str) -> tuple[int, list]:
            try:
                chunk_doc = extract_claims(chunk, model=model, quality_gate=False, article_path=article_path)
                return i, chunk_doc.claims
            except Exception as e:
                print(f"  Chunk {i+1} failed: {e}", file=sys.stderr)
                return i, []

        results_by_idx: dict[int, list] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(_extract_chunk, i, c): i for i, c in enumerate(chunks)}
            with tqdm(total=total, desc="  Chunks", unit="chunk") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    i, claims = future.result()
                    results_by_idx[i] = claims
                    cw = len(chunks[i].split())
                    pbar.set_postfix_str(f"{cw}w → {len(claims)} claims")
                    pbar.update(1)
                    # Incremental save — never lose progress
                    _save_partial_claims(output_path, results_by_idx, total, article_path)

        # Assemble claims in original chunk order
        for i in range(total):
            all_claims.extend(results_by_idx.get(i, []))

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
