"""Stage A: Extract verifiable claims from a Markdown article using LLM structured output."""
from __future__ import annotations

import json
import os
import re
import sys

from pipeline.finding import Target, FindingsDocument
from pipeline.playbook import load_playbook
from pipeline.stage_c_rebuild import find_quote_position


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


def _extract_targets_from_text(text: str, model: str, playbook,
                               tool_schema: dict, system_prompt: str,
                               quality_gate: bool = False,
                               existing_target_texts=None,
                               chunk_context_brief: str = ""):
    """Run a playbook's extraction prompt against a block of text.

    system_prompt is the shared extraction system prompt (see
    pipeline.prompts.build_extraction_system_prompt) sent as the LLM
    `system` param -- required, not the raw playbook.extraction_prompt
    template. chunk_context_brief, when set, is prepended to the user
    message as a short context note (used for chunks after the first).
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

    if chunk_context_brief:
        user_prompt = (
            f"This chunk is from an article about: {chunk_context_brief}. "
            f"Resolve pronouns and generic references against this context.\n\n"
            f"{user_prompt}"
        )

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
        system=system_prompt,
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
                playbook=getattr(playbook, 'slug', playbook.name),
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
                        playbook=getattr(playbook, 'slug', playbook.name),
                        claim_type=t.get("claim_type"),
                    ) for t in args.get("targets", [])]
                    return targets, args.get("article_title", ""), args.get("article_summary", "")
                except (json.JSONDecodeError, KeyError):
                    pass
    raise RuntimeError("LLM did not call the extraction tool or return parseable JSON.")


def _find_paragraph(text: str, quote: str, context_chars: int = 1500,
                    lead_chars: int = 600) -> str:
    """Find the paragraph containing `quote` and return surrounding context.

    Locating the quote is delegated to stage C's `find_quote_position`, which
    matches on a letters-only projection of both strings. That tolerance is the
    whole point: the extraction model routinely drops markdown emphasis when
    copying an anchor verbatim (article says `the **operator cleanup program**`,
    the anchor comes back `the operator cleanup program`). A plain
    whitespace-normalized `find` misses every one of those and yields no context
    at all for the claim.

    The window is the enclosing paragraph, extended backward a whole paragraph
    at a time while the added lead-in stays under `lead_chars`. The lead-in
    matters because in a markdown timeline the date header (`Sep 2021`) and
    section headings (`## Key Facts`) are separate paragraphs -- a claim from a
    dated entry needs that date, which the enclosing paragraph alone would drop.

    Returns "" when the quote cannot be located at all.
    """
    pos = find_quote_position(quote, text)
    if pos is None:
        return ""
    idx, quote_end = pos

    # Enclosing paragraph (double newlines)
    para_start = text.rfind('\n\n', 0, idx)
    para_start = 0 if para_start == -1 else para_start + 2
    para_end = text.find('\n\n', quote_end)
    if para_end == -1:
        para_end = len(text)

    # Extend backward by whole paragraphs while the lead-in fits the budget.
    # `start` is always 0 or a boundary index + 2, so `start - 2` never
    # re-finds the break we just consumed.
    start = para_start
    while start > 0:
        prev = text.rfind('\n\n', 0, start - 2)
        prev = 0 if prev == -1 else prev + 2
        if para_start - prev > lead_chars:
            break
        start = prev

    window = text[start:para_end]
    if len(window) <= context_chars:
        return window.strip()

    # Truncate around the quote using its known absolute offsets -- no second
    # search, so the returned window always contains the quote.
    q_off = idx - start
    half = context_chars // 2
    left = max(0, q_off - half)
    right = min(len(window), (quote_end - start) + half)
    result = window[left:right].strip()
    if left > 0:
        result = "..." + result
    if right < len(window):
        result = result + "..."
    return result


def _chunk_article(text: str, target_words: int = 300, max_words: int = 1000) -> list[str]:
    """Split article into chunks of ~target_words, never exceeding max_words.

    1. Split on double-newlines (paragraph boundaries).
    2. Merge adjacent small chunks until each is at least target_words,
       without exceeding max_words.
    3. Any chunk still over max_words is split on sentences.
    4. Any chunk STILL over max_words is hard-split.
    5. Any resulting tiny (<10-word) fragment is merged into the previous
       chunk (capped at max_words) rather than dropped.
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

    # Step 5: merge tiny trailing fragments into the previous chunk instead
    # of dropping them outright -- a short standalone paragraph (e.g. a
    # pull quote) could otherwise be lost entirely and never sent to claim
    # extraction. Only merges when it fits under max_words, so it can't
    # undo step 3/4's oversized-chunk splitting by re-agglomerating a long
    # run of short sentence fragments back into one big blob.
    final_chunks: list[str] = []
    for c in oversized:
        cw = len(c.split())
        if (cw < 10 and final_chunks
                and len(final_chunks[-1].split()) + cw <= max_words):
            final_chunks[-1] = final_chunks[-1] + "\n\n" + c
        else:
            final_chunks.append(c)

    return final_chunks


def run_stage_a(
    article_path: str,
    output_path: str,
    corpus_root: str,
    project_name: str,
    model: str = "claude-sonnet-5",
    quality_gate: bool = True,
    playbook_dir: str = "pipelines/",
    playbook_names: list[str] | None = None,
    corpus_description: str = "",
) -> FindingsDocument:
    """Read article, extract targets via the playbook-driven path, write targets.json."""
    if not playbook_names:
        raise ValueError("playbook_names is required (no playbooks configured)")

    if not os.path.exists(article_path):
        raise FileNotFoundError(f"Article not found: {article_path}")

    from pathlib import Path as _Path
    from pipeline.prompts import build_extraction_system_prompt

    article_text = _Path(article_path).read_text(encoding="utf-8")
    if not article_text.strip():
        raise ValueError("Article is empty")
    chunks = _chunk_article(article_text)
    extraction_system_prompt = build_extraction_system_prompt(corpus_description)

    playbooks = []
    for name in playbook_names:
        yaml_path = _Path(playbook_dir) / f"{name}.yaml"
        if yaml_path.exists():
            pb = load_playbook(str(yaml_path))
            pb.slug = name  # filename stem = canonical ID for stage-b lookup
            playbooks.append(pb)

    if not playbooks:
        raise ValueError(f"No playbooks found in {playbook_dir} matching {playbook_names}")

    all_targets = []
    article_title = ""
    article_summary = ""
    unlocatable_anchors: list[str] = []

    for pb in playbooks:
        tool_schema = _extraction_tool_for(pb)
        # Warn about unknown placeholders in the playbook's extraction prompt
        _unsub = re.findall(r"\{\{(?!article_text)(?!existing_claims)\w+\}\}", pb.extraction_prompt)
        if _unsub:
            print(f"  stage-a: unknown placeholders in '{pb.name}' extraction prompt: {_unsub}",
                  file=sys.stderr)
        pb_targets = []

        # Phase 1: chunk-based extraction, sequential -- chunk 1's summary
        # becomes the context brief for every later chunk (Tweak 3.3).
        chunk_brief = ""
        for i, chunk in enumerate(chunks):
            targets, title, summary = _extract_targets_from_text(
                chunk, model, pb, tool_schema, quality_gate=False,
                system_prompt=extraction_system_prompt,
                chunk_context_brief=chunk_brief)
            pb_targets.extend(targets)
            if i == 0:
                chunk_brief = summary
            # First non-empty wins (was: last chunk wins).
            if title and not article_title:
                article_title = title
            if summary and not article_summary:
                article_summary = summary

        # Phase 2: quality gate (full-article re-read)
        if quality_gate and pb.quality_gate_enabled:
            print(f"  Quality gate: {pb.name} ({model})...", file=sys.stderr)
            existing_texts = [t.target_text for t in pb_targets]
            missed, _, _ = _extract_targets_from_text(
                article_text, model, pb, tool_schema,
                quality_gate=True, existing_target_texts=existing_texts,
                system_prompt=extraction_system_prompt)
            pb_targets.extend(missed)

        for t in pb_targets:
            t.context = _find_paragraph(article_text, t.anchor_text)
            if not t.context:
                unlocatable_anchors.append(t.anchor_text)

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
    if unlocatable_anchors:
        # These targets reach stage-b verification with no article context at
        # all. Never let that be silent -- it went unnoticed for a long time.
        print(f"  stage-a: WARNING -- {len(unlocatable_anchors)}/{len(all_targets)} anchors "
              f"could not be located in the article; those claims have no context:",
              file=sys.stderr)
        for a in unlocatable_anchors:
            print(f"    - {a!r}", file=sys.stderr)
    return FindingsDocument(
        article_file=article_path,
        article_summary=article_summary,
        findings=[],
    )
