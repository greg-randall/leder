"""Stage C: Rebuild article with numbered source footnotes.

Primarily mechanical string processing. LLM reconciliation is only invoked
for quotes that fail normalized matching.
"""
from __future__ import annotations

import json
import os
import re

from pipeline.models import Claim, ClaimsDocument, Verdict


def normalize_text(text: str) -> str:
    """Normalize text for matching: collapse whitespace, strip punctuation edges, lowercase."""
    text = re.sub(r'\s+', ' ', text)
    text = text.strip().strip('.,;:!?\'"()-[]{}')
    return text.lower()


def find_quote_position(source_quote: str, article_text: str) -> tuple[int, int] | None:
    """Find the (start, end) position of source_quote in article_text.

    Uses normalized matching via regex with flexible whitespace.
    Returns None if not found or ambiguous (multiple matches).
    """
    n_quote = normalize_text(source_quote)
    if not n_quote:
        return None

    # Split normalized quote on whitespace runs, build a regex that allows
    # flexible whitespace between non-whitespace segments.
    parts = re.split(r'(\s+)', n_quote)
    pattern_parts = []
    for part in parts:
        if re.match(r'^\s+$', part):
            pattern_parts.append(r'\s+')
        else:
            pattern_parts.append(re.escape(part))

    pattern = ''.join(pattern_parts)

    matches = list(re.finditer(pattern, article_text, re.IGNORECASE))
    if not matches:
        return None
    if len(matches) > 1:
        return None  # Ambiguous

    m = matches[0]
    return (m.start(), m.end())


def insert_footnote_markers(article_text: str, placed_claims: list[tuple[str, tuple[int, int]]]) -> str:
    """Insert [^N] markers into article_text at claim positions.

    Processes claims in REVERSE position order so that earlier insertions
    do not shift the positions of later insertions.
    """
    sorted_claims = sorted(placed_claims, key=lambda x: x[1][0])
    result = article_text

    for i, (claim_id, (start, end)) in enumerate(reversed(sorted_claims)):
        n = len(placed_claims) - i
        marker = f"[^{n}]"
        result = result[:end] + marker + result[end:]

    return result


def build_footnote_block(claims: list[Claim]) -> str:
    """Build the ## Sources footnote block from verified claims."""
    lines = ["\n\n---\n\n## Sources\n"]

    for i, claim in enumerate(claims):
        n = i + 1

        # Verdict badge
        if claim.verdict == Verdict.SUPPORTED:
            vb = "✓ Supported"
        elif claim.verdict == Verdict.CONTRADICTED:
            vb = "✗ Contradicted"
        else:
            vb = "? Unsupported"

        # Proximity badge
        prox = claim.source_proximity.value if claim.source_proximity else "unverifiable"
        pb = f"[{prox}]".capitalize()

        # Source reference
        if claim.source_path:
            source_ref = f"`{claim.source_path}`"
        elif claim.source_url:
            source_ref = claim.source_url
        else:
            source_ref = "none"

        # Flags
        flags = ""
        if claim.human_review:
            flags += " ⚠️ HUMAN REVIEW"
        if claim.reconciled:
            flags += " 🔧 RECONCILED"

        line = (
            f"[^{n}]: **[{vb}]** {pb} "
            f"{claim.claim_text} — {claim.rationale or 'No rationale provided.'}"
            f"{flags}\n"
            f"    Source: {source_ref}\n"
        )
        lines.append(line)

    return "\n".join(lines)


def reconcile_unmatched_quotes(
    unmatched: list[Claim],
    article_text: str,
    model: str = "claude-sonnet-5",
) -> list[Claim]:
    """Send unmatched claims to LLM for quote reconciliation."""
    if not unmatched:
        return unmatched

    import anthropic
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("DEEPSEEK_API_KEY")
    )

    reconciled = []
    for claim in unmatched:
        prompt = f"""You are fixing a broken quote match. A "source_quote" was supposed to be a
verbatim excerpt from the article below, but it doesn't match any substring of
the article -- even after whitespace normalization.

Your job: find the sentence(s) in the article that most closely correspond to
this quote. Output the best-match substring. It will be matched after whitespace
normalization -- word-for-word precision is ideal but not required. If the best
match is a paraphrase rather than an exact quote, that's acceptable -- the
footnote will be flagged as "reconciled" so the human reviewer knows to check
the placement.

If no sentence in the article corresponds to this quote at all, output NO_MATCH.

Failing quote: {claim.source_quote}
Article:
---
{article_text}
---

Respond with ONLY a JSON object:
{{"corrected_quote": "<best-match substring from article>" | null, "status": "corrected" | "no_match"}}"""

        try:
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
                thinking={"type": "disabled"},
            )
            # Extract text from response, skipping non-text blocks
            text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    text += block.text
            json_match = re.search(r'\{[^}]+\}', text)
            if json_match:
                result = json.loads(json_match.group())
                if result.get("status") == "corrected" and result.get("corrected_quote"):
                    claim.source_quote = result["corrected_quote"]
                    claim.reconciled = True
                    reconciled.append(claim)
                    continue
        except Exception as e:
            print(f"  Reconciliation failed for {claim.claim_id}: {e}")

        reconciled.append(claim)

    return reconciled


def build_unplaced_warning(unplaced: list[Claim]) -> str:
    """Build the UNPLACED CLAIMS warning block."""
    lines = [
        "# UNPLACED CLAIMS\n",
        "These claims were verified but could not be located in the article text. "
        "The source_quote may have been paraphrased or the claim may not exist "
        "in the article. **MANUAL REVIEW REQUIRED.**\n",
    ]
    for claim in unplaced:
        lines.append(f"- **{claim.claim_id}**: {claim.claim_text}")
        lines.append(
            f"  Verdict: {claim.verdict.value if claim.verdict else 'unknown'}"
            f" | Source: {claim.source_path or claim.source_url or 'none'}"
        )
        lines.append(f"  Rationale: {claim.rationale}")
        lines.append(f"  Failing quote: \"{claim.source_quote}\"")
        lines.append("")
    return "\n".join(lines)


def run_stage_c(
    article_path: str,
    claims_path: str,
    output_path: str,
    model: str = "claude-sonnet-5",
) -> str:
    """Rebuild article with source footnotes. Returns output path."""
    with open(article_path, encoding="utf-8") as f:
        article_text = f.read()

    with open(claims_path, encoding="utf-8") as f:
        doc = ClaimsDocument.from_json(f.read())

    claims = doc.claims
    placed = []
    unmatched = []

    # First pass: mechanical matching
    for claim in claims:
        pos = find_quote_position(claim.source_quote, article_text)
        if pos:
            placed.append((claim.claim_id, pos))
        else:
            unmatched.append(claim)

    print(f"Mechanical match: {len(placed)}/{len(claims)} placed, {len(unmatched)} unmatched")

    # Second pass: LLM reconciliation
    if unmatched:
        print(f"Reconciling {len(unmatched)} unmatched quotes via LLM...")
        reconciled_claims = reconcile_unmatched_quotes(unmatched, article_text, model)
        still_unmatched = []
        for claim in reconciled_claims:
            pos = find_quote_position(claim.source_quote, article_text)
            if pos:
                placed.append((claim.claim_id, pos))
            else:
                still_unmatched.append(claim)
        print(
            f"  Reconciliation: {len(reconciled_claims) - len(still_unmatched)} recovered,"
            f" {len(still_unmatched)} still unplaced"
        )
    else:
        still_unmatched = []

    # Build output
    output_parts = []

    if still_unmatched:
        output_parts.append(build_unplaced_warning(still_unmatched))
        output_parts.append("\n---\n\n")

    if placed:
        article_with_markers = insert_footnote_markers(article_text, placed)
    else:
        article_with_markers = article_text

    output_parts.append(article_with_markers)

    # Footnote block in appearance order
    sorted_placed = sorted(placed, key=lambda x: x[1][0])
    placed_ids = [cid for cid, _ in sorted_placed]
    claim_map = {c.claim_id: c for c in claims}
    ordered_claims = [claim_map[cid] for cid in placed_ids]

    output_parts.append(build_footnote_block(ordered_claims))

    result = "\n".join(output_parts)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result)

    print(f"Output written -> {output_path}")
    if still_unmatched:
        print(f"  {len(still_unmatched)} UNPLACED CLAIMS -- see top of output file")

    return output_path
