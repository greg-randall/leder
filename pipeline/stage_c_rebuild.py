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


def _letters_only(text: str) -> str:
    """Strip everything except letters and spaces, lowercase, collapse whitespace."""
    result = re.sub(r'[^a-z\n ]', '', text.lower()).replace('\n', ' ')
    return re.sub(r' +', ' ', result).strip()


def find_quote_position(source_quote: str, article_text: str) -> tuple[int, int] | None:
    """Find the (start, end) position of source_quote in article_text.

    Uses sliding-window Levenshtein: strips to letters-only, lowercases,
    then slides the quote across the article and picks the best match.
    """
    needle = _letters_only(source_quote).split()
    if not needle:
        return None

    haystack = _letters_only(article_text)
    haystack_words = haystack.split()
    needle_len = len(needle)
    haystack_len = len(haystack_words)

    if needle_len > haystack_len:
        return None

    # Try exact subsequence first (fast path)
    needle_str = ' '.join(needle)
    idx = haystack.find(needle_str)
    if idx != -1:
        # Found exact match — find position in original article
        return _letters_pos_to_original(idx, len(needle_str), article_text)

    # Sliding window Levenshtein — compare word sequences
    import difflib
    best_ratio = 0.0
    best_start = None
    best_len = 0

    # Window size: needle_len ± 50% (handles added/removed words)
    for window in range(max(3, needle_len // 2), min(haystack_len, needle_len * 2) + 1):
        for start in range(haystack_len - window + 1):
            window_str = ' '.join(haystack_words[start:start + window])
            ratio = difflib.SequenceMatcher(None, needle_str, window_str).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_start = start
                best_len = window

    # Require at least 60% similarity
    if best_ratio < 0.6 or best_start is None:
        return None

    # Map back to original article position
    window_str = ' '.join(haystack_words[best_start:best_start + best_len])
    char_start = haystack.find(window_str)
    if char_start == -1:
        return None
    return _letters_pos_to_original(char_start, len(window_str), article_text)


def _letters_pos_to_original(l_start: int, l_len: int, original: str) -> tuple[int, int]:
    """Map a position in letters-only text back to the original article.

    Walks the original and the letters-only text in parallel, building
    a mapping from letters-only positions to original positions.
    Collapses multiple spaces to match _letters_only.
    """
    letters_text = _letters_only(original)
    mapping = []
    li = 0  # position in letters_text
    prev_was_space = False

    for oi, ch in enumerate(original):
        if li >= len(letters_text):
            break
        lch = letters_text[li]
        if ch.isalpha():
            if ch.lower() == lch:
                mapping.append(oi)
                li += 1
                prev_was_space = False
        elif ch == ' ' or ch == '\n':
            if lch == ' ' and not prev_was_space:
                mapping.append(oi)
                li += 1
                prev_was_space = True
        # Other chars (punctuation, digits) are skipped in letters_text

    orig_start = mapping[l_start] if l_start < len(mapping) else 0
    end_idx = l_start + l_len
    orig_end = mapping[end_idx - 1] + 1 if end_idx <= len(mapping) and end_idx > 0 else len(original)

    return (orig_start, orig_end)


def insert_footnote_markers(article_text: str, placed_claims: list[tuple[str, tuple[int, int]]]) -> str:
    """Insert [^N] markers at claim positions, snapped to word boundaries.

    Processes claims in REVERSE position order so that earlier insertions
    do not shift the positions of later insertions.
    """
    sorted_claims = sorted(placed_claims, key=lambda x: x[1][0])
    result = article_text

    for i, (claim_id, (start, end)) in enumerate(reversed(sorted_claims)):
        n = len(placed_claims) - i
        marker = f"[^{n}]"
        # Snap to word boundary: if end is mid-word, advance to next space/punctuation
        insert_at = end
        while insert_at < len(result) and result[insert_at].isalpha():
            insert_at += 1
        result = result[:insert_at] + marker + result[insert_at:]

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

        excerpt = ""
        if claim.source_excerpt:
            excerpt = f"\n    Matched: \"{claim.source_excerpt}\""

        line = (
            f"[^{n}]: **[{vb}]** {pb} "
            f"{claim.claim_text} — {claim.rationale or 'No rationale provided.'}"
            f"{excerpt}"
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

    # Deduplicate by claim_text: agree → median length, disagree → one per verdict
    from collections import defaultdict
    groups: dict[str, list[Claim]] = defaultdict(list)
    for c in claims:
        if c.verdict is not None:
            groups[c.claim_text].append(c)
    deduped = []
    for text, dupes in groups.items():
        by_verdict: dict[str, list[Claim]] = defaultdict(list)
        for c in dupes:
            by_verdict[c.verdict or "unsupported"].append(c)
        for vclaims in by_verdict.values():
            vclaims.sort(key=lambda c: len(c.rationale or ""))
            deduped.append(vclaims[len(vclaims) // 2])
    seen_texts = set(groups.keys())
    for c in claims:
        if c.verdict is None and c.claim_text not in seen_texts:
            deduped.append(c)
            seen_texts.add(c.claim_text)
    for i, c in enumerate(deduped):
        c.claim_id = f"c{i+1:04d}"
    claims = deduped

    placed = []
    unmatched = []

    # First pass: mechanical matching
    from tqdm import tqdm
    for claim in tqdm(claims, desc="  Matching", unit="claim"):
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
        for claim in tqdm(reconciled_claims, desc="  Recheck", unit="claim"):
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
