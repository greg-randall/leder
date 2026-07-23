"""Stage C: Rebuild article with numbered source footnotes.

Primarily mechanical string processing. LLM reconciliation is only invoked
for quotes that fail normalized matching.
"""
from __future__ import annotations

import json
import os
import re
import sys
from types import SimpleNamespace

from pipeline.models import Claim, ClaimsDocument, Verdict, SourceProximity, ClaimType


def normalize_text(text: str) -> str:
    """Normalize text for matching: collapse whitespace, strip punctuation edges, lowercase."""
    text = re.sub(r'\s+', ' ', text)
    text = text.strip().strip('.,;:!?\'"()-[]{}')
    return text.lower()


def _letters_only(text: str) -> str:
    """Strip everything except letters and spaces, lowercase, collapse whitespace."""
    result = re.sub(r'[^a-z\n ]', '', text.lower()).replace('\n', ' ')
    return re.sub(r' +', ' ', result).strip()


def find_quote_position(
    source_quote: str, article_text: str, context: str | None = None
) -> tuple[int, int] | None:
    """Find the (start, end) position of source_quote in article_text.

    Uses sliding-window Levenshtein: strips to letters-only, lowercases,
    then slides the quote across the article and picks the best match.

    When the quote text matches more than one position in the article
    (letters-only comparison), ``context`` -- the surrounding paragraph
    the claim was extracted from (Claim.context / Finding.context) -- is
    used to pick the occurrence whose surrounding text best matches it.
    Without context, or when there's only one match, the first occurrence
    is used (unchanged from prior behavior).
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

    needle_str = ' '.join(needle)

    # Try exact subsequence first (fast path) -- collect every occurrence so
    # a duplicate/near-duplicate phrase elsewhere in the article can be
    # disambiguated by context instead of silently taking the first match.
    exact_positions = []
    search_from = 0
    while True:
        idx = haystack.find(needle_str, search_from)
        if idx == -1:
            break
        exact_positions.append(idx)
        search_from = idx + 1

    if exact_positions:
        if len(exact_positions) == 1 or not context:
            idx = exact_positions[0]
        else:
            idx = _best_match_by_context(exact_positions, len(needle_str), haystack, context)
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


def _best_match_by_context(
    positions: list[int], match_len: int, haystack: str, context: str
) -> int:
    """Pick the exact-match position whose surrounding window best matches context.

    Window size scales to the length of ``context`` itself (not a fixed pad),
    so a short context stays localized to its candidate position instead of
    swallowing most of a short article and making every candidate look alike.
    """
    import difflib
    context_letters = _letters_only(context)
    window_size = max(len(context_letters), match_len)
    half_pad = max(0, (window_size - match_len) // 2)

    best_pos = positions[0]
    best_ratio = -1.0
    for pos in positions:
        window_start = max(0, pos - half_pad)
        window_end = min(len(haystack), pos + match_len + half_pad)
        window = haystack[window_start:window_end]
        ratio = difflib.SequenceMatcher(None, context_letters, window).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_pos = pos
    return best_pos


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
    """Build the ## Sources footnote block from verified claims.

    Supports both legacy Claim objects (verdict-based badges) and
    finding-derived objects (check_type + severity badges).
    When multiple items share the same position (merged via
    ``_merged_badges``), all badges are rendered in one footnote.
    """
    lines = ["\n\n---\n\n## Sources\n"]

    for i, claim in enumerate(claims):
        n = i + 1

        # Badge(s) — prioritise merged groups, then single finding, then legacy
        if hasattr(claim, '_merged_badges') and claim._merged_badges:
            vb = " ".join(claim._merged_badges)
        elif hasattr(claim, 'check_type') and claim.check_type:
            sev = claim.severity if hasattr(claim, 'severity') else "WARNING"
            if sev == "PASS":
                vb = f"✓ {claim.check_type}"
            elif sev == "CRITICAL":
                vb = f"✗ {claim.check_type}"
            else:
                vb = f"? {claim.check_type}"
        else:
            # Legacy verdict badges
            if claim.verdict == Verdict.SUPPORTED:
                vb = "✓ Supported"
            elif claim.verdict == Verdict.CONTRADICTED:
                vb = "✗ Contradicted"
            else:
                vb = "? Unsupported"

        # Proximity badge
        if hasattr(claim, 'source_proximity') and claim.source_proximity:
            prox_val = (
                claim.source_proximity.value
                if hasattr(claim.source_proximity, 'value')
                else str(claim.source_proximity)
            )
            pb = f"[{prox_val}]".capitalize()
        else:
            pb = "[Original]"

        # Source reference
        if claim.source_path:
            source_ref = f"`{claim.source_path}`"
        elif claim.source_url:
            source_ref = claim.source_url
        else:
            source_ref = "none"

        # Flags
        flags = ""
        if getattr(claim, 'human_review', False):
            flags += " ⚠️ HUMAN REVIEW"
        if getattr(claim, 'reconciled', False):
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


def _load_findings(findings_path: str) -> list[dict]:
    """Load findings.json and return a list of finding dicts.

    Each dict has the fields the rebuild engine needs, plus
    ``check_type`` and ``severity`` for badge rendering.
    """
    from pipeline.finding import Finding, FindingsDocument

    with open(findings_path, encoding="utf-8") as f:
        data = json.loads(f.read())

    # Handle both FindingsDocument (findings key) and targets.json format
    if "findings" in data:
        doc = FindingsDocument.from_json(json.dumps(data))
    elif "targets" in data:
        findings_list = []
        for t in data["targets"]:
            findings_list.append(Finding(
                target_text=t.get("target_text", ""),
                anchor_text=t.get("anchor_text", ""),
                playbook=t.get("playbook", "fact_check"),
                claim_type=t.get("claim_type", t.get("type", "generalization")),
                verdict=t.get("verdict", "unsupported"),
                source_proximity=t.get("source_proximity", "unverifiable"),
                source_path=t.get("source_path", ""),
                source_url=t.get("source_url", ""),
                rationale=t.get("rationale", ""),
                human_review=t.get("human_review", False),
                confidence=t.get("confidence", 0.0),
            ))
        doc = FindingsDocument(
            article_file=data.get("article_file", ""),
            article_summary=data.get("article_summary", ""),
            findings=findings_list,
        )
    else:
        raise ValueError(
            f"Unknown format in {findings_path}: expected 'findings' or 'targets' key")

    findings = []
    for finding in doc.findings:
        # Map Severity enum → verdict string so the existing pipeline logic
        # (dedup by verdict, placement, etc.) works unchanged.
        if finding.severity.value == "PASS":
            verdict = "supported"
        elif finding.severity.value == "CRITICAL":
            verdict = "contradicted"
        else:
            verdict = "unsupported"

        findings.append({
            "claim_id": finding.finding_id,
            "claim_text": finding.target_text,
            "source_quote": finding.anchor_text,
            "verdict": verdict,
            "check_type": finding.check_type,
            "severity": finding.severity.value,
            "rationale": finding.agent_summary,
            "source_path": finding.source_path,
            "source_excerpt": finding.source_excerpt,
            "recommended_action": finding.recommended_action,
            "human_review": finding.human_review,
        })

    return findings


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


def _finding_dicts_to_claim_like_objects(finding_dicts: list[dict]) -> list:
    """Convert _load_findings dicts to SimpleNamespace objects the pipeline expects."""
    claim_likes = []
    for fd in finding_dicts:
        obj = SimpleNamespace(**fd)
        obj.verdict = Verdict(fd["verdict"])
        obj.source_proximity = SourceProximity.ORIGINAL
        obj.reconciled = False
        obj.source_url = None
        obj.claim_type = ClaimType.GENERALIZATION
        obj.context = ""
        obj.confidence = None
        claim_likes.append(obj)
    return claim_likes


def _merge_placed_groups(
    placed: list[tuple[str, tuple[int, int]]],
    claim_lookup: dict,
) -> tuple[list[tuple[str, tuple[int, int]]], list]:
    """Deduplicate placed claims by position and merge entries with the same position.

    For finding-derived claims that share an anchor position but have
    different ``check_type``/``severity`` values, produces a single
    composite entry with combined badges stored in ``_merged_badges``.
    Returns (deduped_placed, ordered_merged_claims).
    """
    from collections import OrderedDict

    # Group by position, preserving insertion order
    pos_groups: OrderedDict = OrderedDict()
    for cid, pos in placed:
        pos_groups.setdefault(pos, []).append(cid)

    deduped_placed = []
    merged_claims = []

    for pos, cids in pos_groups.items():
        first_cid = cids[0]
        deduped_placed.append((first_cid, pos))

        if len(cids) == 1:
            merged_claims.append(claim_lookup[cids[0]])
        else:
            group = [claim_lookup[cid] for cid in cids]
            primary = group[0]

            # Collect badges from every claim in the group
            badges = []
            for c in group:
                if hasattr(c, 'check_type') and c.check_type:
                    sev = getattr(c, 'severity', "WARNING")
                    if sev == "PASS":
                        badges.append(f"✓ {c.check_type}")
                    elif sev == "CRITICAL":
                        badges.append(f"✗ {c.check_type}")
                    else:
                        badges.append(f"? {c.check_type}")
                else:
                    if c.verdict == Verdict.SUPPORTED:
                        badges.append("✓ Supported")
                    elif c.verdict == Verdict.CONTRADICTED:
                        badges.append("✗ Contradicted")
                    else:
                        badges.append("? Unsupported")

            if badges:
                primary._merged_badges = badges
            merged_claims.append(primary)

    return deduped_placed, merged_claims


def run_stage_c(
    article_path: str,
    claims_path: str | None = None,
    output_path: str | None = None,
    model: str = "claude-sonnet-5",
    findings_path: str | None = None,
) -> str:
    """Rebuild article with source footnotes. Returns output path.

    Accepts either a ``claims_path`` (legacy claims.json) or a
    ``findings_path`` (findings.json from Stage 2).  When both are
    ``None`` or ``output_path`` is ``None`` a ``ValueError`` is raised.
    """
    if output_path is None:
        raise ValueError("output_path is required")

    with open(article_path, encoding="utf-8") as f:
        article_text = f.read()

    # ── Load source data ──────────────────────────────────────────
    if findings_path:
        finding_dicts = _load_findings(findings_path)
        claims = _finding_dicts_to_claim_like_objects(finding_dicts)
    elif claims_path:
        with open(claims_path, encoding="utf-8") as f:
            doc = ClaimsDocument.from_json(f.read())
        claims = doc.claims
    else:
        raise ValueError("Either claims_path or findings_path must be provided")

    # ── Deduplicate ───────────────────────────────────────────────
    # For findings with different check_types, use (claim_text, check_type)
    # as the dedup key so distinct check_types on the same target survive.
    from collections import defaultdict
    groups: dict = defaultdict(list)
    for c in claims:
        if c.verdict is not None:
            if hasattr(c, 'check_type') and c.check_type:
                key = (c.claim_text, c.check_type)
            else:
                key = c.claim_text
            groups[key].append(c)
    deduped = []
    for key, dupes in groups.items():
        by_verdict: dict[str, list] = defaultdict(list)
        for c in dupes:
            by_verdict[c.verdict or "unsupported"].append(c)
        for vclaims in by_verdict.values():
            vclaims.sort(key=lambda c: len(c.rationale or ""))
            deduped.append(vclaims[len(vclaims) // 2])
    seen_texts = set()
    for key in groups:
        if isinstance(key, tuple):
            seen_texts.add(key[0])
        else:
            seen_texts.add(key)
    for c in claims:
        if c.verdict is None and c.claim_text not in seen_texts:
            deduped.append(c)
            seen_texts.add(c.claim_text)
    for i, c in enumerate(deduped):
        c.claim_id = f"c{i+1:04d}"
    dup_removed = len(claims) - len(deduped)
    if dup_removed:
        print(f"Dedup: {len(claims)} → {len(deduped)} ({dup_removed} removed)",
              file=sys.stderr)
    claims = deduped

    # ── Mechanical matching ───────────────────────────────────────
    placed = []
    unmatched = []

    from tqdm import tqdm
    for claim in tqdm(claims, desc="  Matching", unit="claim"):
        pos = find_quote_position(claim.source_quote, article_text, context=getattr(claim, 'context', None))
        if pos:
            placed.append((claim.claim_id, pos))
        else:
            unmatched.append(claim)

    print(f"Mechanical match: {len(placed)}/{len(claims)} placed, {len(unmatched)} unmatched")

    # ── LLM reconciliation ────────────────────────────────────────
    if unmatched:
        print(f"Reconciling {len(unmatched)} unmatched quotes via LLM...")
        reconciled_claims = reconcile_unmatched_quotes(unmatched, article_text, model)
        still_unmatched = []
        for claim in tqdm(reconciled_claims, desc="  Recheck", unit="claim"):
            pos = find_quote_position(claim.source_quote, article_text, context=getattr(claim, 'context', None))
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

    # ── Merge claims sharing the same position ────────────────────
    claim_lookup = {c.claim_id: c for c in claims}
    deduped_placed, merged_claims = _merge_placed_groups(placed, claim_lookup)

    # ── Build output ──────────────────────────────────────────────
    output_parts = []

    if still_unmatched:
        output_parts.append(build_unplaced_warning(still_unmatched))
        output_parts.append("\n---\n\n")

    if deduped_placed:
        article_with_markers = insert_footnote_markers(article_text, deduped_placed)
    else:
        article_with_markers = article_text

    output_parts.append(article_with_markers)
    output_parts.append(build_footnote_block(merged_claims))

    result = "\n".join(output_parts)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result)

    print(f"Output written -> {output_path}")
    if still_unmatched:
        print(f"  {len(still_unmatched)} UNPLACED CLAIMS -- see top of output file")

    return output_path
