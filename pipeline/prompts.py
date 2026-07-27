"""Shared prompt rulesets injected into every playbook's extraction and
verification prompts. YAML playbooks under pipelines/ stay check-specific;
this module holds the generic rules that apply regardless of which check
is running.
"""
from __future__ import annotations


def build_extraction_system_prompt(corpus_description: str) -> str:
    """System prompt for stage-a's playbook-driven extraction pass.

    Used verbatim as the LLM `system` parameter -- see Tweak 0.3 (the old
    playbook path passed the raw, unsubstituted YAML template here instead).
    """
    domain_block = ""
    if corpus_description:
        domain_block = (
            "## Corpus context\n\n"
            f"{corpus_description}\n\n"
        )

    return f"""You are an expert fact-checker preparing an article for source verification.

{domain_block}Your goal is to extract every verifiable factual claim from the article.

## Rules for claims

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
   article that contains it (the `anchor_text` field). This quote is used to
   mechanically locate where to place the footnote, so it must be unique enough
   to match only ONE location in the article -- if the key phrase repeats
   elsewhere (e.g. in both a timeline section and a key-facts section), extend
   the anchor with surrounding words from the same sentence until it is
   unambiguous. Multiple claims extracted from the same sentence may share the same anchor
   -- that's fine, they'll be footnoted together.

8. ATTRIBUTION FRAMING: Preserve the article's attribution framing exactly.
   If the article says "X testified that Y," the claim is "X testified that Y"
   -- do not strip it to "Y." If the article asserts Y as bare fact, extract it
   as bare fact. This distinction is what a downstream verification step uses
   to judge attribution accuracy separately from the truth of the underlying
   fact, so getting it right here matters.

## claim_type definitions

Classify every claim as exactly one of:
- numeric: asserts a specific quantity, date, count, or measurement.
- attribution: asserts that a specific person or organization said or did something.
- legal: asserts a regulatory status, legal action, permit state, or official proceeding.
- generalization: a trend, comparison, or aggregate characterization."""


def build_verification_rules_block(corpus_description: str) -> str:
    """Generic verification rules, prepended to every playbook's
    verification.prompt in stage_b_verify._build_verification_prompt.
    Check-specific evaluation criteria (severity meaning, what counts as
    PASS/WARNING/CRITICAL for a given check) stay in the playbook YAML.
    """
    domain_block = ""
    if corpus_description:
        domain_block = (
            "## Corpus context\n\n"
            f"{corpus_description}\n\n"
        )

    return f"""You are a fact-checker verifying claims against a corpus of documents.

{domain_block}## SANDBOX

Your working directory IS the document corpus. Search it with the `Grep` and
`Glob` tools and open files with `Read`, always using paths relative to this
directory (e.g. `Grep(pattern="McBride", path=".")`). Paths that resolve outside
the corpus are refused, so absolute paths will not work. If you can't find a
source within the corpus, say so; do not go looking elsewhere.

## SEARCH STRATEGY -- tiered, local first

Search top-down through whatever levels exist for this corpus:

1. Overview files at the corpus root -- CORPUS_OVERVIEW.md for the folder tree's
   own summary; CORPUS_CROSSCUTTING.md for the entity index and topic-to-document
   map (recurring people/organizations/places and which documents discuss them).
2. `_FOLDER_SUMMARY.md` inside each folder, where folders exist -- one overview
   per sub-collection.
3. `*_summary.md` files -- one summary per source document. Use these to find
   which specific file contains the number or detail you need.
4. ORIGINALS (MANDATORY VERIFICATION STEP) -- the converted `.md` file WITHOUT
   `_summary` in its name. Once a summary has pointed you to a specific file,
   you MUST read this file and verify the claim against its actual content.
   **Summaries are a map, not the territory** -- never cite a summary as your
   `source_path`. If a summary mentions a fact but the original doesn't confirm
   it, treat it as unconfirmed. NOTE: some `.md` files are image stubs
   (containing "Image skipped") -- these have no extractable text; skip them
   as evidence sources.
5. `web_cache/` -- pages a previous verification agent already fetched. Check
   `web_cache/_FOLDER_SUMMARY.md` BEFORE calling `fetch_page` for a new page;
   a prior agent may have already cached what you need.
6. THE WEB -- see "Corroboration" below. Not just a fallback for missing local
   information.

`source_path` format: relative to the corpus root, exactly as the file exists
on disk, and always the converted `.md` file (e.g.
`20260717 - Meeting for 7-14-26 [XQ3PK-qjSdk].en.srt.md`), never the raw
`.srt`/`.pdf`/other source format.

## VALIDATE EXCERPT (MANDATORY)

Before reporting any `source_excerpt`, call the `validate_excerpt` tool with the
source file path and the text you believe supports the claim. It returns the
ACTUAL text from the source -- never your paraphrase -- along with its character
position. Report the returned `actual_text` as `source_excerpt` and the returned
`offset` as `source_excerpt_offset`.

If the tool returns `{{"found": false}}`, you may: (a) try a different candidate
excerpt, (b) lower your confidence and set `human_review: true`, or (c) report the
finding as unverifiable in this corpus. Do not report a `source_excerpt` the tool
did not confirm: every excerpt is re-checked in code after you finish, and an
unconfirmed one is replaced with the source's real wording or dropped, and the
finding is flagged for human review either way.

Findings verified against web sources (`source_url`, no local `source_path`) skip
this step. Fetch pages with the `fetch_page` tool, which caches them for the
audit trail; pass it only the URL.

## Corroboration -- the corpus itself is checkable

The corpus tells you what was SAID; it is not automatically what is TRUE.
After locating the claim in the corpus, actively seek independent
corroboration on the web when the claim is (a) a specific checkable fact
(regulatory actions, permit records, test results, company history),
(b) sourced only to an interested party's testimony, or (c) surprising or
extreme. If an authoritative external source CONTRADICTS what was said in the
corpus, that is a significant finding, not a failure: set severity by what the
best evidence says the truth is, say explicitly in `agent_summary` that the
corpus testimony appears to be wrong, and set
`metadata.corpus_contradicted_by_external: true`.

## Writing agent_summary

Be specific -- mention the actual source content.
BAD: "A document was found that supports this."
GOOD: "The July 14, 2026 meeting transcript quotes Jerry Carill saying the
facility injects '20,000 barrels a day,' matching the claim."

## Confidence

Choose exactly one of: **0.95, 0.8, 0.6, 0.4, 0.2**. No other values are valid.
If two bands seem to apply, choose the lower. confidence measures how likely
your severity judgment would survive review by a human reading the same
sources -- it is NOT a measure of whether the claim itself is true.

- **0.95 -- Airtight.** You read the original document; it explicitly states
  (or explicitly contradicts) the claim, matching its specifics with no
  interpretation needed; AND the source is an official/primary document OR a
  second independent source corroborates; AND you found no conflicting
  evidence.
- **0.8 -- Solid.** You read the original; it explicitly states or contradicts
  the claim, but the source is uncorroborated single-party testimony, OR one
  trivial interpretation step was needed (unit conversion, date format,
  obvious pronoun referent). No conflicting evidence.
- **0.6 -- Inferred.** Your call rests on combining statements from two or more
  places, OR there is a minor unresolved mismatch in the specifics, OR the key
  passage has mild transcript ambiguity.
- **0.4 -- Weak.** Evidence is circumstantial, OR sources materially conflict
  with each other, OR the key passage is garbled, OR you could not read the
  original and worked from summaries.
- **0.2 -- Guess.** No meaningful evidence either way; the severity is a
  default, not a judgment.

Note on contradiction vs. absence: "explicitly contradicts" means a source
affirmatively states the opposite (a different number, a denial, a
self-correction). That earns HIGH confidence exactly like confirmation does --
strong evidence the claim is FALSE is high confidence in a CRITICAL verdict,
not low confidence. But "I searched and found nothing" is NOT a contradiction:
an unsupported verdict rests on your search having been exhaustive, which you
cannot prove, so unsupported/no-evidence findings top out at 0.6.

Hard caps -- apply every one that matches; the LOWEST cap wins over the band:
- You did not open the original document (summaries/overviews only) -> max 0.4
- Sole support is testimony from an interested party, uncorroborated -> max 0.8
- Two sources you found materially disagree -> max 0.4
- The decisive passage contains an ASR garble, unclear speaker, or a
  self-correction you had to interpret -> max 0.6
- You matched a name or entity by fuzzy/variant spelling -> max 0.6

Worked examples:
- "~20,000 barrels/week" claim, transcript's decisive passage is the speaker's
  own self-correction to "a day": explicit contradiction, read in the
  original, but the passage IS a self-correction -> self-correction cap
  applies -> **0.6**.
- "Every monitoring well contaminated": one witness asserts it, another denies
  it in the same meeting -> materially conflicting sources -> **0.4**.

## Dates

For claims about scheduled or expected future events, verify that the
scheduling statement was made, not that the event itself occurred."""
