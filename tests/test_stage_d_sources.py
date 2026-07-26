"""Tests for pipeline/stage_d_sources.py."""
import html as html_mod


def test_mark_excerpts_no_spans_returns_escaped_text():
    from pipeline.stage_d_sources import mark_excerpts

    text = "Plain text with <a tag> in it & an ampersand."
    result = mark_excerpts(text, [])
    assert result == html_mod.escape(text, quote=False)
    assert "<mark" not in result


def test_mark_excerpts_single_span():
    from pipeline.stage_d_sources import mark_excerpts

    text = "The facility injects 20000 barrels a day into the well."
    start = text.index("injects 20000 barrels a day")
    end = start + len("injects 20000 barrels a day")
    result = mark_excerpts(text, [(start, end, "fn3", "WARNING")])

    assert '<mark id="exc-fn3" data-findings="fn3" data-severity="WARNING">' in result
    assert "injects 20000 barrels a day" in result
    assert result.startswith("The facility ")
    assert result.count("<mark") == 1


def test_mark_excerpts_multiple_non_overlapping_spans():
    from pipeline.stage_d_sources import mark_excerpts

    text = "First claim here. Middle text. Second claim there."
    s1 = text.index("First claim here")
    e1 = s1 + len("First claim here")
    s2 = text.index("Second claim there")
    e2 = s2 + len("Second claim there")
    result = mark_excerpts(text, [
        (s1, e1, "fn1", "PASS"),
        (s2, e2, "fn2", "CRITICAL"),
    ])

    assert 'id="exc-fn1" data-findings="fn1" data-severity="PASS"' in result
    assert 'id="exc-fn2" data-findings="fn2" data-severity="CRITICAL"' in result
    assert result.count("<mark") == 2
    assert "Middle text." in result
    # Both marks' content still present and in original order
    assert result.index("First claim here") < result.index("Second claim there")


def test_mark_excerpts_overlapping_spans_split_into_three_segments():
    from pipeline.stage_d_sources import mark_excerpts

    text = "the quick brown fox jumps over the lazy dog"
    # Span A: "quick brown fox" (fn1), Span B: "brown fox jumps" (fn2) -- overlap on "brown fox"
    s1 = text.index("quick brown fox")
    e1 = s1 + len("quick brown fox")
    s2 = text.index("brown fox jumps")
    e2 = s2 + len("brown fox jumps")
    result = mark_excerpts(text, [
        (s1, e1, "fn1", "WARNING"),
        (s2, e2, "fn2", "CRITICAL"),
    ])

    # Three marked segments: "quick " (fn1 only), "brown fox" (both), " jumps" (fn2 only)
    assert result.count("<mark") == 3
    assert 'data-findings="fn1"' in result
    assert 'data-findings="fn2"' in result
    assert 'data-findings="fn1,fn2"' in result
    # Order preserved
    assert result.index('data-findings="fn1"') < result.index('data-findings="fn1,fn2"')
    assert result.index('data-findings="fn1,fn2"') < result.index('data-findings="fn2"')


def test_mark_excerpts_escapes_html_special_chars_in_and_around_marks():
    from pipeline.stage_d_sources import mark_excerpts

    text = "Cost < $5 & rising, per the <report>."
    start = text.index("<report>")
    end = start + len("<report>")
    result = mark_excerpts(text, [(start, end, "fn1", "PASS")])

    assert "Cost &lt; $5 &amp; rising" in result
    assert "&lt;report&gt;" in result
    assert "<report>" not in result  # never appears unescaped


def _finding(finding_id, source_path=None, source_url=None, source_excerpt="a quote",
             severity="PASS"):
    return {
        "finding_id": finding_id, "severity": severity,
        "source_path": source_path, "source_url": source_url,
        "source_excerpt": source_excerpt,
    }


def test_resolve_cited_sources_corpus_path(tmp_path):
    from pipeline.stage_d_sources import resolve_cited_sources

    corpus = tmp_path / "corpus"
    (corpus / "docs").mkdir(parents=True)
    (corpus / "docs" / "a.md").write_text("Some corpus content.")

    findings = [_finding("fn1", source_path="docs/a.md", source_excerpt="corpus content")]
    resolved = resolve_cited_sources(findings, str(corpus), str(tmp_path / "web_cache"))

    assert "docs/a.md" in resolved
    entry = resolved["docs/a.md"]
    assert entry["kind"] == "corpus"
    assert entry["local_path"] == str(corpus / "docs" / "a.md")
    assert entry["excerpts"] == [("corpus content", "fn1", "PASS")]


def test_resolve_cited_sources_web_cache_snapshot(tmp_path):
    from pipeline.stage_d_sources import resolve_cited_sources

    wc = tmp_path / "web_cache" / "fn2"
    wc.mkdir(parents=True)
    (wc / "page.md").write_text("Real fetched page content.")

    findings = [_finding("fn2", source_url="https://example.com/x",
                         source_excerpt="fetched page content", severity="WARNING")]
    resolved = resolve_cited_sources(findings, str(tmp_path / "corpus"), str(tmp_path / "web_cache"))

    assert "fn2" in resolved
    entry = resolved["fn2"]
    assert entry["kind"] == "web"
    assert entry["local_path"] == str(wc / "page.md")
    assert entry["excerpts"] == [("fetched page content", "fn2", "WARNING")]


def test_resolve_cited_sources_failed_fetch_placeholder_not_resolved(tmp_path):
    from pipeline.stage_d_sources import resolve_cited_sources

    wc = tmp_path / "web_cache" / "fn3"
    wc.mkdir(parents=True)
    (wc / "page.md").write_text("(failed to fetch https://example.com/dead)\n")

    findings = [_finding("fn3", source_url="https://example.com/dead")]
    resolved = resolve_cited_sources(findings, str(tmp_path / "corpus"), str(tmp_path / "web_cache"))

    assert "fn3" not in resolved


def test_resolve_cited_sources_multiple_findings_same_document(tmp_path):
    from pipeline.stage_d_sources import resolve_cited_sources

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("First bit. Second bit.")

    findings = [
        _finding("fn1", source_path="a.md", source_excerpt="First bit", severity="PASS"),
        _finding("fn2", source_path="a.md", source_excerpt="Second bit", severity="CRITICAL"),
    ]
    resolved = resolve_cited_sources(findings, str(corpus), str(tmp_path / "web_cache"))

    assert len(resolved) == 1
    assert resolved["a.md"]["excerpts"] == [
        ("First bit", "fn1", "PASS"),
        ("Second bit", "fn2", "CRITICAL"),
    ]


def test_resolve_cited_sources_skips_findings_with_no_source(tmp_path):
    from pipeline.stage_d_sources import resolve_cited_sources

    findings = [_finding("fn1", source_path=None, source_url=None)]
    resolved = resolve_cited_sources(findings, str(tmp_path / "corpus"), str(tmp_path / "web_cache"))

    assert resolved == {}


def test_resolve_cited_sources_missing_corpus_file_not_resolved(tmp_path):
    from pipeline.stage_d_sources import resolve_cited_sources

    findings = [_finding("fn1", source_path="does/not/exist.md")]
    resolved = resolve_cited_sources(findings, str(tmp_path / "corpus"), str(tmp_path / "web_cache"))

    assert resolved == {}
