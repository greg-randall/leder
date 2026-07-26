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
