"""Tests for pipeline/prepare_reflow.py."""
import re


def _whitespace_collapsed(s: str) -> str:
    return re.sub(r'\s+', ' ', s)


def test_short_lines_pass_through_unchanged():
    from pipeline.prepare_reflow import reflow_pipeline_text
    text = "Line one.\nLine two.\nLine three."
    assert reflow_pipeline_text(text) == text


def test_long_punctuated_line_wraps_at_sentence_boundaries():
    from pipeline.prepare_reflow import reflow_pipeline_text
    sentence = "This is a reasonably long sentence with real words in it. "
    long_line = (sentence * 10).strip()
    assert len(long_line) > 300
    result = reflow_pipeline_text(long_line)
    assert "\n" in result
    for line in result.split("\n"):
        assert len(line) <= 300


def test_long_unpunctuated_line_wraps_at_word_boundaries():
    from pipeline.prepare_reflow import reflow_pipeline_text
    long_line = " ".join(["word"] * 100)  # 499 chars, zero sentence punctuation
    assert len(long_line) > 300
    result = reflow_pipeline_text(long_line)
    assert "\n" in result
    for line in result.split("\n"):
        assert len(line) <= 120
        assert not line.startswith(" ") and not line.endswith(" ")


def test_whitespace_only_invariant_holds_for_punctuated_text():
    from pipeline.prepare_reflow import reflow_pipeline_text
    text = "First sentence here. " * 20
    result = reflow_pipeline_text(text)
    assert _whitespace_collapsed(text) == _whitespace_collapsed(result)


def test_whitespace_only_invariant_holds_for_unpunctuated_text():
    from pipeline.prepare_reflow import reflow_pipeline_text
    text = " ".join(["caption"] * 80)
    result = reflow_pipeline_text(text)
    assert _whitespace_collapsed(text) == _whitespace_collapsed(result)


def test_whitespace_only_invariant_holds_with_hyphens_numbers_unicode():
    from pipeline.prepare_reflow import reflow_pipeline_text
    text = ("The facility injects 20,000 barrels/day — per testimony from "
            "Jerry Carill (also rendered Carville) at the July 14 meeting. ") * 8
    assert len(text) > 300
    result = reflow_pipeline_text(text)
    assert _whitespace_collapsed(text) == _whitespace_collapsed(result)
    assert "20,000 barrels/day" in result.replace("\n", " ")


def test_never_hyphenates_or_alters_non_whitespace():
    from pipeline.prepare_reflow import reflow_pipeline_text
    text = " ".join(["supercalifragilisticexpialidocious"] * 15)
    result = reflow_pipeline_text(text)
    assert "supercalifragilisticexpialidocious-" not in result
    assert result.replace("\n", " ").split() == text.split()


def test_multiline_input_only_reflows_the_long_lines():
    from pipeline.prepare_reflow import reflow_pipeline_text
    short = "A short line."
    long_line = " ".join(["word"] * 100)
    text = f"{short}\n{long_line}\n{short}"
    result = reflow_pipeline_text(text)
    lines = result.split("\n")
    assert lines[0] == short
    assert lines[-1] == short
    assert len(lines) > 3
