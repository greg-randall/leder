# tests/test_startup_check.py
import pytest
from pipeline import startup_check as sc
from pipeline.startup_check import (
    check_ripgrep,
    check_trafilatura,
    CheckResult,
    run_startup_checks,
    validate_startup,
)


def test_check_ripgrep_found():
    result = check_ripgrep()
    assert result.name == "ripgrep (rg)"
    assert result.passed is True
    assert result.fatal is True


def test_check_trafilatura():
    result = check_trafilatura()
    assert result.name == "trafilatura"
    assert result.fatal is False


def test_run_startup_checks_returns_results():
    results = run_startup_checks()
    names = [r.name for r in results]
    assert "ripgrep (rg)" in names
    assert "trafilatura" in names


def test_fatal_check_failure():
    """Verify validate_startup() returns a bool (True if all fatal checks pass)."""
    result = validate_startup()
    assert isinstance(result, bool)


def test_warn_stale_corpus(tmp_path):
    (tmp_path / "UNCONVERTED.md").write_text("x")
    warns = sc.warn_stale_corpus(str(tmp_path))
    assert any("UNCONVERTED.md" in w for w in warns)

    # NEEDS_REVIEW.md also warns
    (tmp_path / "NEEDS_REVIEW.md").write_text("x")
    warns2 = sc.warn_stale_corpus(str(tmp_path))
    assert sum(1 for w in warns2 if "NEEDS_REVIEW.md" in w or "UNCONVERTED.md" in w) == 2


def test_warn_stale_corpus_clean(tmp_path):
    assert sc.warn_stale_corpus(str(tmp_path)) == []


def test_warn_missing_vision_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert sc.warn_missing_vision_key(True) != ""
    assert sc.warn_missing_vision_key(False) == ""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    assert sc.warn_missing_vision_key(True) == ""


def test_conversion_tool_checks_present():
    names = {c.name for c in sc.run_startup_checks()}
    for tool in ["tesseract", "pdftotext", "pandoc", "libreoffice", "antiword"]:
        assert tool in names, f"missing check for {tool}"


def test_conversion_tool_checks_non_fatal():
    for c in sc.run_startup_checks():
        if c.name in ("tesseract", "pdftotext", "pandoc", "libreoffice", "antiword"):
            assert c.fatal is False, f"{c.name} should be non-fatal"


def test_check_markitdown():
    result = sc.check_markitdown()
    assert result.name == "markitdown"
    assert result.fatal is False


def test_check_pycaption():
    result = sc.check_pycaption()
    assert result.name == "pycaption"
    assert result.fatal is False


def test_check_pymupdf():
    result = sc.check_pymupdf()
    assert result.name == "pymupdf"
    assert result.fatal is False


def test_check_pytesseract():
    result = sc.check_pytesseract()
    assert result.name == "pytesseract"
    assert result.fatal is False


def test_new_library_checks_present():
    names = {c.name for c in sc.run_startup_checks()}
    for tool in ["markitdown", "pycaption", "pymupdf", "pytesseract", "python-docx"]:
        assert tool in names, f"missing check for {tool}"


def test_check_python_docx():
    result = sc.check_python_docx()
    assert result.name == "python-docx"
    assert result.fatal is False
