# tests/test_startup_check.py
import pytest
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
