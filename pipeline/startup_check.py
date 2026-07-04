"""Validate that all required external tools are available before pipeline runs."""
from __future__ import annotations

import os
import sys
import subprocess
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CheckResult:
    name: str
    passed: bool
    fatal: bool
    detail: str = ""
    install_hint: str = ""


def check_ripgrep() -> CheckResult:
    """Check that ripgrep is installed. FATAL — cannot search corpus without it."""
    path = shutil.which("rg")
    if path:
        try:
            result = subprocess.run(["rg", "--version"], capture_output=True, text=True, timeout=5)
            version = result.stdout.splitlines()[0] if result.stdout else "unknown"
            return CheckResult(name="ripgrep (rg)", passed=True, fatal=True, detail=version)
        except Exception:
            pass
    return CheckResult(
        name="ripgrep (rg)",
        passed=False,
        fatal=True,
        detail="NOT FOUND",
        install_hint="Install: sudo apt install ripgrep  or  brew install ripgrep  or  cargo install ripgrep",
    )


def check_jina() -> CheckResult:
    """Check that Jina AI is reachable."""
    try:
        import httpx
        r = httpx.get("https://r.jina.ai", timeout=10)
        return CheckResult(name="jina (r.jina.ai)", passed=r.status_code < 500, fatal=False,
                           detail="reachable" if r.status_code < 500 else f"status {r.status_code}")
    except Exception as e:
        hint = "Install: pip install httpx" if "No module named" in str(e) else "Check network connectivity"
        return CheckResult(name="jina (r.jina.ai)", passed=False, fatal=False,
                           detail=f"unreachable: {e}", install_hint=hint)


def check_obscura() -> CheckResult:
    """Check that obscura is installed."""
    path = shutil.which("obscura")
    if path:
        try:
            result = subprocess.run(["obscura", "--version"], capture_output=True, text=True, timeout=10)
            version = result.stdout.strip() or result.stderr.strip() or "installed"
            return CheckResult(name="obscura", passed=True, fatal=False, detail=version)
        except Exception:
            pass
    return CheckResult(
        name="obscura",
        passed=False,
        fatal=False,
        detail="NOT FOUND",
        install_hint="Install: https://github.com/h4ckf0r0day/obscura",
    )


def check_trafilatura() -> CheckResult:
    """Check that trafilatura Python package is importable."""
    try:
        import trafilatura
        version = getattr(trafilatura, "__version__", "installed")
        return CheckResult(name="trafilatura", passed=True, fatal=False, detail=version)
    except ImportError:
        return CheckResult(
            name="trafilatura",
            passed=False,
            fatal=False,
            detail="NOT FOUND",
            install_hint="Install: pip install trafilatura",
        )


def check_scrape_skill() -> CheckResult:
    """Check that scrape-url-skill is available.

    This is determined at runtime by the harness — here we note it as informational.
    """
    return CheckResult(name="scrape-url-skill", passed=True, fatal=False,
                       detail="available (invoked at runtime via Skill tool)")


# ── Non-fatal conversion-tool checks ──────────────────────────────────────

def _check_tool(cmd: str, hint: str) -> CheckResult:
    path = shutil.which(cmd)
    if path:
        return CheckResult(name=cmd, passed=True, fatal=False, detail="installed")
    return CheckResult(name=cmd, passed=False, fatal=False, detail="NOT FOUND",
                       install_hint=hint)


def check_tesseract() -> CheckResult:
    return _check_tool("tesseract", "Install: sudo apt install tesseract-ocr")


def check_pdftoppm() -> CheckResult:
    return _check_tool("pdftoppm", "Install: sudo apt install poppler-utils")


def check_pdftotext() -> CheckResult:
    return _check_tool("pdftotext", "Install: sudo apt install poppler-utils")


def check_pandoc() -> CheckResult:
    return _check_tool("pandoc", "Install: sudo apt install pandoc")


def check_libreoffice() -> CheckResult:
    return _check_tool("libreoffice", "Install: sudo apt install libreoffice")


def check_antiword() -> CheckResult:
    return _check_tool("antiword", "Install: sudo apt install antiword")


def check_faster_whisper() -> CheckResult:
    """Check faster-whisper (local audio transcription) and note GPU vs CPU."""
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return CheckResult(name="faster-whisper", passed=False, fatal=False,
                           detail="NOT FOUND (audio files will be skipped)",
                           install_hint="Install: pip install faster-whisper")
    try:
        import ctranslate2
        gpu = ctranslate2.get_cuda_device_count() > 0
    except Exception:
        gpu = False
    return CheckResult(name="faster-whisper", passed=True, fatal=False,
                       detail=f"installed ({'CUDA GPU' if gpu else 'CPU only — slow for larger models'})")


# ── Corpus / vision-key warnings (non-fatal) ─────────────────────────────

def warn_stale_corpus(corpus_root: str) -> list[str]:
    """Return warning strings if prep left report files behind."""
    warns = []
    root = Path(corpus_root)
    for fname, what in (("UNCONVERTED.md", "unconvertible files"),
                        ("NEEDS_REVIEW.md", "fallback-converted files to verify")):
        if (root / fname).exists():
            warns.append(f"  {fname} exists at corpus root — {what}. See {root / fname}")
    return warns


def warn_missing_vision_key(vision_enabled: bool) -> str:
    """Warning if vision is on but OPENAI_API_KEY is unset."""
    if vision_enabled and not os.environ.get("OPENAI_API_KEY"):
        return ("  prepare.vision_fallback is enabled but OPENAI_API_KEY "
                "is not set — pip install markitdown[all] and set OPENAI_API_KEY "
                "for image/PDF vision fallback.")
    return ""


def warn_missing_audio_deps(audio_enabled: bool) -> str:
    """Warning if audio transcription is on but faster-whisper is not installed."""
    if not audio_enabled:
        return ""
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return ("  prepare.audio is enabled but faster-whisper is not installed — "
                "audio files will be skipped. Install: pip install faster-whisper")
    return ""


_ALL_CHECKS = [
    check_ripgrep, check_jina, check_obscura, check_trafilatura, check_scrape_skill,
    check_tesseract, check_pdftoppm, check_pdftotext, check_pandoc,
    check_libreoffice, check_antiword, check_faster_whisper,
]


def run_startup_checks() -> list[CheckResult]:
    """Run all prerequisite checks. Returns list of results."""
    results = []
    for check_fn in _ALL_CHECKS:
        results.append(check_fn())
    return results


def validate_startup() -> bool:
    """Run checks and print results. Returns True if all FATAL checks passed."""
    print("Checking prerequisites...\n")
    results = run_startup_checks()
    fatal_failures = []
    all_failures = []

    for r in results:
        status = "✓" if r.passed else "✗"
        marker = " [FATAL]" if r.fatal else ""
        print(f"  {r.name:30s} {status} {r.detail}{marker}")
        if not r.passed:
            all_failures.append(r)
            if r.fatal:
                fatal_failures.append(r)
            if r.install_hint:
                print(f"    → {r.install_hint}")

    print()
    if fatal_failures:
        print(f"{len(fatal_failures)} fatal tool(s) missing. Install them and re-run.")
        return False
    elif all_failures:
        print(f"{len(all_failures)} non-fatal tool(s) missing. Web fetch will skip unavailable tiers.")
    else:
        print("All checks passed.")
    return True


if __name__ == "__main__":
    sys.exit(0 if validate_startup() else 1)
