#!/usr/bin/env python3
"""Fetch a web page and save the raw content verbatim to web_cache/<target_id>/page.md.

Usage: python3 pipeline/tools/fetch_page.py <url> <target_id>

Called by Stage B agents during verification. Four tiers:
  1. jina.ai (r.jina.ai) — free, fast, clean markdown
  2. obscura — headless browser, handles bot-protected / JS pages
  3. playwright — real headless browser, last resort for JS-heavy pages
  4. archive.is — paywall bypass via snapshot (slow, rate-limited)

Saves the RAW content (no summarization) to web_cache/<target_id>/page.md
and prints it to stdout so the agent can read it directly.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse, urlunparse


def _try_jina(url: str, timeout: int = 30) -> tuple[str | None, str]:
    """Fetch via jina.ai markdown proxy."""
    try:
        import httpx
        r = httpx.get(f"https://r.jina.ai/{url}", timeout=timeout, follow_redirects=True)
        if r.status_code == 200 and len(r.text) > 500 and "CAPTCHA" not in r.text:
            return r.text, "jina.ai"
    except Exception as e:
        print(f"  jina.ai: {type(e).__name__}: {e}", file=sys.stderr)
    return None, "jina.ai"


def _try_obscura(url: str, timeout: int = 30) -> tuple[str | None, str]:
    """Fetch via obscura headless browser, extract markdown with trafilatura."""
    try:
        import trafilatura
        r = subprocess.run(
            ["obscura", "fetch", url, "--dump", "html", "--timeout", str(timeout)],
            capture_output=True, text=True, timeout=timeout + 15,
        )
        if r.returncode == 0 and len(r.stdout) > 200:
            text = trafilatura.extract(r.stdout, output_format="markdown",
                                       include_comments=False)
            if text and len(text) > 200:
                return text, "obscura"
    except Exception as e:
        print(f"  obscura: {type(e).__name__}: {e}", file=sys.stderr)
    return None, "obscura"


def _try_archive_is(url: str, timeout: int = 60) -> tuple[str | None, str]:
    """Fetch via archive.is paywall bypass (Camoufox stealth browser)."""
    try:
        from camoufox.sync_api import Camoufox
        import trafilatura

        # Strip query params / fragments — archive.is indexes by clean URL
        parsed = urlparse(url)
        clean_url = urlunparse(parsed._replace(query="", fragment=""))
        archive_url = f"https://archive.is/newest/{clean_url}"

        with Camoufox(headless=True, geoip=True, humanize=True) as browser:
            page = browser.new_page()

            # Warm up: visit archive.is homepage, wait for the search input to render
            try:
                page.goto("https://archive.is/", wait_until="domcontentloaded",
                          timeout=20_000)
                page.wait_for_selector('input[type="text"]', timeout=10_000)
            except Exception:
                pass  # Cloudflare challenge or network issue — continue anyway

            # Navigate to snapshot, then wait for actual content (not a redirect loop)
            try:
                page.goto(archive_url, wait_until="domcontentloaded",
                          timeout=30_000)
                page.wait_for_load_state("networkidle", timeout=15_000)
            except Exception:
                pass  # Timeout is OK — grab whatever rendered

            final_url = page.url
            html = page.content()

        # If we stayed on /newest/ or bounced to homepage, no snapshot exists
        if "/newest/" in final_url or final_url.rstrip("/") in (
            "https://archive.is", "https://archive.ph",
        ):
            return None, "archive.is"

        text = trafilatura.extract(html, output_format="markdown",
                                   include_comments=False)
        if text and len(text) > 200:
            return text, "archive.is"
    except Exception as e:
        print(f"  archive.is: {type(e).__name__}: {e}", file=sys.stderr)
    return None, "archive.is"


def _try_playwright(url: str, timeout: int = 30) -> tuple[str | None, str]:
    """Fetch via Playwright headless Chromium, extract markdown with trafilatura."""
    try:
        import trafilatura
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(url, timeout=timeout * 1000, wait_until="domcontentloaded")
            html = page.content()
            browser.close()
        text = trafilatura.extract(html, output_format="markdown",
                                   include_comments=False)
        if text and len(text) > 200:
            return text, "playwright"
    except Exception as e:
        print(f"  playwright: {type(e).__name__}: {e}", file=sys.stderr)
    return None, "playwright"


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 pipeline/tools/fetch_page.py <url> <target_id>", file=sys.stderr)
        sys.exit(1)

    url = sys.argv[1]
    target_id = sys.argv[2]

    cache_dir = Path("web_cache") / target_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / "page.md"

    _PAYWALL_SIGNALS = (
        "Supported by", "Subscribe to continue", "Already a subscriber?",
        "Create a free account", "To continue reading", "Please sign in",
        "You've reached your limit", "subscribers only",
    )

    content = None
    method = None

    # Tier 1: jina.ai — fast, free, but fails on paywalls
    jina_result, _ = _try_jina(url)
    if jina_result:
        content, method = jina_result, "jina.ai"
    else:
        # Jina hit a paywall/CAPTCHA — try archive.is for the full article
        content, method = _try_archive_is(url)
        if not content:
            # No archive.is snapshot — fall back to obscura / playwright
            for fetcher in (_try_obscura, _try_playwright):
                content, method = fetcher(url)
                if content:
                    break
        # Quality gate: if the best we got looks like a paywall stub, flag it
        if content and any(signal.lower() in content[:500].lower()
                           for signal in _PAYWALL_SIGNALS):
            print(f"  WARNING: content may be a paywall preview "
                  f"({len(content)} chars). Consider finding an alternate source.",
                  file=sys.stderr)

    if content:
        out_path.write_text(content, encoding="utf-8")
        print(f"[fetch_page] {url} -> {out_path}  ({method})", file=sys.stderr)
        # Print to stdout so the agent can read it via $(python3 ...)
        print(content)
    else:
        out_path.write_text(f"(failed to fetch {url})\n", encoding="utf-8")
        print(f"[fetch_page] FAILED: {url} — all methods exhausted", file=sys.stderr)
        print(f"(failed to fetch {url})")


if __name__ == "__main__":
    main()
