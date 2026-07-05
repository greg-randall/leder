#!/usr/bin/env python3
"""Fetch a web page and save the raw content verbatim to web_cache/<target_id>/page.md.

Usage: python3 pipeline/tools/fetch_page.py <url> <target_id>

Called by Stage B agents during verification. Two tiers:
  1. jina.ai (r.jina.ai) — free, fast, clean markdown
  2. playwright — real headless browser for JS/paywall pages

Saves the RAW content (no summarization) to web_cache/<target_id>/page.md
and prints it to stdout so the agent can read it directly.
"""
from __future__ import annotations

import sys
from pathlib import Path


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


def _try_playwright(url: str, timeout: int = 30) -> tuple[str | None, str]:
    """Fetch via Playwright headless Chromium."""
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(url, timeout=timeout * 1000, wait_until="domcontentloaded")
            text = page.inner_text("body")
            browser.close()
            if text and len(text) > 100:
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

    content = None
    method = None
    for fetcher in (_try_jina, _try_playwright):
        content, method = fetcher(url)
        if content:
            break

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
