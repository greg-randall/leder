#!/usr/bin/env python3
"""Fetch a web page and save the raw content verbatim to web_cache/<target_id>/page.md.

Usage: python3 pipeline/tools/fetch_page.py <url> <target_id> [--debug]

Four tiers (in order):
  1. jina.ai (r.jina.ai) — free, fast, clean markdown
  2. archive.is — paywall bypass via Camoufox snapshot
  3. obscura — headless browser for bot-protected / JS pages
  4. playwright — real headless browser, last resort

With --debug, raw output from EVERY tier is saved to the cache directory:
  jina.md, archive_is.html, archive_is.png, obscura.html, playwright.html
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import httpx
import trafilatura


def _try_jina(url: str, timeout: int = 30, debug_dir: Path | None = None
              ) -> tuple[str | None, str]:
    r = httpx.get(f"https://r.jina.ai/{url}", timeout=timeout, follow_redirects=True)
    if debug_dir:
        (debug_dir / "jina.md").write_text(r.text, encoding="utf-8")
        print(f"  [debug] jina: status={r.status_code}  bytes={len(r.text)}",
              file=sys.stderr)
    if r.status_code == 200 and len(r.text) > 500 and "CAPTCHA" not in r.text:
        return r.text, "jina.ai"
    return None, "jina.ai"


def _try_obscura(url: str, timeout: int = 30, debug_dir: Path | None = None
                 ) -> tuple[str | None, str]:
    r = subprocess.run(
        ["obscura", "fetch", url, "--dump", "html", "--timeout", str(timeout),
         "--wait-until", "networkidle0"],
        capture_output=True, text=True, timeout=timeout + 30,
    )
    if debug_dir:
        (debug_dir / "obscura.html").write_text(r.stdout, encoding="utf-8")
        print(f"  [debug] obscura: exit={r.returncode}  html={len(r.stdout)} bytes"
              f"  stderr={r.stderr[:100]}", file=sys.stderr)
    if r.returncode != 0:
        print(f"  obscura exit {r.returncode}: {r.stderr[:200]}", file=sys.stderr)
        return None, "obscura"
    if len(r.stdout) < 200:
        print(f"  obscura returned {len(r.stdout)} bytes (too short)", file=sys.stderr)
        return None, "obscura"
    text = trafilatura.extract(r.stdout, output_format="markdown", include_comments=False)
    if not text or len(text) < 200:
        print(f"  obscura trafilatura extracted {len(text) if text else 0} bytes "
              f"from {len(r.stdout)} bytes of HTML", file=sys.stderr)
        return None, "obscura"
    return text, "obscura"


def _try_playwright(url: str, timeout: int = 30, debug_dir: Path | None = None
                    ) -> tuple[str | None, str]:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url, timeout=timeout * 1000, wait_until="domcontentloaded")
        html = page.content()
        browser.close()
    if debug_dir:
        (debug_dir / "playwright.html").write_text(html, encoding="utf-8")
        print(f"  [debug] playwright: html={len(html)} bytes", file=sys.stderr)
    text = trafilatura.extract(html, output_format="markdown", include_comments=False)
    if text and len(text) > 200:
        return text, "playwright"
    print(f"  playwright trafilatura extracted {len(text) if text else 0} bytes "
          f"from {len(html)} bytes of HTML", file=sys.stderr)
    return None, "playwright"


def _try_archive_is(url: str, timeout: int = 60, debug_dir: Path | None = None
                    ) -> tuple[str | None, str]:
    import os
    import random
    from pathlib import Path as _Path
    from camoufox.sync_api import Camoufox

    # Browserforge Screen fingerprinting (randomized viewport within range)
    try:
        from browserforge.fingerprints import Screen
        screen = Screen(min_width=1280, max_width=1920, min_height=768, max_height=1080)
    except ImportError:
        screen = None

    parsed = urlparse(url)
    clean_url = urlunparse(parsed._replace(query="", fragment=""))

    # Remove display env so Camoufox monitor detection returns None
    os.environ.pop("DISPLAY", None)
    os.environ.pop("WAYLAND_DISPLAY", None)

    # Persistent context: save Cloudflare cookies to survive across runs
    user_dir = _Path.home() / ".cache" / "leder-camoufox"
    user_dir.mkdir(parents=True, exist_ok=True)

    with Camoufox(
        headless="virtual",          # real display context, not detectable as headless
        humanize=True,
        os="windows",
        screen=screen,
        persistent_context=True,     # save/restore cookies across sessions
        user_data_dir=str(user_dir),
        disable_coop=True, i_know_what_im_doing=True,           # allow clicks into cross-origin iframes (Turnstile)
        geoip=False,
        config={
            "mediaDevices:enabled": True,
            "mediaDevices:micros": 1,
            "mediaDevices:webcams": 1,
            "mediaDevices:speakers": 2,
            "voices:blockIfNotDefined": True,
            "webrtc:localipv4": f"192.168.1.{random.randint(24, 140)}",
        },
    ) as browser:
        page = browser.new_page()
        page.set_viewport_size({"width": 1920, "height": 1080})

        page.goto("https://archive.is/", wait_until="load", timeout=30_000)
        time.sleep(4)

        # If Cloudflare bounced us off archive.is, give up now
        if "archive.is" not in page.url:
            if debug_dir:
                page.screenshot(path=str(debug_dir / "archive_is_cf_block.png"),
                                full_page=True)
            print(f"  archive.is: Cloudflare block (url={page.url[:80]})",
                  file=sys.stderr)
            return None, "archive.is"

        page.goto(f"https://archive.is/newest/{clean_url}", wait_until="load",
                  timeout=30_000)
        time.sleep(3)

        final_url = page.url
        html = page.content()

        if debug_dir:
            (debug_dir / "archive_is.html").write_text(html, encoding="utf-8")
            page.screenshot(path=str(debug_dir / "archive_is.png"), full_page=True)
            print(f"  [debug] archive.is: final_url={final_url[:100]}  html={len(html)} bytes",
                  file=sys.stderr)

    if "/newest/" in final_url or final_url.rstrip("/") in (
        "https://archive.is", "https://archive.ph",
    ):
        print(f"  archive.is: no snapshot ({final_url[:80]})", file=sys.stderr)
        return None, "archive.is"

    # Return raw HTML (matching futureheist). Trafilatura extraction happens
    # downstream — the caller saves page.md from whatever content we return.
    text = trafilatura.extract(html, output_format="markdown", include_comments=False)
    if text and len(text) > 200:
        return text, "archive.is"
    # If trafilatura failed, return the raw HTML anyway — better than nothing
    print(f"  archive.is: trafilatura got {len(text) if text else 0} bytes, "
          f"falling back to raw HTML ({len(html)} bytes)", file=sys.stderr)
    return html, "archive.is-raw"


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 pipeline/tools/fetch_page.py <url> <target_id> [--debug]",
              file=sys.stderr)
        sys.exit(1)

    url = sys.argv[1]
    target_id = sys.argv[2]
    debug = "--debug" in sys.argv

    cache_dir = Path("web_cache") / target_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / "page.md"
    debug_dir = cache_dir if debug else None

    _PAYWALL_SIGNALS = (
        "Supported by", "Subscribe to continue", "Already a subscriber?",
        "Create a free account", "To continue reading", "Please sign in",
        "You've reached your limit", "subscribers only",
    )

    content = None
    method = None

    # Tier 1: jina.ai — fast, free, but fails on paywalls
    jina_result, _ = _try_jina(url, debug_dir=debug_dir)
    if jina_result:
        content, method = jina_result, "jina.ai"
    else:
        # Jina hit a paywall — try archive.is for the full article
        content, method = _try_archive_is(url, debug_dir=debug_dir)
        if not content:
            for fetcher in (_try_obscura, _try_playwright):
                content, method = fetcher(url, debug_dir=debug_dir)
                if content:
                    break
        if content and any(signal.lower() in content[:500].lower()
                           for signal in _PAYWALL_SIGNALS):
            print(f"  WARNING: content may be a paywall preview "
                  f"({len(content)} chars). Consider finding an alternate source.",
                  file=sys.stderr)

    if content:
        out_path.write_text(content, encoding="utf-8")
        print(f"[fetch_page] {url} -> {out_path}  ({method})", file=sys.stderr)
        print(content)
    else:
        out_path.write_text(f"(failed to fetch {url})\n", encoding="utf-8")
        print(f"[fetch_page] FAILED: {url} — all methods exhausted", file=sys.stderr)
        print(f"(failed to fetch {url})")


if __name__ == "__main__":
    main()
