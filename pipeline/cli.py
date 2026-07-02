"""Source-linking pipeline CLI.

Usage:
    python3 -m pipeline.cli all
    python3 -m pipeline.cli stage-a
    python3 -m pipeline.cli stage-b
    python3 -m pipeline.cli stage-c
    python3 -m pipeline.cli check

Environment:
    Set ANTHROPIC_API_KEY, or create a .env file with DEEPSEEK_API_KEY
    (or any other provider key). The pipeline loads .env from the project
    root and sets ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN / ANTHROPIC_MODEL
    accordingly.

    For DeepSeek, your .env should contain:
        DEEPSEEK_API_KEY=sk-...
        ANTHROPIC_PROVIDER=deepseek

    The pipeline will auto-configure:
        ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
        ANTHROPIC_AUTH_TOKEN=<DEEPSEEK_API_KEY>
        ANTHROPIC_MODEL=<stage.model from config>
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _find_project_root() -> Path:
    """Find the project root (where .env and pipeline/ live)."""
    return Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """Load .env file from the project root into os.environ."""
    env_file = _find_project_root() / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key not in os.environ:
                os.environ[key] = val


def _setup_provider_env(config) -> None:
    """Configure ANTHROPIC_* env vars based on the provider.

    If DEEPSEEK_API_KEY is set (from .env or environment), configure
    the Anthropic-compatible endpoint for DeepSeek.
    """
    provider = os.environ.get("ANTHROPIC_PROVIDER", "").lower()

    if provider == "deepseek" or os.environ.get("DEEPSEEK_API_KEY"):
        deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if deepseek_key:
            os.environ.setdefault("ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic")
            # ANTHROPIC_AUTH_TOKEN for Claude Code CLI and SDK —
            # matches sample-code pattern, avoids connector warning
            os.environ.setdefault("ANTHROPIC_AUTH_TOKEN", deepseek_key)
            os.environ.setdefault("ANTHROPIC_DEFAULT_OPUS_MODEL", "deepseek-v4-pro[1m]")
            os.environ.setdefault("ANTHROPIC_DEFAULT_SONNET_MODEL", "deepseek-v4-pro[1m]")
            os.environ.setdefault("ANTHROPIC_DEFAULT_HAIKU_MODEL", "deepseek-v4-flash")
            os.environ.setdefault("CLAUDE_CODE_SUBAGENT_MODEL", "deepseek-v4-flash")
            os.environ.setdefault("CLAUDE_CODE_EFFORT_LEVEL", "max")

    # Set the primary model from config if available
    stage_b_model = getattr(getattr(config, 'stage_b', None), 'model', None)
    if stage_b_model:
        os.environ["ANTHROPIC_MODEL"] = stage_b_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Source-linking pipeline: extract, verify, and footnote factual claims.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Pipeline stage to run")

    all_parser = subparsers.add_parser("all", help="Run the full pipeline")
    all_parser.add_argument(
        "--config", default="pipeline/config.yaml", help="Path to config file"
    )
    all_parser.add_argument("--skip-startup-check", action="store_true")

    a_parser = subparsers.add_parser("stage-a", help="Extract claims from article")
    a_parser.add_argument("--article", help="Path to article markdown")
    a_parser.add_argument("--output", default="claims.json")
    a_parser.add_argument(
        "--config", default="pipeline/config.yaml", help="Path to config file"
    )
    a_parser.add_argument(
        "--no-quality-gate",
        action="store_true",
        help="Skip the quality-gate re-read of the article",
    )

    b_parser = subparsers.add_parser(
        "stage-b", help="Verify claims against corpus"
    )
    b_parser.add_argument("--claims", default="claims.json")
    b_parser.add_argument("--output", default="claims.json")
    b_parser.add_argument(
        "--config", default="pipeline/config.yaml", help="Path to config file"
    )
    b_parser.add_argument(
        "--debug", type=int, default=0, metavar="N",
        help="Randomly sample N claims, save agent transcripts to debug/",
    )

    c_parser = subparsers.add_parser(
        "stage-c", help="Rebuild article with footnotes"
    )
    c_parser.add_argument("--article", help="Path to article markdown")
    c_parser.add_argument("--claims", default="claims.json")
    c_parser.add_argument("--output", default="article-sourced.md")
    c_parser.add_argument(
        "--config", default="pipeline/config.yaml", help="Path to config file"
    )

    d_parser = subparsers.add_parser(
        "stage-d", help="Convert sourced article to HTML"
    )
    d_parser.add_argument("--input", default="article-sourced.md")
    d_parser.add_argument("--output", default="article-sourced.html")
    d_parser.add_argument(
        "--config", default="pipeline/config.yaml", help="Path to config file"
    )

    e_parser = subparsers.add_parser(
        "stage-e", help="Generate .docx with comments from sourced article"
    )
    e_parser.add_argument("--article", default="article-sourced.md")
    e_parser.add_argument("--claims", default="claims-full-article-verified.json")
    e_parser.add_argument("--output", default="article-sourced.docx")
    e_parser.add_argument(
        "--config", default="pipeline/config.yaml", help="Path to config file"
    )

    check_parser = subparsers.add_parser(
        "check", help="Run startup validation only"
    )
    check_parser.add_argument(
        "--config", default="pipeline/config.yaml", help=argparse.SUPPRESS
    )

    return parser


def _resolve_config_path(config_arg: str) -> str:
    """Resolve a config file path.

    If the path is relative, resolve it against the current working directory.
    Returns the absolute path.
    """
    if os.path.isabs(config_arg):
        return config_arg
    return str(Path.cwd() / config_arg)


def _load_config(config_arg: str):
    """Load pipeline config from the given path.

    Prints an error and exits if the file does not exist or is invalid.
    Returns None if called with no argument (should not happen in practice).
    """
    from pipeline.config import PipelineConfig

    config_path = _resolve_config_path(config_arg)
    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}")
        sys.exit(1)
    try:
        return PipelineConfig.from_yaml(config_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Config error: {e}")
        sys.exit(1)


def _resolve_output_path(path: str) -> str:
    """Check for conflicts and let the user choose what to do.

    Returns the final path to use (may differ from input if user chose timestamp).
    """
    if not os.path.exists(path):
        return path

    size = os.path.getsize(path)
    print(f"\n⚠️  Output file exists: {path} ({size:,} bytes)", file=sys.stderr)
    print(f"    [o] overwrite   [t] timestamp   [q] quit", file=sys.stderr)

    while True:
        try:
            choice = input("    Choose: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.", file=sys.stderr)
            sys.exit(1)

        if choice in ("o", "overwrite"):
            return path
        elif choice in ("t", "timestamp"):
            from datetime import datetime
            stem = Path(path).stem
            suffix = Path(path).suffix
            parent = Path(path).parent
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            new_path = str(parent / f"{stem}-{ts}{suffix}")
            print(f"    Using: {new_path}", file=sys.stderr)
            return new_path
        elif choice in ("q", "quit"):
            print("Cancelled.", file=sys.stderr)
            sys.exit(1)
        else:
            print(f"    Unknown option: '{choice}' — [o]verwrite, [t]imestamp, [q]uit", file=sys.stderr)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Load .env before anything else — providers need env vars set up
    _load_dotenv()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # --- check: startup validation only, no config needed ---
    if args.command == "check":
        from pipeline.startup_check import validate_startup

        ok = validate_startup()
        sys.exit(0 if ok else 1)

    # --- Load config for all other commands ---
    config = _load_config(args.config)

    # --- Configure provider (DeepSeek or other) ---
    _setup_provider_env(config)

    # --- 'all' may run the startup check first ---
    if args.command == "all" and not getattr(args, "skip_startup_check", False):
        from pipeline.startup_check import validate_startup

        if not validate_startup():
            sys.exit(1)

    # --- Stage A: Claim extraction ---
    if args.command in ("all", "stage-a"):
        article_path = (
            getattr(args, "article", None)
            or config.resolve_path(config.article.path)
        )
        output_path = config.resolve_path(
            getattr(args, "output", "claims.json")
        )

        from pipeline.stage_a_extract import run_stage_a

        output_path = _resolve_output_path(output_path)
        doc = run_stage_a(
            article_path=article_path,
            output_path=output_path,
            corpus_root=config.resolve_path(config.corpus.root),
            project_name=config.corpus.project if hasattr(config.corpus, "project") else "",
            model=config.stage_a.model,
            quality_gate=not getattr(args, "no_quality_gate", False),
        )
        print(f"Stage A complete: {len(doc.claims)} claims extracted")

    # --- Stage B: Claim verification ---
    if args.command in ("all", "stage-b"):
        claims_path = config.resolve_path(
            getattr(args, "claims", "claims.json")
        )
        output_path = config.resolve_path(
            getattr(args, "output", getattr(args, "claims", "claims.json"))
        )

        from pipeline.stage_b_verify import run_stage_b

        output_path = _resolve_output_path(output_path)
        doc = run_stage_b(
            claims_path=claims_path,
            output_path=output_path,
            corpus_root=config.resolve_path(config.corpus.root),
            concurrency=config.stage_b.concurrency,
            timeout=config.stage_b.timeout,
            max_turns=config.stage_b.max_turns,
            debug_count=getattr(args, "debug", 0),
        )
        verified_count = sum(1 for c in doc.claims if c.verdict is not None)
        print(f"Stage B complete: {verified_count} claims verified")

    # --- Stage C: Article rebuild with footnotes ---
    if args.command in ("all", "stage-c"):
        article_path = (
            getattr(args, "article", None)
            or config.resolve_path(config.article.path)
        )
        claims_path = config.resolve_path(
            getattr(args, "claims", "claims.json")
        )
        output_path = config.resolve_path(
            getattr(args, "output", "article-sourced.md")
        )

        from pipeline.stage_c_rebuild import run_stage_c

        output_path = _resolve_output_path(output_path)
        run_stage_c(
            article_path=article_path,
            claims_path=claims_path,
            output_path=output_path,
            model=config.stage_a.model,
        )
        print(f"Stage C complete: {output_path}")

    # --- Stage D: HTML conversion ---
    if args.command in ("all", "stage-d"):
        input_path = getattr(args, "input", "article-sourced.md")
        input_path = config.resolve_path(input_path) if config else input_path
        output_path = getattr(args, "output", "article-sourced.html")
        output_path = config.resolve_path(output_path) if config else output_path

        from pipeline.stage_d_html import run_stage_d
        run_stage_d(input_path, output_path)
        print(f"Stage D complete: {output_path}")

    # --- Stage E: .docx with comments ---
    if args.command in ("all", "stage-e"):
        article_path = getattr(args, "article", "article-sourced.md")
        article_path = config.resolve_path(article_path) if config else article_path
        claims_path = getattr(args, "claims", "claims-full-article-verified.json")
        claims_path = config.resolve_path(claims_path) if config else claims_path
        output_path = getattr(args, "output", "article-sourced.docx")
        output_path = config.resolve_path(output_path) if config else output_path

        from pipeline.stage_e_docx import run_stage_e
        run_stage_e(article_path, claims_path, output_path)
        print(f"Stage E complete: {output_path}")


if __name__ == "__main__":
    main()
