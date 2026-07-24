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

    # Parent parser with common flags shared across all subcommands
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--skip-startup-check", action="store_true",
                        help="Skip prerequisite tool validation")
    common.add_argument("--recheck", action="store_true",
                        help="Force re-check of prerequisites (ignore cache)")

    all_parser = subparsers.add_parser("all", parents=[common],
                                        help="Run the full pipeline")
    all_parser.add_argument(
        "--config", default="config.yaml", help="Path to config file"
    )
    a_parser = subparsers.add_parser("stage-a", parents=[common], help="Extract claims from article")
    a_parser.add_argument("--article", help="Path to article markdown")
    a_parser.add_argument("--output", default="targets.json")
    a_parser.add_argument(
        "--config", default="config.yaml", help="Path to config file"
    )
    a_parser.add_argument(
        "--no-quality-gate",
        action="store_true",
        help="Skip the quality-gate re-read of the article",
    )
    a_parser.add_argument("--playbooks", help="Comma-separated list of playbooks to run")

    b_parser = subparsers.add_parser(
        "stage-b", parents=[common], help="Verify claims against corpus"
    )
    b_parser.add_argument("--claims", default=None, help=argparse.SUPPRESS)
    b_parser.add_argument("--targets", default="targets.json", help="Input targets file")
    b_parser.add_argument("--findings", default="findings.json", help="Output findings file")
    b_parser.add_argument("--output", default="claims.json")
    b_parser.add_argument(
        "--config", default="config.yaml", help="Path to config file"
    )
    b_parser.add_argument(
        "--force-run", action="store_true",
        help="Skip corpus readiness check (prepare-2/3 summaries not required)",
    )
    b_parser.add_argument(
        "--debug", type=int, default=0, metavar="N",
        help="Randomly sample N claims, save agent transcripts to debug/",
    )
    b_parser.add_argument(
        "--debug-ids", type=str, default=None, metavar="1,3,5",
        help="Debug specific target indices (comma-separated, 0-based)",
    )

    c_parser = subparsers.add_parser(
        "stage-c", parents=[common], help="Rebuild article with footnotes"
    )
    c_parser.add_argument("--article", help="Path to article markdown")
    c_parser.add_argument("--claims", default=None, help=argparse.SUPPRESS)
    c_parser.add_argument("--findings", default="findings.json", help="Input findings file")
    c_parser.add_argument("--output", default="article-sourced.md")
    c_parser.add_argument(
        "--config", default="config.yaml", help="Path to config file"
    )

    d_parser = subparsers.add_parser(
        "stage-d", parents=[common], help="Convert sourced article to HTML"
    )
    d_parser.add_argument("--input", default="article-sourced.md")
    d_parser.add_argument("--output", default="article-sourced.html")
    d_parser.add_argument(
        "--config", default="config.yaml", help="Path to config file"
    )

    e_parser = subparsers.add_parser(
        "stage-e", parents=[common], help="Generate .docx with comments from sourced article"
    )
    e_parser.add_argument("--article", default="article-sourced.md")
    e_parser.add_argument("--claims", default="findings.json")
    e_parser.add_argument("--output", default="article-sourced.docx")
    e_parser.add_argument(
        "--config", default="config.yaml", help="Path to config file"
    )

    check_parser = subparsers.add_parser(
        "check", parents=[common], help="Run startup validation only"
    )
    check_parser.add_argument(
        "--config", default="config.yaml", help=argparse.SUPPRESS
    )

    p1 = subparsers.add_parser("prepare-1", parents=[common], help="Convert raw source files to markdown")
    p1.add_argument("--config", default="config.yaml")
    p1.add_argument("--source", help="Override prepare.source_root")
    p1.add_argument("--whisper-model", help="Override prepare.audio.model (e.g. small, medium)")
    p1.add_argument("--whisper-device", choices=["auto", "cuda", "cpu"],
                    help="Override prepare.audio.device")
    p1.add_argument("--force", action="store_true")

    p2 = subparsers.add_parser("prepare-2", parents=[common], help="Summarize converted documents")
    p2.add_argument("--config", default="config.yaml")
    p2.add_argument("--model", help="Override prepare.summarize.model")
    p2.add_argument("--force", action="store_true")

    p3 = subparsers.add_parser("prepare-3", parents=[common], help="Recursive folder + crosscutting rollup")
    p3.add_argument("--config", default="config.yaml")
    p3.add_argument("--only", choices=["tree", "crosscutting", "concat"],
                    help="Run only one rollup phase (default: all)")
    p3.add_argument("--force", action="store_true")

    pp = subparsers.add_parser("prepare", parents=[common], help="Run prepare-1, prepare-2, prepare-3 in order")
    pp.add_argument("--config", default="config.yaml")
    pp.add_argument("--whisper-model", help="Override prepare.audio.model (e.g. small, medium)")
    pp.add_argument("--whisper-device", choices=["auto", "cuda", "cpu"],
                    help="Override prepare.audio.device")
    pp.add_argument("--force", action="store_true")

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
    print("    [o] overwrite   [t] timestamp   [q] quit", file=sys.stderr)

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

        ok = validate_startup(force=getattr(args, "recheck", False))
        sys.exit(0 if ok else 1)

    # --- Load config for all other commands ---
    config = _load_config(args.config)

    # --- Configure provider (DeepSeek or other) ---
    _setup_provider_env(config)

    # --- Startup check for all commands (unless skipped) ---
    if not getattr(args, "skip_startup_check", False):
        from pipeline.startup_check import validate_startup
        if not validate_startup(force=getattr(args, "recheck", False)):
            sys.exit(1)

    # --- Corpus prep (prepare-1/2/3 — NOT part of 'all') ---
    if args.command in ("prepare-1", "prepare-2", "prepare-3", "prepare"):
        from pipeline import startup_check as _sc

        corpus_root = config.resolve_path(config.corpus.root)
        prep = config.prepare

        for w in _sc.warn_stale_corpus(corpus_root):
            print(w, file=sys.stderr)
        vkey_warn = _sc.warn_missing_vision_key(prep.vision_fallback.enabled)
        if vkey_warn:
            print(vkey_warn, file=sys.stderr)
        audio_warn = _sc.warn_missing_audio_deps(prep.audio.enabled)
        if audio_warn:
            print(audio_warn, file=sys.stderr)

        force = getattr(args, "force", False)
        convert_failures = 0

        if args.command in ("prepare-1", "prepare"):
            from pipeline.prepare_1_convert import run_prepare_1
            source_root = config.resolve_path(
                getattr(args, "source", None) or prep.source_root
            )
            vision_cfg = {
                "enabled": prep.vision_fallback.enabled,
                "model": prep.vision_fallback.model,
                "min_words": prep.vision_fallback.min_words,
                "max_pages_per_doc": prep.vision_fallback.max_pages_per_doc,
                "ocr_images": prep.ocr_images,
            }
            audio_cfg = {
                "enabled": prep.audio.enabled,
                "model": getattr(args, "whisper_model", None) or prep.audio.model,
                "device": getattr(args, "whisper_device", None) or prep.audio.device,
            }
            report = run_prepare_1(
                source_root=source_root, corpus_root=corpus_root,
                workers=prep.convert_workers,
                vision_cfg=vision_cfg, audio_cfg=audio_cfg, force=force,
                text_native_exts={e.lower() for e in prep.text_native_extensions},
            )
            convert_failures = report["failure_count"]
            if args.command == "prepare-1":
                sys.exit(1 if convert_failures else 0)

        if args.command in ("prepare-2", "prepare"):
            from pipeline.prepare_2_summarize import run_prepare_2
            run_prepare_2(
                corpus_root=corpus_root,
                model=getattr(args, "model", None) or prep.summarize.model,
                workers=prep.summarize.workers, force=force,
            )

        if args.command in ("prepare-3", "prepare"):
            from pipeline.prepare_3_rollup import run_prepare_3
            run_prepare_3(
                corpus_root=corpus_root, model=prep.rollup.model,
                big_call_model=prep.rollup.big_call_model,
                workers=prep.rollup.workers, crosscutting=prep.rollup.crosscutting,
                force=force, only=getattr(args, "only", None),
            )

        # Combined 'prepare': surface prepare-1 failures in the exit code even
        # though 2 and 3 still ran.
        if args.command == "prepare" and convert_failures:
            print(f"prepare: {convert_failures} file(s) failed conversion — "
                  "see UNCONVERTED.md.", file=sys.stderr)
            sys.exit(1)
        return

    # --- Stage A: Claim extraction ---
    if args.command in ("all", "stage-a"):
        playbook_names: list[str] | None = (
            args.playbooks.split(",")
            if hasattr(args, "playbooks") and args.playbooks
            else config.prepare.playbooks.active
        )
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
            playbook_dir=config.prepare.playbooks.dir,
            playbook_names=playbook_names,
        )
        if doc is not None:
            n = getattr(doc, 'total_findings', None)
            if n is None or n == 0:
                # New playbook path: count from the written file
                try:
                    import json
                    with open(output_path) as f:
                        data = json.load(f)
                    n = len(data.get("targets", []))
                except Exception:
                    n = len(getattr(doc, 'claims', []))
            print(f"Stage A complete: {n} targets extracted")

    # --- Stage B: Claim verification ---
    if args.command in ("all", "stage-b"):
        # Backward compat: --claims maps to targets_path
        args_targets = getattr(args, "targets", "targets.json")
        args_claims = getattr(args, "claims", None)
        targets_path = config.resolve_path(
            args_claims if args_claims is not None else args_targets
        )
        output_path = config.resolve_path(
            getattr(args, "output", getattr(args, "findings", "findings.json"))
        )

        from pipeline.stage_b_verify import run_stage_b

        output_path = _resolve_output_path(output_path)
        doc = run_stage_b(
            targets_path=targets_path,
            output_path=output_path,
            corpus_root=config.resolve_path(config.corpus.root),
            concurrency=config.stage_b.concurrency,
            timeout=config.stage_b.timeout,
            max_turns=config.stage_b.max_turns,
            debug_count=getattr(args, "debug", 0),
            debug_ids=(
                [int(x.strip()) for x in args.debug_ids.split(",")]
                if getattr(args, "debug_ids", None) else None
            ),
            playbook_dir=config.prepare.playbooks.dir,
            pricing=config.pricing,
            force_run=getattr(args, "force_run", False),
        )
        if hasattr(doc, 'findings'):
            verified_count = len(doc.findings)
        else:
            verified_count = sum(1 for c in doc.claims if c.verdict is not None)
        print(f"Stage B complete: {verified_count} findings")

    # --- Stage C: Article rebuild with footnotes ---
    if args.command in ("all", "stage-c"):
        article_path = (
            getattr(args, "article", None)
            or config.resolve_path(config.article.path)
        )
        # Backward compat: --claims feeds into findings input
        args_findings = getattr(args, "findings", "findings.json")
        args_claims_c = getattr(args, "claims", None)
        claims_path = config.resolve_path(
            args_claims_c if args_claims_c is not None else args_findings
        )
        output_path = config.resolve_path(
            getattr(args, "output", "article-sourced.md")
        )

        from pipeline.stage_c_rebuild import run_stage_c

        if not os.path.exists(args_findings):
            alt = os.path.join(os.path.dirname(args_findings) or ".", "claims.json")
            if os.path.exists(alt):
                print(f"Note: {args_findings} not found, using {alt} instead.",
                      file=sys.stderr)
                args_findings = alt
            else:
                print(f"ERROR: {args_findings} not found. Run stage-b first to generate it.",
                      file=sys.stderr)
                sys.exit(1)

        output_path = _resolve_output_path(output_path)
        run_stage_c(
            article_path=article_path,
            claims_path=args_claims_c,
            findings_path=args_findings,
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
        if not os.path.exists(input_path):
            print(f"ERROR: {input_path} not found. Run stage-c first to generate it.",
                  file=sys.stderr)
            sys.exit(1)
        run_stage_d(input_path, output_path)
        print(f"Stage D complete: {output_path}")

    # --- Stage E: .docx with comments ---
    if args.command in ("all", "stage-e"):
        article_path = getattr(args, "article", "article-sourced.md")
        article_path = config.resolve_path(article_path) if config else article_path
        claims_path = getattr(args, "claims", "findings.json")
        claims_path = config.resolve_path(claims_path) if config else claims_path
        output_path = getattr(args, "output", "article-sourced.docx")
        output_path = config.resolve_path(output_path) if config else output_path

        if not os.path.exists(claims_path):
            alt = os.path.join(os.path.dirname(claims_path) or ".", "claims.json")
            if os.path.exists(alt):
                print(f"Note: {claims_path} not found, using {alt} instead.",
                      file=sys.stderr)
                claims_path = alt

        for fpath, stage in [(article_path, "stage-c"), (claims_path, "stage-b")]:
            if not os.path.exists(fpath):
                print(f"ERROR: {fpath} not found. Run {stage} first to generate it.",
                      file=sys.stderr)
                sys.exit(1)

        from pipeline.stage_e_docx import run_stage_e
        run_stage_e(article_path, claims_path, output_path)
        print(f"Stage E complete: {output_path}")


if __name__ == "__main__":
    main()
