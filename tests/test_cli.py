"""Tests for the CLI entry point."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import yaml

from pipeline.cli import build_parser, _resolve_config_path, _load_config


# ---------------------------------------------------------------------------
# Parser construction and argument routing
# ---------------------------------------------------------------------------


def test_build_parser():
    """build_parser returns a configured ArgumentParser."""
    parser = build_parser()
    assert parser is not None
    assert parser.description is not None


def test_parser_default_config():
    """The default --config value is pipeline/config.yaml for stage commands."""
    parser = build_parser()
    args = parser.parse_args(["stage-a"])
    assert args.config == "pipeline/config.yaml"


def test_parser_stage_a():
    parser = build_parser()
    args = parser.parse_args(
        ["stage-a", "--article", "test.md", "--output", "out.json", "--no-quality-gate"]
    )
    assert args.command == "stage-a"
    assert args.article == "test.md"
    assert args.output == "out.json"
    assert args.no_quality_gate is True


def test_parser_stage_a_defaults():
    """stage-a without flags uses defaults."""
    parser = build_parser()
    args = parser.parse_args(["stage-a"])
    assert args.article is None
    assert args.output == "claims.json"
    assert args.no_quality_gate is False


def test_parser_all():
    parser = build_parser()
    args = parser.parse_args(["all", "--config", "custom.yaml", "--skip-startup-check"])
    assert args.command == "all"
    assert args.config == "custom.yaml"
    assert args.skip_startup_check is True


def test_parser_all_defaults():
    parser = build_parser()
    args = parser.parse_args(["all"])
    assert args.config == "pipeline/config.yaml"
    assert args.skip_startup_check is False


def test_parser_stage_b():
    parser = build_parser()
    args = parser.parse_args(
        ["stage-b", "--claims", "my_claims.json", "--output", "verified.json"]
    )
    assert args.command == "stage-b"
    assert args.claims == "my_claims.json"
    assert args.output == "verified.json"


def test_parser_stage_b_defaults():
    parser = build_parser()
    args = parser.parse_args(["stage-b"])
    assert args.claims == "claims.json"
    assert args.output == "claims.json"


def test_parser_stage_c():
    parser = build_parser()
    args = parser.parse_args(
        ["stage-c", "--article", "article.md", "--claims", "cl.json", "--output", "out.md"]
    )
    assert args.command == "stage-c"
    assert args.article == "article.md"
    assert args.claims == "cl.json"
    assert args.output == "out.md"


def test_parser_stage_c_defaults():
    parser = build_parser()
    args = parser.parse_args(["stage-c"])
    assert args.article is None
    assert args.claims == "claims.json"
    assert args.output == "article-sourced.md"


def test_parser_check():
    parser = build_parser()
    args = parser.parse_args(["check"])
    assert args.command == "check"


def test_parser_unknown_command():
    """Unknown subcommand should exit with error."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["unknown"])


def test_parser_no_command():
    """No command should print help and exit 0."""
    parser = build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--help"])
    assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# _resolve_config_path
# ---------------------------------------------------------------------------


def test_resolve_config_path_absolute():
    result = _resolve_config_path("/etc/myapp/config.yaml")
    assert result == "/etc/myapp/config.yaml"


def test_resolve_config_path_relative():
    result = _resolve_config_path("pipeline/config.yaml")
    expected = str(Path.cwd() / "pipeline/config.yaml")
    assert result == expected


# ---------------------------------------------------------------------------
# _load_config
# ---------------------------------------------------------------------------


def test_load_config_missing():
    """Loading a non-existent config prints error and exits."""
    with pytest.raises(SystemExit) as exc_info:
        _load_config("/tmp/nonexistent_config_12345.yaml")
    assert exc_info.value.code == 1


def test_load_config_invalid_yaml(tmp_path):
    """Loading an invalid YAML file prints error and exits."""
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(": : invalid yaml [[")
    with pytest.raises(SystemExit) as exc_info:
        _load_config(str(cfg))
    assert exc_info.value.code == 1


def test_load_config_valid(tmp_path):
    """Loading a valid config returns a PipelineConfig instance."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        yaml.dump(
            {
                "article": {"path": "article.md"},
                "corpus": {"root": "source-docs-and-summaries/"},
                "output": {"dir": ".", "web_cache_dir": "web_cache/"},
                "stage_a": {"model": "deepseek-v4-pro", "quality_gate": True},
                "stage_b": {"model": "deepseek-v4-flash", "concurrency": 32, "timeout": 600, "max_turns": 60},
            }
        )
    )
    config = _load_config(str(cfg))
    from pipeline.config import PipelineConfig

    assert isinstance(config, PipelineConfig)
    assert config.article.path == "article.md"
    assert config.corpus.root == "source-docs-and-summaries/"
    assert config.stage_b.concurrency == 32


# ---------------------------------------------------------------------------
# main() — edge cases only, actual stages mocked
# ---------------------------------------------------------------------------


def test_main_no_command(monkeypatch):
    """Running main() with no command prints help and exits 0."""
    monkeypatch.setattr("sys.argv", ["pipeline.cli"])
    from pipeline.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0


def test_main_check_calls_validate(monkeypatch):
    """'check' subcommand calls validate_startup and exits accordingly."""
    called = False

    def mock_validate():
        nonlocal called
        called = True
        return True

    monkeypatch.setattr("pipeline.startup_check.validate_startup", mock_validate)
    monkeypatch.setattr("sys.argv", ["pipeline.cli", "check"])

    from pipeline.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0
    assert called, "validate_startup was not called"


def test_main_check_failure(monkeypatch):
    """When validate_startup returns False, main exits with code 1."""
    monkeypatch.setattr("pipeline.startup_check.validate_startup", lambda: False)
    monkeypatch.setattr("sys.argv", ["pipeline.cli", "check"])

    from pipeline.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1
