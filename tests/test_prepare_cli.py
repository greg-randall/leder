"""Tests for prepare-* CLI wiring."""
from __future__ import annotations

from pipeline.cli import build_parser


def test_parser_prepare_subcommands():
    parser = build_parser()
    for cmd in ["prepare-1", "prepare-2", "prepare-3", "prepare"]:
        args = parser.parse_args([cmd])
        assert args.command == cmd


def test_prepare_1_flags():
    parser = build_parser()
    args = parser.parse_args(["prepare-1", "--source", "raw/", "--force"])
    assert args.command == "prepare-1"
    assert args.source == "raw/"
    assert args.force is True


def test_prepare_3_only_flag():
    parser = build_parser()
    args = parser.parse_args(["prepare-3", "--only", "crosscutting"])
    assert args.only == "crosscutting"


def test_prepare_not_in_all_choices():
    parser = build_parser()
    # 'all' does its own thing, but prepare commands aren't special-cased into it.
    # Verify parser distinguishes them.
    args = parser.parse_args(["all"])
    assert args.command == "all"
