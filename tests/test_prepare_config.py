"""Tests for the prepare: config section."""
from __future__ import annotations

import textwrap
from pathlib import Path

from pipeline.config import PipelineConfig


def _write(tmp_path: Path, body: str) -> str:
    # config.yaml lives at project root; project_root = parent.parent of the file.
    root = tmp_path / "proj"
    (root / "pipeline").mkdir(parents=True)
    cfg = root / "config.yaml"
    cfg.write_text(textwrap.dedent(body))
    return str(cfg)


def test_corpus_description_defaults_to_empty(tmp_path):
    cfg_path = _write(tmp_path, """
        article: {path: "article.md"}
        corpus: {root: "corpus/"}
    """)
    cfg = PipelineConfig.from_yaml(cfg_path)
    assert cfg.corpus.description == ""


def test_corpus_description_parsed_from_yaml(tmp_path):
    cfg_path = _write(tmp_path, """
        article: {path: "article.md"}
        corpus:
          root: "corpus/"
          description: "Transcripts of city council meetings."
    """)
    cfg = PipelineConfig.from_yaml(cfg_path)
    assert cfg.corpus.description == "Transcripts of city council meetings."


def test_prepare_audio_defaults(tmp_path):
    cfg_path = _write(tmp_path, """
        article: {path: "article.md"}
        corpus: {root: "corpus/"}
    """)
    cfg = PipelineConfig.from_yaml(cfg_path)
    assert cfg.prepare.audio.enabled is True
    assert cfg.prepare.audio.model == "medium"
    assert cfg.prepare.audio.device == "auto"


def test_prepare_audio_parsed(tmp_path):
    cfg_path = _write(tmp_path, """
        article: {path: "article.md"}
        corpus: {root: "corpus/"}
        prepare:
          audio: {enabled: false, model: "small", device: "cuda"}
    """)
    cfg = PipelineConfig.from_yaml(cfg_path)
    assert cfg.prepare.audio.enabled is False
    assert cfg.prepare.audio.model == "small"
    assert cfg.prepare.audio.device == "cuda"


def test_prepare_defaults_when_absent(tmp_path):
    cfg_path = _write(tmp_path, """
        article: {path: "article.md"}
        corpus: {root: "corpus/"}
    """)
    cfg = PipelineConfig.from_yaml(cfg_path)
    assert cfg.prepare.ocr_images is True
    assert cfg.prepare.vision_fallback.enabled is True
    assert cfg.prepare.vision_fallback.model == "gpt-4o-mini"
    assert cfg.prepare.rollup.crosscutting is True
    assert cfg.prepare.summarize.model == "deepseek-v4-flash"


def test_prepare_parsed_from_yaml(tmp_path):
    cfg_path = _write(tmp_path, """
        article: {path: "article.md"}
        corpus: {root: "corpus/"}
        prepare:
          source_root: "raw/"
          convert_workers: 4
          ocr_images: false
          vision_fallback: {enabled: false, model: "gpt-4o", min_words: 5, max_pages_per_doc: 10}
          summarize: {model: "m-flash", workers: 7}
          rollup: {model: "m-pro", big_call_model: "m-pro[1m]", workers: 3, crosscutting: false}
    """)
    cfg = PipelineConfig.from_yaml(cfg_path)
    assert cfg.prepare.source_root == "raw/"
    assert cfg.prepare.convert_workers == 4
    assert cfg.prepare.ocr_images is False
    assert cfg.prepare.vision_fallback.min_words == 5
    assert cfg.prepare.vision_fallback.max_pages_per_doc == 10
    assert cfg.prepare.summarize.workers == 7
    assert cfg.prepare.rollup.big_call_model == "m-pro[1m]"
    assert cfg.prepare.rollup.crosscutting is False


def test_file_timeout_and_min_content_bytes_default_when_unset(tmp_path):
    cfg_path = _write(tmp_path, """
        article: {path: "article.md"}
        corpus: {root: "corpus/"}
    """)
    cfg = PipelineConfig.from_yaml(cfg_path)
    assert cfg.prepare.convert.file_timeout == 300
    assert cfg.prepare.convert.min_content_bytes == 100


def test_file_timeout_and_min_content_bytes_parsed_from_yaml(tmp_path):
    cfg_path = _write(tmp_path, """
        article: {path: "article.md"}
        corpus: {root: "corpus/"}
        prepare:
          convert:
            file_timeout: 600
            min_content_bytes: 50
    """)
    cfg = PipelineConfig.from_yaml(cfg_path)
    assert cfg.prepare.convert.file_timeout == 600
    assert cfg.prepare.convert.min_content_bytes == 50
