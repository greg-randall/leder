"""Load and validate pipeline configuration."""
from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field

import yaml


@dataclass
class ArticleConfig:
    path: str

@dataclass
class CorpusConfig:
    root: str

@dataclass
class OutputConfig:
    dir: str = "."
    web_cache_dir: str = "web_cache/"

@dataclass
class StageAConfig:
    model: str = "deepseek-v4-pro[1m]"
    quality_gate: bool = True

@dataclass
class StageBConfig:
    model: str = "deepseek-v4-flash"
    concurrency: int = 3

@dataclass
class StageCConfig:
    quote_match_method: str = "normalized"

@dataclass
class PipelineConfig:
    article: ArticleConfig
    corpus: CorpusConfig
    output: OutputConfig = field(default_factory=OutputConfig)
    stage_a: StageAConfig = field(default_factory=StageAConfig)
    stage_b: StageBConfig = field(default_factory=StageBConfig)
    stage_c: StageCConfig = field(default_factory=StageCConfig)
    project_root: str = ""

    @classmethod
    def from_yaml(cls, path: str) -> PipelineConfig:
        try:
            with open(path) as f:
                raw = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse config YAML at {path}: {e}")

        if raw is None:
            raise ValueError(f"Config file is empty: {path}")

        article_raw = raw.get("article")
        if article_raw is None:
            raise ValueError(
                "Missing required config section: 'article'. "
                "Your config.yaml must include an 'article' section with a 'path' field."
            )
        corpus_raw = raw.get("corpus")
        if corpus_raw is None:
            raise ValueError(
                "Missing required config section: 'corpus'. "
                "Your config.yaml must include a 'corpus' section with a 'root' field."
            )

        config = cls(
            article=ArticleConfig(**article_raw),
            corpus=CorpusConfig(**corpus_raw),
            output=OutputConfig(**raw.get("output", {})),
            stage_a=StageAConfig(**raw.get("stage_a", {})),
            stage_b=StageBConfig(**raw.get("stage_b", {})),
            stage_c=StageCConfig(**raw.get("stage_c", {})),
        )
        config.project_root = str(Path(path).resolve().parent.parent)
        return config

    def resolve_path(self, relative_path: str) -> str:
        """Resolve a path relative to the project root."""
        return str(Path(self.project_root) / relative_path)
