"""Load and validate pipeline configuration."""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field

import yaml


@dataclass
class ArticleConfig:
    path: str


@dataclass
class CorpusConfig:
    root: str = "./"


@dataclass
class OutputConfig:
    dir: str = "."


@dataclass
class StageAConfig:
    model: str = "deepseek-v4-pro"
    quality_gate: bool = True


@dataclass
class StageBConfig:
    model: str = "deepseek-v4-pro"
    concurrency: int = 32
    timeout: int = 600
    max_turns: int = 60


@dataclass
class StageCConfig:
    quote_match_method: str = "normalized"


@dataclass
class VisionFallbackConfig:
    enabled: bool = True
    model: str = "gpt-4o-mini"
    min_words: int = 20
    max_pages_per_doc: int = 30


@dataclass
class PrepareSummarizeConfig:
    model: str = "deepseek-v4-flash"
    workers: int = 50


@dataclass
class PrepareRollupConfig:
    model: str = "deepseek-v4-pro"
    big_call_model: str = "deepseek-v4-pro[1m]"
    workers: int = 12
    crosscutting: bool = True


@dataclass
class AudioConfig:
    enabled: bool = True
    model: str = "medium"
    device: str = "auto"   # auto = GPU if available else CPU (with warning)


@dataclass
class PlaybooksConfig:
    dir: str = "pipelines/"
    active: list[str] = field(default_factory=lambda: ["fact_check"])


@dataclass
class PrepareConfig:
    source_root: str = "corpus/"
    convert_workers: int = 8
    ocr_images: bool = True
    vision_fallback: VisionFallbackConfig = field(default_factory=VisionFallbackConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    summarize: PrepareSummarizeConfig = field(default_factory=PrepareSummarizeConfig)
    rollup: PrepareRollupConfig = field(default_factory=PrepareRollupConfig)
    playbooks: PlaybooksConfig = field(default_factory=PlaybooksConfig)

    @classmethod
    def from_raw(cls, raw: dict | None) -> "PrepareConfig":
        raw = raw or {}
        return cls(
            source_root=raw.get("source_root", "corpus/"),
            convert_workers=raw.get("convert_workers", 8),
            ocr_images=raw.get("ocr_images", True),
            vision_fallback=VisionFallbackConfig(**(raw.get("vision_fallback") or {})),
            audio=AudioConfig(**(raw.get("audio") or {})),
            summarize=PrepareSummarizeConfig(**(raw.get("summarize") or {})),
            rollup=PrepareRollupConfig(**(raw.get("rollup") or {})),
            playbooks=PlaybooksConfig(**raw.get("playbooks", {})),
        )


@dataclass
class PipelineConfig:
    article: ArticleConfig
    corpus: CorpusConfig
    output: OutputConfig = field(default_factory=OutputConfig)
    stage_a: StageAConfig = field(default_factory=StageAConfig)
    stage_b: StageBConfig = field(default_factory=StageBConfig)
    stage_c: StageCConfig = field(default_factory=StageCConfig)
    prepare: PrepareConfig = field(default_factory=PrepareConfig)
    pricing: dict = field(default_factory=dict)
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
            prepare=PrepareConfig.from_raw(raw.get("prepare")),
            pricing=raw.get("pricing", {}),
        )
        config.project_root = str(Path(path).resolve().parent)
        return config

    def resolve_path(self, relative_path: str) -> str:
        """Resolve a path relative to the project root."""
        return str(Path(self.project_root) / relative_path)
