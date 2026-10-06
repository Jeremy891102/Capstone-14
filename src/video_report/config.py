"""Experiment configuration (one YAML file per experiment directory).

Relative paths in a config file are resolved against the directory containing that file, so an
experiment is self-contained and independent of the current working directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from video_report.benchmarks.schema import Id


class ConfigError(ValueError):
    """The experiment configuration is invalid."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(_Strict):
    benchmark: str  # benchmark directory containing manifest.json
    report_ids: list[str] | None = None  # None = all reports
    field_ids: list[str] | None = None  # None = all fields of each selected report
    video_root: str | None = None  # see datasets.base.choose_video_root for precedence
    require_local_videos: bool = False  # fail run creation if a local video file is missing


class MethodConfig(_Strict):
    name: Literal["whole_report", "per_field"]


class PromptConfig(_Strict):
    system: str | None = None  # path to a template file
    user: str  # path to a template file
    # Whitelist of report.input.context keys rendered into the prompt. Nothing else from the
    # report (source, metadata, annotation refs) is ever rendered.
    context_keys: list[str] = Field(default_factory=list)


FailureKind = Literal["retryable", "permanent", "timeout"]


class MockProviderConfig(_Strict):
    # Canned answers keyed by report/field; defaults to the benchmark's mock_responses_file.
    responses: str | None = None
    # Testing aids (fault injection). All default to off.
    delay_s: float = Field(default=0.0, ge=0)
    fail_plan: dict[str, list[FailureKind]] = Field(default_factory=dict)


class GeminiProviderConfig(_Strict):
    api_key_env: str = "GEMINI_API_KEY"
    file_poll_interval_s: float = Field(default=2.0, gt=0)
    file_active_timeout_s: float = Field(default=300.0, gt=0)
    delete_uploaded_files: bool = True


class OpenRouterProviderConfig(_Strict):
    api_key_env: str = "OPENROUTER_API_KEY"
    base_url: str = "https://openrouter.ai/api/v1"
    clip_height: int = Field(default=480, gt=0)  # clips are downscaled to this height
    clip_fps: float = Field(default=2.0, gt=0)  # used when video.fps is null


class ProviderConfig(_Strict):
    name: Literal["mock", "gemini", "openrouter"]
    model: str = Field(min_length=1)  # always explicit; never a presumed "latest" default
    mock: MockProviderConfig = Field(default_factory=MockProviderConfig)
    gemini: GeminiProviderConfig = Field(default_factory=GeminiProviderConfig)
    openrouter: OpenRouterProviderConfig = Field(default_factory=OpenRouterProviderConfig)


class GenerationConfig(_Strict):
    temperature: float | None = Field(default=0.0, ge=0)
    top_p: float | None = Field(default=None, gt=0, le=1)
    max_output_tokens: int | None = Field(default=None, gt=0)
    seed: int | None = None
    response_format: Literal["json", "text"] = "json"


class VideoConfig(_Strict):
    # Frames per second sampled by the provider. None = provider default.
    fps: float | None = Field(default=None, gt=0, le=24)
    # Send report time ranges as clip offsets. If False, time ranges are dropped *explicitly*
    # (recorded in the frozen config) and whole videos are sent.
    use_time_range: bool = True


class ExecutionConfig(_Strict):
    """Scheduling only. May be overridden on resume; excluded from the compatibility check."""

    concurrency: int = Field(default=4, ge=1, le=64)
    timeout_s: float = Field(default=300.0, gt=0)
    max_attempts: int = Field(default=3, ge=1, le=10)  # per call, per session (includes first)
    backoff_initial_s: float = Field(default=2.0, ge=0)
    backoff_max_s: float = Field(default=60.0, ge=0)
    backoff_jitter: float = Field(default=0.25, ge=0, le=1)


class PricingConfig(_Strict):
    currency: str = "USD"
    input_per_million: float = Field(ge=0)
    output_per_million: float = Field(ge=0)  # applied to output + reasoning tokens
    cached_input_per_million: float | None = Field(default=None, ge=0)
    source: str  # where the prices came from, and the date checked


class ExperimentConfig(_Strict):
    name: Id
    description: str = ""
    data: DataConfig
    method: MethodConfig
    prompt: PromptConfig
    provider: ProviderConfig
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    pricing: PricingConfig | None = None

    @model_validator(mode="after")
    def _check(self) -> ExperimentConfig:
        if self.provider.model.strip().upper().startswith("SET-ME"):
            raise ValueError("provider.model is the SET-ME placeholder; set a real model id")
        if self.provider.name != "mock" and self.provider.mock != MockProviderConfig():
            raise ValueError("provider.mock settings are only valid with provider.name: mock")
        return self

    def inference_view(self) -> dict[str, Any]:
        """Everything that affects model inputs or outputs (used for resume compatibility)."""
        data = self.model_dump(mode="json")
        data.pop("execution")
        data.pop("description")
        data.pop("pricing")  # pricing only affects evaluation
        return data


def load_config(path: Path) -> tuple[ExperimentConfig, str]:
    """Return the parsed config and the exact source text (for freezing)."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from exc
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    try:
        return ExperimentConfig.model_validate(raw), text
    except ValidationError as exc:
        msgs = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
        raise ConfigError(f"{path}: {msgs}") from exc


def resolve_relative(base_dir: Path, value: str) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else (base_dir / p).resolve()
