"""Provider construction. Two providers in v1: ``mock`` and ``gemini``."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from video_report.config import ProviderConfig
from video_report.providers.base import ModelRequest, Provider


def build_provider(cfg: ProviderConfig, mock_responses_path: Path | None) -> Provider:
    if cfg.name == "mock":
        from video_report.providers.mock import MockProvider

        if mock_responses_path is None:
            raise ValueError("mock provider needs a responses file (provider.mock.responses)")
        return MockProvider(cfg.model, mock_responses_path, cfg.mock)
    if cfg.name == "gemini":
        from video_report.providers.gemini import GeminiProvider

        return GeminiProvider(cfg.model, cfg.gemini)
    raise ValueError(f"unknown provider {cfg.name!r}")


def validate_request_for(provider_name: str, request: ModelRequest) -> None:
    """Reject requests a provider cannot honour, before any call is made."""
    if provider_name == "gemini":
        from video_report.providers.gemini import validate_request

        validate_request(request)


def preflight(cfg: ProviderConfig, environ: dict[str, str] | None = None) -> None:
    """Cheap checks before a run directory is created (no network)."""
    if cfg.name == "gemini":
        env = os.environ if environ is None else environ
        if (
            importlib.util.find_spec("google") is None
            or importlib.util.find_spec("google.genai") is None
        ):
            raise ValueError("provider gemini needs the SDK: pip install -e '.[gemini]'")
        if not env.get(cfg.gemini.api_key_env):
            raise ValueError(
                f"provider gemini needs the environment variable {cfg.gemini.api_key_env}"
            )
