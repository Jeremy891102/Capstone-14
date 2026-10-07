"""Provider construction. Mock, Gemini (Developer or Vertex Express), OpenRouter and OpenAI."""

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
        if cfg.gemini.backend == "vertex_express":
            from video_report.providers.vertex_gemini import VertexGeminiProvider

            return VertexGeminiProvider(cfg.model, cfg.gemini)
        from video_report.providers.gemini import GeminiProvider

        return GeminiProvider(cfg.model, cfg.gemini)
    if cfg.name == "openrouter":
        from video_report.providers.openrouter import OpenRouterProvider

        return OpenRouterProvider(cfg.model, cfg.openrouter)
    if cfg.name == "openai":
        from video_report.providers.openai import OpenAIProvider

        return OpenAIProvider(cfg.model, cfg.openai)
    raise ValueError(f"unknown provider {cfg.name!r}")


def validate_request_for(
    provider_name: str, request: ModelRequest, cfg: ProviderConfig | None = None
) -> None:
    """Reject requests a provider cannot honour, before any call is made."""
    if provider_name == "gemini" and cfg and cfg.gemini.backend == "vertex_express":
        from video_report.providers.vertex_gemini import validate_request

        validate_request(request)
    elif provider_name == "openai":
        from video_report.providers.openai import validate_request as validate_openai

        validate_openai(request, cfg.openai if cfg else None)
    elif provider_name == "gemini":
        from video_report.providers.gemini import validate_request

        validate_request(request)
    elif provider_name == "openrouter":
        from video_report.providers.openrouter import validate_request as validate

        validate(request)


def preflight(cfg: ProviderConfig, environ: dict[str, str] | None = None) -> None:
    """Cheap checks before a run directory is created (no network)."""
    if cfg.name == "gemini" and cfg.gemini.backend == "developer":
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
    if cfg.name in {"openrouter", "openai"} or (
        cfg.name == "gemini" and cfg.gemini.backend == "vertex_express"
    ):
        env = os.environ if environ is None else environ
        for mod in ("httpx", "imageio_ffmpeg"):
            if importlib.util.find_spec(mod) is None:
                raise ValueError(f"provider {cfg.name} needs httpx and imageio-ffmpeg")
        options = getattr(cfg, cfg.name)
        if not env.get(options.api_key_env, "").strip():
            raise ValueError(
                f"provider {cfg.name} needs the environment variable {options.api_key_env}"
            )
