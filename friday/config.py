"""
Configuration module for Friday AI Assistant.

Loads and validates all environment variables for LiveKit,
Google Gemini, STT, and TTS services. Provides typed dataclasses
for clean access throughout the application.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (one directory up from this file)
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_env_path)


def _require_env(key: str) -> str:
    """Get a required environment variable or raise."""
    value = os.getenv(key)
    if not value:
        raise ValueError(
            f"Missing required environment variable: {key}\n"
            f"Please set it in the .env file or export it in your shell."
        )
    return value


def _optional_env(key: str, default: str | None = None) -> str | None:
    """Get an optional environment variable."""
    return os.getenv(key, default)


@dataclass(frozen=True)
class LiveKitConfig:
    """LiveKit server connection configuration."""

    url: str = field(default_factory=lambda: _require_env("LIVEKIT_URL"))
    api_key: str = field(default_factory=lambda: _require_env("LIVEKIT_API_KEY"))
    api_secret: str = field(default_factory=lambda: _require_env("LIVEKIT_API_SECRET"))


@dataclass(frozen=True)
class GeminiConfig:
    """Google Gemini API configuration."""

    api_key: str = field(default_factory=lambda: _require_env("GOOGLE_API_KEY"))
    model: str = field(
        default_factory=lambda: _optional_env("GEMINI_MODEL", "gemini-2.0-flash")
    )


@dataclass(frozen=True)
class STTConfig:
    """Speech-to-text configuration."""

    backend: str = field(
        default_factory=lambda: _optional_env("STT_BACKEND", "deepgram")
    )
    deepgram_api_key: str | None = field(
        default_factory=lambda: _optional_env("DEEPGRAM_API_KEY")
    )


@dataclass(frozen=True)
class TTSConfig:
    """Text-to-speech configuration."""

    backend: str = field(
        default_factory=lambda: _optional_env("TTS_BACKEND", "cartesia")
    )
    cartesia_api_key: str | None = field(
        default_factory=lambda: _optional_env("CARTESIA_API_KEY")
    )
    elevenlabs_api_key: str | None = field(
        default_factory=lambda: _optional_env("ELEVENLABS_API_KEY")
    )
    openai_api_key: str | None = field(
        default_factory=lambda: _optional_env("OPENAI_API_KEY")
    )


@dataclass(frozen=True)
class SSLConfig:
    """TLS certificate verification settings.

    Some machines (e.g. with AVG/Avast "Web Shield" TLS interception)
    serve re-signed certificates that Python's OpenSSL rejects. Set
    ``FRIDAY_SSL_VERIFY=false`` to disable certificate verification for
    the STT/TTS/LLM API calls so the assistant can still work.
    """

    verify: bool = field(
        default_factory=lambda: _optional_env("FRIDAY_SSL_VERIFY", "true").lower()
        in ("1", "true", "yes")
    )


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration."""

    livekit: LiveKitConfig = field(default_factory=LiveKitConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    ssl: SSLConfig = field(default_factory=SSLConfig)
    log_level: str = field(default_factory=lambda: _optional_env("LOG_LEVEL", "INFO"))
    sample_rate: int = 16000
    channels: int = 1


def load_config() -> AppConfig:
    """Load and return the validated application configuration.

    Returns:
        AppConfig with all settings populated from environment variables.

    Raises:
        ValueError: If any required environment variable is missing.
    """
    return AppConfig()
