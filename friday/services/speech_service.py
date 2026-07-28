"""
Speech service factory for Friday AI Assistant.

Provides factory functions to create STT (Speech-to-Text) and
TTS (Text-to-Speech) instances based on configuration.
Supports multiple backends with a consistent interface.
"""

from __future__ import annotations

from typing import Any

from livekit.agents.stt import STT
from livekit.agents.tts import TTS
from livekit.agents.vad import VAD

from config import STTConfig, TTSConfig
from logger import get_logger

logger = get_logger("friday.speech")


def create_vad() -> VAD:
    """Create a Voice Activity Detection instance.

    Uses Silero VAD which runs locally and requires no API key.

    Returns:
        Configured VAD instance.
    """
    from livekit.plugins import silero

    logger.info("VAD initialized: Silero (local)")
    return silero.VAD()


def create_stt(config: STTConfig) -> STT:
    """Create a Speech-to-Text instance based on configuration.

    Args:
        config: STT configuration specifying backend and API keys.

    Returns:
        Configured STT instance.

    Raises:
        ValueError: If the specified backend is unsupported or
                    missing required API keys.
    """
    backend = config.backend.lower()

    if backend == "deepgram":
        if not config.deepgram_api_key:
            raise ValueError(
                "DEEPGRAM_API_KEY is required when STT_BACKEND=deepgram.\n"
                "Get a key at https://console.deepgram.com/"
            )
        from livekit.plugins import deepgram

        logger.info("STT initialized: Deepgram")
        return deepgram.STT(api_key=config.deepgram_api_key)

    elif backend == "google":
        # Uses Google Cloud Speech-to-Text
        # Requires google-cloud-speech and GOOGLE_APPLICATION_CREDENTIALS
        try:
            from livekit.plugins import google

            logger.info("STT initialized: Google Cloud Speech-to-Text")
            return google.stt.STT()
        except ImportError:
            raise ValueError(
                "Google STT requires livekit-plugins-google package.\n"
                "Install with: pip install livekit-plugins-google\n"
                "Also set GOOGLE_APPLICATION_CREDENTIALS in your env."
            )

    else:
        raise ValueError(
            f"Unsupported STT backend: {backend}. "
            f"Supported options: deepgram, google"
        )


def create_tts(config: TTSConfig) -> TTS:
    """Create a Text-to-Speech instance based on configuration.

    Args:
        config: TTS configuration specifying backend and API keys.

    Returns:
        Configured TTS instance.

    Raises:
        ValueError: If the specified backend is unsupported or
                    missing required API keys.
    """
    backend = config.backend.lower()

    if backend == "cartesia":
        if not config.cartesia_api_key:
            raise ValueError(
                "CARTESIA_API_KEY is required when TTS_BACKEND=cartesia.\n"
                "Get a key at https://play.cartesia.ai/"
            )
        from livekit.plugins import cartesia

        logger.info("TTS initialized: Cartesia")
        return cartesia.TTS(api_key=config.cartesia_api_key)

    elif backend == "elevenlabs":
        if not config.elevenlabs_api_key:
            raise ValueError(
                "ELEVENLABS_API_KEY is required when TTS_BACKEND=elevenlabs.\n"
                "Get a key at https://elevenlabs.io/"
            )
        from livekit.plugins import elevenlabs

        logger.info("TTS initialized: ElevenLabs")
        return elevenlabs.TTS(api_key=config.elevenlabs_api_key)

    elif backend == "openai":
        if not config.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when TTS_BACKEND=openai.\n"
                "Get a key at https://platform.openai.com/"
            )
        from livekit.plugins import openai

        logger.info("TTS initialized: OpenAI TTS")
        return openai.TTS(api_key=config.openai_api_key)

    elif backend == "google":
        # Uses Google Cloud Text-to-Speech
        try:
            from livekit.plugins import google

            logger.info("TTS initialized: Google Cloud Text-to-Speech")
            return google.tts.TTS()
        except ImportError:
            raise ValueError(
                "Google TTS requires livekit-plugins-google package.\n"
                "Install with: pip install livekit-plugins-google"
            )

    else:
        raise ValueError(
            f"Unsupported TTS backend: {backend}. "
            f"Supported options: cartesia, elevenlabs, openai, google"
        )


def list_available_backends() -> dict[str, list[str]]:
    """List all available STT and TTS backends.

    Returns:
        Dictionary with "stt" and "tts" keys listing backend names.
    """
    return {
        "stt": ["deepgram", "google"],
        "tts": ["cartesia", "elevenlabs", "openai", "google"],
    }
