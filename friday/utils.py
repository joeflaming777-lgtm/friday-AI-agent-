"""
Utility functions for Friday AI Assistant.

Provides audio conversion helpers and text processing utilities
used across the application.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
from livekit.rtc import AudioFrame


# ── Audio Conversion ────────────────────────────────────────────────────────

def numpy_to_audio_frame(
    array: np.ndarray,
    sample_rate: int = 16000,
    num_channels: int = 1,
) -> AudioFrame:
    """Convert a numpy array to a LiveKit AudioFrame.

    Args:
        array: 1D (mono) or 2D (multi-channel) int16 numpy array.
        sample_rate: Audio sample rate in Hz (default: 16000).
        num_channels: Number of audio channels (default: 1).

    Returns:
        LiveKit AudioFrame containing the audio data.
    """
    if array.dtype != np.int16:
        array = (array * 32767).astype(np.int16)

    if array.ndim == 1:
        samples_per_channel = array.shape[0]
    else:
        samples_per_channel = array.shape[0]

    return AudioFrame(
        data=array.tobytes(),
        sample_rate=sample_rate,
        num_channels=num_channels,
        samples_per_channel=samples_per_channel,
    )


def audio_frame_to_numpy(frame: AudioFrame) -> np.ndarray:
    """Convert a LiveKit AudioFrame to a numpy array.

    Args:
        frame: LiveKit AudioFrame to convert.

    Returns:
        2D numpy array of shape (samples, channels) with dtype int16.
    """
    array = np.frombuffer(frame.data, dtype=np.int16)
    return array.reshape(-1, frame.num_channels)


async def async_iter_from_list(
    items: list[AudioFrame],
) -> AsyncIterator[AudioFrame]:
    """Create an async iterator from a list of AudioFrames.

    Args:
        items: List of AudioFrame objects to iterate over.

    Yields:
        Each AudioFrame in the list with a small delay to simulate
        real-time audio flow.
    """
    for item in items:
        yield item
        await asyncio.sleep(0)  # Yield control to event loop


# ── Text Processing ─────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Clean and normalize transcribed text.

    Removes leading/trailing whitespace, normalizes multiple spaces,
    and strips incomplete utterances.

    Args:
        text: Raw transcribed text.

    Returns:
        Cleaned text string.
    """
    text = text.strip()
    # Remove trailing incomplete words (heuristic: ends mid-word)
    if text and not text.endswith((".", "?", "!", "...")):
        # Keep it — it's likely a complete utterance even without punctuation
        pass
    return text


def is_wake_word(text: str, wake_words: list[str]) -> bool:
    """Check if the spoken text contains a wake word.

    Args:
        text: Transcribed text to check.
        wake_words: List of wake word phrases to look for.

    Returns:
        True if any wake word is found at the start of the text.
    """
    lower_text = text.lower().strip()
    for word in wake_words:
        if lower_text.startswith(word):
            return True
    return False


def strip_wake_word(text: str, wake_words: list[str]) -> str:
    """Remove the wake word from the beginning of text.

    Args:
        text: Text potentially starting with a wake word.
        wake_words: List of wake word phrases to strip.

    Returns:
        Text with the wake word removed, or original text if no
        wake word was found.
    """
    lower_text = text.lower().strip()
    for word in wake_words:
        if lower_text.startswith(word):
            return text[len(word) :].strip().lstrip(",").strip()
    return text


# ── Timing ──────────────────────────────────────────────────────────────────

class Timer:
    """Simple context manager for timing code blocks."""

    def __init__(self) -> None:
        self.start_time: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.elapsed = time.perf_counter() - self.start_time

    @property
    def seconds(self) -> float:
        """Return elapsed time in seconds."""
        return self.elapsed

    @property
    def milliseconds(self) -> float:
        """Return elapsed time in milliseconds."""
        return self.elapsed * 1000.0
