"""
Voice pipeline for Friday AI Assistant.

Handles the real-time audio processing pipeline:
  Microphone -> VAD -> STT -> Gemini -> TTS -> Speakers

Manages audio capture, voice activity detection, speech-to-text,
LLM inference, text-to-speech, and audio playback with
interruption (barge-in) support.
"""

from __future__ import annotations

import asyncio
import queue
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import aiohttp
import numpy as np
from livekit.agents.stt import STT
from livekit.agents.tts import TTS, SynthesizedAudio
from livekit.agents.vad import VAD, VADEventType, VADStream
from livekit.rtc import AudioFrame

from config import AppConfig
from logger import get_logger
from utils import audio_frame_to_numpy
from services.gemini_service import GeminiService
from services.speech_service import (
    build_http_session,
    create_stt,
    create_tts,
    create_vad,
)

logger = get_logger("friday.pipeline")

#: Seconds after Friday starts speaking during which we ignore VAD start
#: events. This prevents Friday's own voice coming back through the
#: microphone from interrupting herself (acoustic echo) while still
#: allowing genuine barge-in a moment later.
ECHO_GUARD_SECONDS = 0.35


# ── Audio Capture ───────────────────────────────────────────────────────────

class AudioCapture:
    """Captures microphone audio in a background thread.

    Uses sounddevice to read from the default microphone and
    delivers LiveKit AudioFrame objects via an async generator.

    Attributes:
        sample_rate: Audio sample rate in Hz.
        channels: Number of audio channels.
        block_size: Frames per audio block.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        block_size: int = 480,
    ) -> None:
        """Initialize audio capture.

        Args:
            sample_rate: Sample rate in Hz (default: 16000).
            channels: Number of channels (default: 1, mono).
            block_size: Frames per block (default: 480 = 30ms @ 16kHz).
        """
        self.sample_rate: int = sample_rate
        self.channels: int = channels
        self.block_size: int = block_size
        self._queue: queue.Queue[AudioFrame | None] = queue.Queue(maxsize=200)
        self._stream: Any = None
        self._running: bool = False

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: Any,
    ) -> None:
        """Sounddevice callback - runs in background thread."""
        if status:
            logger.warning(f"Audio capture status: {status}")
        try:
            frame = AudioFrame(
                data=indata.tobytes(),
                sample_rate=self.sample_rate,
                num_channels=self.channels,
                samples_per_channel=frames,
            )
            self._queue.put_nowait(frame)
        except queue.Full:
            # Drop oldest frame if queue is full
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(frame)
            except queue.Empty:
                pass

    def start(self) -> None:
        """Start capturing audio from the default microphone."""
        import sounddevice as sd

        self._running = True
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self._callback,
            blocksize=self.block_size,
            dtype="int16",
        )
        self._stream.start()
        logger.info(f"Audio capture started ({self.sample_rate}Hz, {self.channels}ch)")

    def stop(self) -> None:
        """Stop audio capture."""
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:
                logger.warning(f"Audio capture stop error: {exc}")
            self._stream = None
        # Signal end to any readers
        self._queue.put_nowait(None)
        logger.info("Audio capture stopped")

    async def read(self) -> AudioFrame | None:
        """Read the next audio frame asynchronously.

        Returns:
            AudioFrame or None if capture has stopped.
        """
        while self._running:
            try:
                return self._queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.005)
        return None

    def __aiter__(self) -> AsyncIterator[AudioFrame]:
        """Iterate over audio frames asynchronously."""
        return self._async_generator()

    async def _async_generator(self) -> AsyncIterator[AudioFrame]:
        """Async generator yielding audio frames."""
        while self._running:
            try:
                frame = self._queue.get_nowait()
                if frame is None:
                    break
                yield frame
            except queue.Empty:
                await asyncio.sleep(0.005)


# ── Audio Playback ──────────────────────────────────────────────────────────

class AudioPlayback:
    """Plays audio through speakers using sounddevice.

    Provides a queue-based playback system with support for
    interruption (clear the queue to stop playback immediately).
    Handles chunks that are longer than a single output block by
    buffering leftover samples between callbacks.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        channels: int = 1,
    ) -> None:
        """Initialize audio playback.

        Args:
            sample_rate: Sample rate in Hz (default: 24000).
            channels: Number of channels (default: 1, mono).
        """
        self.sample_rate: int = sample_rate
        self.channels: int = channels
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: Any = None
        self._running: bool = False
        self._is_playing: bool = False
        # Leftover samples from a chunk that did not fit in one output block
        self._residual: np.ndarray | None = None

    def _callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: Any,
        status: Any,
    ) -> None:
        """Sounddevice callback - runs in background thread."""
        if status:
            logger.warning(f"Audio playback status: {status}")

        written = 0
        remaining = frames
        outdata.fill(0)

        while remaining > 0:
            # Refill residual from the queue when it runs dry
            if self._residual is None:
                try:
                    data = self._queue.get_nowait()
                except queue.Empty:
                    break
                self._residual = self._to_mono_1d(data)

            take = min(len(self._residual), remaining)
            outdata[written : written + take, 0] = self._residual[:take]
            written += take
            remaining -= take

            self._residual = (
                self._residual[take:] if take < len(self._residual) else None
            )

        self._is_playing = written > 0

    @staticmethod
    def _to_mono_1d(arr: np.ndarray) -> np.ndarray:
        """Flatten a (samples, channels) int16 array to mono 1D int16."""
        if arr.ndim > 1:
            if arr.shape[1] > 1:
                return np.mean(arr, axis=1, dtype=np.int16)
            return arr[:, 0]
        return arr

    def start(self) -> None:
        """Start the audio playback stream."""
        import sounddevice as sd

        self._running = True
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self._callback,
            blocksize=960,  # 40ms @ 24kHz
            dtype="int16",
        )
        self._stream.start()
        logger.info(f"Audio playback started ({self.sample_rate}Hz, {self.channels}ch)")

    def stop(self) -> None:
        """Stop audio playback and clear queued data."""
        self._running = False
        self.clear()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:
                logger.warning(f"Audio playback stop error: {exc}")
            self._stream = None
        logger.info("Audio playback stopped")

    def play_frame(self, frame: AudioFrame) -> None:
        """Queue an AudioFrame for playback.

        Handles sample rate conversion if the frame's sample rate
        differs from the playback stream's sample rate.

        Args:
            frame: AudioFrame to play.
        """
        arr = audio_frame_to_numpy(frame)

        # Resample if needed
        if frame.sample_rate != self.sample_rate:
            arr = self._resample(arr, frame.sample_rate, self.sample_rate)

        self._queue.put(arr)

    def clear(self) -> None:
        """Clear all queued audio (interrupt playback)."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._residual = None
        self._is_playing = False

    @property
    def is_playing(self) -> bool:
        """Whether audio is currently playing."""
        return self._is_playing

    @staticmethod
    def _resample(
        data: np.ndarray,
        orig_rate: int,
        target_rate: int,
    ) -> np.ndarray:
        """Resample audio data using linear interpolation.

        Args:
            data: Input audio array (1D or 2D).
            orig_rate: Original sample rate.
            target_rate: Target sample rate.

        Returns:
            Resampled audio array.
        """
        if orig_rate == target_rate:
            return data

        flat = data if data.ndim == 1 else data[:, 0]
        duration = len(flat) / orig_rate
        target_length = int(duration * target_rate)
        orig_indices = np.linspace(0, len(flat) - 1, target_length)
        resampled = np.interp(
            orig_indices, np.arange(len(flat)), flat.astype(np.float64)
        ).astype(np.int16)
        return resampled.reshape(-1, 1)


# ── Callback Types ──────────────────────────────────────────────────────────

MessageCallback = Callable[[str], None]


# ── Voice Pipeline ──────────────────────────────────────────────────────────

@dataclass
class PipelineState:
    """Shared mutable state for the voice pipeline."""

    is_speaking: bool = False
    should_stop: bool = False


class VoicePipeline:
    """Full-duplex voice processing pipeline.

    Orchestrates the complete voice interaction flow:
        Microphone -> VAD detection -> STT transcription
        -> Gemini LLM -> TTS synthesis -> Speaker playback

    Speech segments are detected by the VAD, transcribed, answered by
    Gemini, and spoken aloud. If the user speaks while the assistant is
    responding, playback stops and the new speech is queued and
    processed.
    """

    def __init__(
        self,
        config: AppConfig,
        gemini: GeminiService,
        *,
        on_user_message: MessageCallback | None = None,
        on_friday_message: MessageCallback | None = None,
    ) -> None:
        """Initialize the voice pipeline.

        Args:
            config: Application configuration.
            gemini: Initialized Gemini service.
            on_user_message: Callback for user transcription.
            on_friday_message: Callback for Friday's response.
        """
        self.config: AppConfig = config
        self.gemini: GeminiService = gemini

        # Callbacks
        self._on_user_message: MessageCallback | None = on_user_message
        self._on_friday_message: MessageCallback | None = on_friday_message

        # Components (initialized in start())
        self._vad: VAD | None = None
        self._stt: STT | None = None
        self._tts: TTS | None = None
        self._vad_stream: VADStream | None = None
        self._capture: AudioCapture | None = None
        self._playback: AudioPlayback | None = None
        # Shared HTTP session for STT/TTS plugins
        self._http_session: aiohttp.ClientSession | None = None

        # State
        self._state: PipelineState = PipelineState()
        self._segment_queue: asyncio.Queue[list[AudioFrame]] = asyncio.Queue()
        self._tasks: list[asyncio.Task[Any]] = []
        self._running: bool = False
        self._ignore_speech_until: float = 0.0

    async def start(self) -> None:
        """Start the voice pipeline.

        Initializes all components and begins audio capture and
        processing.
        """
        logger.info("Starting voice pipeline...")

        # Initialize components. The STT/TTS plugins share one aiohttp
        # session so their TLS settings (see FRIDAY_SSL_VERIFY) apply to
        # both and we can close it cleanly on shutdown.
        self._http_session = build_http_session(verify=self.config.ssl.verify)
        try:
            self._vad = create_vad()
            self._stt = create_stt(self.config.stt, http_session=self._http_session)
            self._tts = create_tts(self.config.tts, http_session=self._http_session)
        except Exception as exc:
            await self._close_http_session()
            logger.error(f"Failed to initialize voice components: {exc}")
            raise

        # A single VAD stream is shared by the capture loop (which pushes
        # audio) and the VAD loop (which reads speech events). Using two
        # separate streams would silently break detection.
        self._vad_stream = self._vad.stream()

        # Start audio capture and playback
        self._capture = AudioCapture(
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
        )
        self._playback = AudioPlayback()

        try:
            self._capture.start()
            self._playback.start()
        except Exception as exc:
            logger.error(f"Failed to start audio I/O: {exc}")
            raise

        self._running = True

        # Start concurrent tasks
        self._tasks = [
            asyncio.create_task(self._capture_loop()),
            asyncio.create_task(self._vad_loop()),
            asyncio.create_task(self._segment_worker()),
        ]

        logger.info("Voice pipeline started — listening...")

    async def stop(self) -> None:
        """Stop the voice pipeline gracefully.

        Stops audio capture, playback, and all processing tasks.
        """
        logger.info("Stopping voice pipeline...")
        self._state.should_stop = True
        self._running = False

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        # Close the VAD stream
        if self._vad_stream is not None:
            try:
                await self._vad_stream.aclose()
            except Exception as exc:
                logger.warning(f"VAD stream close error: {exc}")
            self._vad_stream = None

        # Stop audio
        if self._capture is not None:
            self._capture.stop()
        if self._playback is not None:
            self._playback.stop()

        # Close the shared HTTP session
        await self._close_http_session()

        logger.info("Voice pipeline stopped")

    async def _close_http_session(self) -> None:
        """Close the shared HTTP session used by STT/TTS plugins."""
        if self._http_session is not None:
            try:
                await self._http_session.close()
            except Exception as exc:
                logger.warning(f"HTTP session close error: {exc}")
            self._http_session = None

    # ── Public API ──────────────────────────────────────────────────────

    async def say(self, text: str) -> None:
        """Speak a line out loud (used for greetings, etc.)."""
        logger.print_friday_message(text)
        await self._speak(text)

    # ── Internal: Capture Loop ──────────────────────────────────────────

    async def _capture_loop(self) -> None:
        """Capture audio frames and feed them to the VAD processor.

        Reads frames from the microphone and pushes them into the
        shared VAD stream for voice activity detection.
        """
        assert self._capture is not None
        assert self._vad_stream is not None

        try:
            async for frame in self._capture:
                if self._state.should_stop:
                    break
                self._vad_stream.push_frame(frame)
        except Exception as exc:
            if not self._state.should_stop:
                logger.error(f"Capture loop error: {exc}")
        finally:
            # Signal the end of input so the VAD stream winds down cleanly
            try:
                self._vad_stream.end_input()
            except Exception:
                pass
            logger.debug("Capture loop ended")

    # ── Internal: VAD Loop ──────────────────────────────────────────────

    async def _vad_loop(self) -> None:
        """Process VAD events from the voice activity detector.

        Handles speech start/end events and triggers speech
        processing.
        """
        assert self._vad_stream is not None

        try:
            async for event in self._vad_stream:
                if self._state.should_stop:
                    break

                if event.type == VADEventType.START_OF_SPEECH:
                    self._on_speech_start()
                elif event.type == VADEventType.END_OF_SPEECH:
                    self._on_speech_end(event.frames)

        except Exception as exc:
            if not self._state.should_stop:
                logger.error(f"VAD loop error: {exc}")
        finally:
            logger.debug("VAD loop ended")

    # ── Internal: Speech Events ─────────────────────────────────────────

    def _on_speech_start(self) -> None:
        """Handle the start of user speech.

        Interrupts any ongoing playback so the user can barge in.
        """
        # Ignore speech that starts within a few hundred ms of Friday
        # beginning to speak — this is usually her own voice echoing
        # back through the microphone.
        if time.monotonic() < self._ignore_speech_until:
            logger.debug("Ignoring speech start (echo guard)")
            return

        self._state.is_speaking = True

        if self._playback is not None and self._playback.is_playing:
            logger.info("User interrupted — stopping playback")
            self._playback.clear()

    def _on_speech_end(self, frames: list[AudioFrame]) -> None:
        """Handle the end of user speech.

        Queues the captured speech frames for transcription.

        Args:
            frames: Audio frames containing the complete speech segment.
        """
        self._state.is_speaking = False
        if not frames:
            return
        self._segment_queue.put_nowait(frames)

    # ── Internal: Segment Worker ────────────────────────────────────────

    async def _segment_worker(self) -> None:
        """Process queued speech segments one at a time."""
        while not self._state.should_stop:
            try:
                frames = await self._segment_queue.get()
            except asyncio.CancelledError:
                break
            await self._process_speech_segment(frames)

    # ── Internal: Speech Processing ─────────────────────────────────────

    async def _process_speech_segment(self, frames: list[AudioFrame]) -> None:
        """Process a speech segment through the full pipeline.

        Transcribes audio to text, sends to Gemini, and speaks
        the response.

        Args:
            frames: Audio frames containing the speech segment.
        """
        try:
            # Step 1: Transcribe
            text = await self._transcribe(frames)
            if not text:
                logger.debug("No speech recognized")
                return

            if self._on_user_message:
                self._on_user_message(text)

            # Step 2: Get response from Gemini (streaming)
            full_response: list[str] = []
            async for chunk in self.gemini.send_message_stream(text):
                full_response.append(chunk)

            response_text = "".join(full_response).strip()
            if not response_text:
                logger.debug("Empty Gemini response")
                return

            if self._on_friday_message:
                self._on_friday_message(response_text)

            # Step 3: Speak the response
            await self._speak(response_text)

        except asyncio.CancelledError:
            logger.debug("Speech processing cancelled")
        except Exception as exc:
            logger.error(f"Speech processing error: {exc}")
            if self._on_friday_message:
                self._on_friday_message(
                    "I'm sorry Boss, I couldn't reach my AI service."
                )

    async def _transcribe(self, frames: list[AudioFrame]) -> str | None:
        """Transcribe audio frames to text using STT.

        Args:
            frames: Audio frames to transcribe.

        Returns:
            Transcribed text or None if no speech detected.
        """
        assert self._stt is not None

        try:
            result = await self._stt.recognize(buffer=frames)
            if result.alternatives:
                return result.alternatives[0].text
        except Exception as exc:
            logger.error(f"STT transcription error: {exc}")

        return None

    async def _speak(self, text: str) -> None:
        """Synthesize text to speech and play it.

        Uses the configured cloud TTS when available, otherwise
        falls back to the local pyttsx3 voice engine.

        Args:
            text: Text to speak.
        """
        spoken = False
        if self._tts is not None and self._playback is not None:
            try:
                async for item in self._tts.synthesize(text):
                    # Check if the user started speaking (barge-in)
                    if self._state.is_speaking or self._state.should_stop:
                        self._playback.clear()
                        break

                    self._playback.play_frame(item.frame)
                    spoken = True
                    # Suppress VAD echo triggers for the first slice of
                    # Friday's own audio on the speakers.
                    self._ignore_speech_until = time.monotonic() + ECHO_GUARD_SECONDS

            except Exception as exc:
                logger.error(f"Cloud TTS synthesis error: {exc}")

        if not spoken and text:
            logger.info("Speaking via local pyttsx3 voice engine...")
            loop = asyncio.get_event_loop()

            def _say() -> None:
                try:
                    import pyttsx3
                    engine = pyttsx3.init()
                    engine.say(text)
                    engine.runAndWait()
                except Exception as e:
                    logger.error(f"pyttsx3 speech error: {e}")

            await loop.run_in_executor(None, _say)

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        """Whether the pipeline is currently running."""
        return self._running

    @property
    def is_speaking(self) -> bool:
        """Whether the user is currently speaking."""
        return self._state.is_speaking
