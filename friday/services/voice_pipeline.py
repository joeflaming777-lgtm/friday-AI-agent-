"""
Voice pipeline for Friday AI Assistant.

Handles the real-time audio processing pipeline:
  Microphone -> VAD -> STT -> Gemini -> TTS -> Speakers

Manages audio capture, voice activity detection, speech-to-text,
LLM inference, text-to-speech, and audio playback with
interruption support.
"""

from __future__ import annotations

import asyncio
import queue
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from livekit.agents.stt import STT
from livekit.agents.tts import TTS
from livekit.agents.vad import VAD, VADEventType
from livekit.rtc import AudioFrame

from config import AppConfig
from logger import get_logger
from utils import audio_frame_to_numpy, numpy_to_audio_frame
from services.gemini_service import GeminiService
from services.speech_service import create_stt, create_tts, create_vad

logger = get_logger("friday.pipeline")


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
                frame = self._queue.get_nowait()
                return frame
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

    Attributes:
        sample_rate: Playback sample rate in Hz.
        channels: Number of playback channels.
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
        try:
            data = self._queue.get_nowait()
            # Handle shorter/longer data
            if len(data) < frames:
                outdata[: len(data)] = data.reshape(-1, 1)
                outdata[len(data) :].fill(0)
            else:
                outdata[:] = data[:frames].reshape(-1, 1)
            self._is_playing = True
        except queue.Empty:
            outdata.fill(0)
            self._is_playing = False

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

        # Ensure mono
        if arr.ndim > 1 and arr.shape[1] > 1:
            arr = np.mean(arr, axis=1, dtype=np.int16)

        self._queue.put(arr)

    def clear(self) -> None:
        """Clear all queued audio (interrupt playback)."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
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
            data: Input audio array (1D).
            orig_rate: Original sample rate.
            target_rate: Target sample rate.

        Returns:
            Resampled audio array.
        """
        if orig_rate == target_rate:
            return data

        duration = len(data) / orig_rate
        target_length = int(duration * target_rate)
        orig_indices = np.linspace(0, len(data) - 1, target_length)
        return np.interp(
            orig_indices, np.arange(len(data)), data.astype(np.float64)
        ).astype(np.int16)


# ── Callback Types ──────────────────────────────────────────────────────────

MessageCallback = Callable[[str], None]


# ── Voice Pipeline ──────────────────────────────────────────────────────────

@dataclass
class PipelineState:
    """Shared mutable state for the voice pipeline."""

    is_speaking: bool = False
    is_processing: bool = False
    speech_buffer: list[AudioFrame] = field(default_factory=list)
    should_stop: bool = False


class VoicePipeline:
    """Full-duplex voice processing pipeline.

    Orchestrates the complete voice interaction flow:
        Microphone -> VAD detection -> STT transcription
        -> Gemini LLM -> TTS synthesis -> Speaker playback

    Supports interruption: if the user speaks while the assistant
    is responding, playback stops and the new speech is processed.
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
        self._capture: AudioCapture | None = None
        self._playback: AudioPlayback | None = None

        # State
        self._state: PipelineState = PipelineState()
        self._tasks: list[asyncio.Task[Any]] = []
        self._running: bool = False

    async def start(self) -> None:
        """Start the voice pipeline.

        Initializes all components and begins audio capture and
        processing.
        """
        logger.info("Starting voice pipeline...")

        # Initialize components
        try:
            self._vad = create_vad()
            self._stt = create_stt(self.config.stt)
            self._tts = create_tts(self.config.tts)
        except Exception as exc:
            logger.error(f"Failed to initialize voice components: {exc}")
            raise

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
            asyncio.create_task(self._vad_loop()),
            asyncio.create_task(self._capture_loop()),
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

        # Stop audio
        if self._capture is not None:
            self._capture.stop()
        if self._playback is not None:
            self._playback.stop()

        logger.info("Voice pipeline stopped")

    # ── Internal: Capture Loop ──────────────────────────────────────────

    async def _capture_loop(self) -> None:
        """Capture audio frames and feed them to the VAD processor.

        Reads frames from the microphone and pushes them into
        the VAD stream for voice activity detection.
        """
        assert self._vad is not None
        assert self._capture is not None

        vad_stream = self._vad.stream()

        try:
            async for frame in self._capture:
                if self._state.should_stop:
                    break

                # Push frame to VAD
                vad_stream.push_frame(frame)

                # If user is speaking, buffer the frame
                if self._state.is_speaking:
                    self._state.speech_buffer.append(frame)

        except Exception as exc:
            if not self._state.should_stop:
                logger.error(f"Capture loop error: {exc}")
        finally:
            logger.debug("Capture loop ended")

    # ── Internal: VAD Loop ──────────────────────────────────────────────

    async def _vad_loop(self) -> None:
        """Process VAD events from the voice activity detector.

        Handles speech start/end events and triggers speech
        processing.
        """
        assert self._vad is not None

        vad_stream = self._vad.stream()

        try:
            async for event in vad_stream:
                if self._state.should_stop:
                    break

                if event.type == VADEventType.START_OF_SPEECH:
                    await self._on_speech_start()

                elif event.type == VADEventType.END_OF_SPEECH:
                    await self._on_speech_end()

        except Exception as exc:
            if not self._state.should_stop:
                logger.error(f"VAD loop error: {exc}")
        finally:
            logger.debug("VAD loop ended")

    async def _on_speech_start(self) -> None:
        """Handle the start of user speech.

        Resets the speech buffer and interrupts any ongoing
        playback.
        """
        self._state.is_speaking = True
        self._state.speech_buffer = []

        # Interrupt playback if the assistant is speaking
        if self._playback is not None and self._playback.is_playing:
            logger.info("User interrupted — stopping playback")
            self._playback.clear()
            # Cancel any ongoing TTS processing
            if self._state.is_processing:
                await self.gemini.cancel_stream()

    async def _on_speech_end(self) -> None:
        """Handle the end of user speech.

        Captures the buffered audio and starts processing it
        through STT -> Gemini -> TTS.
        """
        self._state.is_speaking = False

        # Get the buffered speech
        buffer = list(self._state.speech_buffer)
        self._state.speech_buffer = []

        if not buffer:
            return

        # Process in background task
        asyncio.create_task(self._process_speech_segment(buffer))

    # ── Internal: Speech Processing ─────────────────────────────────────

    async def _process_speech_segment(self, frames: list[AudioFrame]) -> None:
        """Process a speech segment through the full pipeline.

        Transcribes audio to text, sends to Gemini, and speaks
        the response.

        Args:
            frames: Audio frames containing the speech segment.
        """
        if self._state.is_processing:
            logger.debug("Already processing, skipping...")
            return

        self._state.is_processing = True

        try:
            # Step 1: Transcribe
            text = await self._transcribe(frames)
            if not text:
                logger.debug("No speech recognized")
                return

            # Callback for user message
            if self._on_user_message:
                self._on_user_message(text)

            # Step 2: Get response from Gemini (streaming)
            full_response: list[str] = []
            async for chunk in self.gemini.send_message_stream(text):
                full_response.append(chunk)

            response_text = "".join(full_response)
            if not response_text:
                logger.debug("Empty Gemini response")
                return

            # Callback for Friday message
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
        finally:
            self._state.is_processing = False

    async def _transcribe(self, frames: list[AudioFrame]) -> str | None:
        """Transcribe audio frames to text using STT.

        Args:
            frames: Audio frames to transcribe.

        Returns:
            Transcribed text or None if no speech detected.
        """
        assert self._stt is not None

        async def audio_stream() -> AsyncIterator[AudioFrame]:
            for frame in frames:
                yield frame
                await asyncio.sleep(0)

        try:
            result = await self._stt.recognize(buffer=audio_stream())
            if result.alternatives:
                return result.alternatives[0].text
        except Exception as exc:
            logger.error(f"STT transcription error: {exc}")

        return None

    async def _speak(self, text: str) -> None:
        """Synthesize text to speech and play it.

        Args:
            text: Text to speak.
        """
        spoken = False
        if self._tts is not None and self._playback is not None:
            try:
                async for frame in self._tts.synthesize(text):
                    # Check if interrupted
                    if (
                        self._state.is_speaking
                        or self._state.should_stop
                    ):
                        self._playback.clear()
                        break

                    self._playback.play_frame(frame)
                    spoken = True

            except Exception as exc:
                logger.error(f"Cloud TTS synthesis error: {exc}")

        if not spoken:
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
