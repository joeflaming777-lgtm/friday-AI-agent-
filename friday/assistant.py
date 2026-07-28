"""
Friday AI Assistant - Main orchestrator.

Bridges the voice pipeline or text input with the Gemini LLM.
Handles startup, wake word detection, conversation management,
and graceful shutdown.
"""

from __future__ import annotations

import asyncio
import signal
import sys
from collections.abc import AsyncIterator
from typing import Any

from config import AppConfig, load_config
from logger import get_logger
from prompts import WAKE_WORDS
from services import GeminiService, VoicePipeline
from utils import is_wake_word, strip_wake_word

logger = get_logger("friday")


class FridayAssistant:
    """Friday AI Voice Assistant.

    The main orchestrator class that ties together speech
    recognition, LLM inference, and speech synthesis.

    Supports both voice and text interaction modes with
    automatic wake word detection and graceful shutdown.

    Attributes:
        config: Application configuration.
        gemini: Gemini LLM service with conversation memory.
        pipeline: Voice processing pipeline (voice mode only).
        mode: Operating mode ('voice' or 'text').
    """

    def __init__(
        self,
        config: AppConfig | None = None,
        mode: str = "voice",
    ) -> None:
        """Initialize the Friday assistant.

        Args:
            config: Application config. Loaded from env if not provided.
            mode: Operating mode - 'voice' or 'text'.

        Raises:
            ValueError: If an unsupported mode is specified.
        """
        self.config: AppConfig = config or load_config()
        if mode not in ("voice", "text"):
            raise ValueError(f"Unsupported mode: {mode}. Use 'voice' or 'text'.")
        self.mode: str = mode
        self._running: bool = False

        # Initialize Gemini service
        self.gemini: GeminiService = GeminiService(self.config.gemini)

        # Voice pipeline (voice mode only)
        self.pipeline: VoicePipeline | None = None

        # Wake word state
        self._awaiting_wake_word: bool = False

        logger.info(f"Friday initialized (mode: {mode})")

    # ── Public API ──────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Start the assistant and begin the interaction loop.

        Enters the main interaction mode based on the configured
        mode ('voice' or 'text') and runs until shutdown.
        """
        self._running = True

        # Set up signal handlers for graceful shutdown
        self._setup_signal_handlers()

        # Print banner
        logger.print_banner()

        if self.mode == "voice":
            await self._run_voice_mode()
        else:
            await self._run_text_mode()

    async def shutdown(self) -> None:
        """Shut down the assistant gracefully.

        Stops the voice pipeline and cleans up resources.
        """
        if not self._running:
            return

        logger.print_status("Shutting down Friday...")
        self._running = False

        if self.pipeline is not None:
            await self.pipeline.stop()

        logger.print_success("Friday has been shut down. Goodbye Boss!")

    # ── Signal Handling ─────────────────────────────────────────────────────

    def _setup_signal_handlers(self) -> None:
        """Register signal handlers for graceful shutdown."""

        def _handle_signal(sig: int, frame: Any) -> None:
            logger.info(f"Received signal {signal.Signals(sig).name}")
            asyncio.create_task(self.shutdown())

        try:
            signal.signal(signal.SIGINT, _handle_signal)
            signal.signal(signal.SIGTERM, _handle_signal)
        except (ValueError, AttributeError):
            # Windows may not support SIGTERM the same way
            try:
                signal.signal(signal.SIGINT, _handle_signal)
            except (ValueError, AttributeError):
                logger.warning("Signal handlers not fully supported on this platform")

    # ── Voice Mode ──────────────────────────────────────────────────────────

    async def _run_voice_mode(self) -> None:
        """Run the assistant in voice mode.

        Starts the voice pipeline, which captures microphone audio
        and processes it through VAD -> STT -> Gemini -> TTS.
        """
        logger.print_status("Initializing voice pipeline...")

        try:
            self.pipeline = VoicePipeline(
                config=self.config,
                gemini=self.gemini,
                on_user_message=self._on_user_message,
                on_friday_message=self._on_friday_message,
            )

            await self.pipeline.start()
            logger.print_success("Connected and listening")
            logger.print_listening()

            # Keep running until shutdown
            while self._running:
                await asyncio.sleep(0.5)

        except ValueError as exc:
            # Configuration error (missing API key, etc.)
            logger.print_error(str(exc))
            logger.print_status(
                "Falling back to text mode. Type your messages below."
            )
            await self._run_text_mode()

        except OSError as exc:
            # Microphone/sounddevice error
            logger.print_error(f"Microphone unavailable: {exc}")
            logger.print_status(
                "Audio device not found. Falling back to text mode."
            )
            await self._run_text_mode()

        except ImportError as exc:
            logger.print_error(f"Missing dependency: {exc}")
            logger.print_status("Please install required packages: pip install -r requirements.txt")
            await self._run_text_mode()

        except Exception as exc:
            logger.print_error(f"Voice pipeline error: {exc}")
            logger.print_status("Falling back to text mode.")
            await self._run_text_mode()

    def _on_user_message(self, text: str) -> None:
        """Handle a transcribed user message.

        Processes wake word detection and displays the
        transcribed text.

        Args:
            text: Transcribed user speech.
        """
        text = text.strip()
        if not text:
            return

        logger.print_user_message(text)

    def _on_friday_message(self, text: str) -> None:
        """Handle Friday's response message.

        Displays the response text in the terminal.

        Args:
            text: Friday's response text.
        """
        logger.print_friday_message(text)
        logger.print_listening()

    # ── Text Mode ───────────────────────────────────────────────────────────

    async def _run_text_mode(self) -> None:
        """Run the assistant in text/typing mode.

        Reads user input from the terminal and sends it to Gemini.
        Useful as a fallback when audio is unavailable.
        """
        logger.print_status("Text interaction mode")
        print()  # spacing

        while self._running:
            try:
                # Read input asynchronously
                user_input = await self._async_input("You: ")
            except EOFError:
                break
            except asyncio.CancelledError:
                break

            if not user_input:
                continue

            # Check for exit commands
            if user_input.lower() in ("exit", "quit", "bye"):
                logger.print_friday_message("Goodbye Boss!")
                self._running = False
                break

            # Check for wake word
            if self._awaiting_wake_word and not is_wake_word(user_input, WAKE_WORDS):
                logger.print_friday_message("Say 'Friday' to wake me up, Boss.")
                continue

            self._awaiting_wake_word = False

            # Strip wake word if present
            clean_input = strip_wake_word(user_input, WAKE_WORDS)

            # Process through Gemini
            try:
                full_response: list[str] = []
                logger.print_status("Friday is thinking...")

                async for chunk in self.gemini.send_message_stream(clean_input):
                    full_response.append(chunk)

                response_text = "".join(full_response)
                if response_text:
                    logger.print_friday_message(response_text)
                print()

            except Exception as exc:
                logger.print_error(
                    "I'm sorry Boss, I couldn't reach my AI service."
                )
                logger.debug(f"Gemini error: {exc}")

    # ── Wake Word ───────────────────────────────────────────────────────────

    def enable_wake_word(self) -> None:
        """Enable wake word detection.

        When enabled, the assistant will only process speech
        that starts with the configured wake word ("Friday").
        """
        self._awaiting_wake_word = True
        logger.print_status(
            f"Wake word enabled. Say '{WAKE_WORDS[0]}' to start."
        )

    def disable_wake_word(self) -> None:
        """Disable wake word detection.

        When disabled, the assistant processes all speech
        without requiring a wake word.
        """
        self._awaiting_wake_word = False
        logger.print_status("Wake word disabled — always listening.")

    # ── Utilities ───────────────────────────────────────────────────────────

    @staticmethod
    async def _async_input(prompt: str = "") -> str:
        """Read a line of input from stdin asynchronously.

        Args:
            prompt: Optional prompt string to display.

        Returns:
            The user's input string.
        """
        loop = asyncio.get_event_loop()

        def _input() -> str:
            try:
                return input(prompt)
            except EOFError:
                return "exit"
            except KeyboardInterrupt:
                return "exit"

        return await loop.run_in_executor(None, _input)
