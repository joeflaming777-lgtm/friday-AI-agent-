"""
Gemini service for Friday AI Assistant.

Provides integration with Google's Gemini API for conversational AI,
with conversation memory, streaming responses, and error handling.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import google.generativeai as genai

from config import GeminiConfig
from logger import get_logger
from prompts import SYSTEM_PROMPT

logger = get_logger("friday.gemini")


class GeminiService:
    """Gemini API service with conversation memory.

    Handles all communication with Google's Gemini model,
    maintains conversation history, and supports streaming responses.

    Attributes:
        model_name: The Gemini model identifier.
        history: Full conversation history as role/content dicts.
    """

    def __init__(self, config: GeminiConfig) -> None:
        """Initialize the Gemini service.

        Args:
            config: Gemini configuration with API key and model name.
        """
        genai.configure(api_key=config.api_key)
        self.model_name: str = config.model
        self._generation_config: dict[str, Any] = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }
        # Conversation memory: list of {"role": ..., "parts": [...]}
        self.history: list[dict[str, Any]] = []
        # System instruction
        self._system_instruction: str | None = SYSTEM_PROMPT
        # Initialize the model
        self._model: genai.GenerativeModel = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=self._generation_config,
            system_instruction=self._system_instruction,
        )
        # Chat session for turn-based conversation
        self._chat: genai.ChatSession | None = None
        logger.info(f"Gemini model initialized: {self.model_name}")

    def _ensure_chat(self) -> genai.ChatSession:
        """Get or create a chat session, preserving history.

        Returns:
            An active Gemini ChatSession with loaded history.
        """
        if self._chat is None:
            # Convert saved history to Gemini format
            gemini_history: list[dict] = []
            for msg in self.history:
                role = "user" if msg["role"] == "user" else "model"
                gemini_history.append({
                    "role": role,
                    "parts": msg["parts"],
                })
            self._chat = self._model.start_chat(history=gemini_history)
        return self._chat

    async def send_message(
        self, text: str
    ) -> str:
        """Send a message and receive a complete response.

        Maintains conversation history for context.

        Args:
            text: The user's message text.

        Returns:
            The model's response text.

        Raises:
            Exception: If the Gemini API call fails.
        """
        try:
            # Store user message in history
            self.history.append({"role": "user", "parts": [text]})

            chat = self._ensure_chat()

            # Run the synchronous API call in a thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: chat.send_message(text),
            )

            response_text = response.text

            # Store assistant response in history
            self.history.append({"role": "model", "parts": [response_text]})

            logger.info("Gemini response received")
            return response_text

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

    async def send_message_stream(
        self, text: str
    ) -> AsyncIterator[str]:
        """Send a message and stream the response chunks.

        Yields text chunks as they are received from the API,
        and stores the complete response in conversation history.

        Args:
            text: The user's message text.

        Yields:
            Text chunks from the streaming response.

        Raises:
            Exception: If the Gemini API call fails.
        """
        try:
            # Store user message in history
            self.history.append({"role": "user", "parts": [text]})

            chat = self._ensure_chat()

            # Run synchronous streaming in thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: chat.send_message(text, stream=True),
            )

            full_response: list[str] = []

            for chunk in response:
                if chunk.text:
                    chunk_text = chunk.text
                    full_response.append(chunk_text)
                    yield chunk_text
                    await asyncio.sleep(0)  # Yield control

            # Store the complete response in history
            complete = "".join(full_response)
            self.history.append({"role": "model", "parts": [complete]})

            logger.info("Gemini streaming response complete")

        except Exception as e:
            logger.error(f"Gemini streaming error: {e}")
            raise

    async def cancel_stream(self) -> None:
        """Cancel any ongoing streaming request."""
        # Note: google-generativeai doesn't support true cancellation
        # of in-flight requests. This is a best-effort placeholder.
        logger.debug("Stream cancellation requested")

    def clear_history(self) -> None:
        """Clear conversation history and start fresh."""
        self.history.clear()
        self._chat = None
        logger.info("Conversation history cleared")

    def truncate_history(self, max_turns: int = 20) -> None:
        """Trim conversation history to prevent context overflow.

        Args:
            max_turns: Maximum number of user+assistant turns to keep.
        """
        # Count user turns
        user_turns = [m for m in self.history if m["role"] == "user"]
        if len(user_turns) > max_turns:
            # Remove oldest turns
            excess = len(user_turns) - max_turns
            removed = 0
            while removed < excess and self.history:
                # Find and remove oldest user+model pair
                for i, msg in enumerate(self.history):
                    if msg["role"] == "user":
                        self.history.pop(i)
                        removed += 1
                        # Remove corresponding model response if exists
                        if i < len(self.history) and self.history[i]["role"] == "model":
                            self.history.pop(i)
                        break
            # Reset chat session
            self._chat = None
            logger.info(f"History truncated to {max_turns} turns")
