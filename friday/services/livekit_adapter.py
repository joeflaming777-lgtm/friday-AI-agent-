"""
LiveKit LLM adapter for Google Gemini.

Provides a GeminiLLM class that implements the livekit.agents.llm.LLM
interface, enabling Google Gemini to be used as the language model
within LiveKit's VoicePipelineAgent.

This is used exclusively in LiveKit worker mode.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import google.generativeai as genai

from logger import get_logger
from prompts import SYSTEM_PROMPT

logger = get_logger("friday.gemini.adapter")

# Try importing LiveKit LLM types (only in worker mode)
try:
    from livekit.agents.llm import (
        LLM,
        LLMStream,
        ChatContext,
        ChatMessage,
        ChatChunk,
        ChoiceDelta,
    )
    _HAS_LIVEKIT_LLM = True
except ImportError:
    _HAS_LIVEKIT_LLM = False
    # Placeholder for type checking
    LLM = object  # type: ignore
    LLMStream = object  # type: ignore


class GeminiLLM(LLM):
    """LiveKit LLM adapter for Google Gemini.

    Implements the LiveKit LLM interface to allow Gemini to
    be used as the intelligence backend for VoicePipelineAgent.

    Attributes:
        model_name: The Gemini model identifier.
        system_instruction: System prompt for conversation context.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gemini-2.0-flash",
        system_instruction: str | None = None,
    ) -> None:
        """Initialize the Gemini LLM adapter.

        Args:
            api_key: Google Gemini API key.
            model: Gemini model name (default: gemini-2.0-flash).
            system_instruction: Custom system prompt. Uses default
                                Friday prompt if not provided.
        """
        genai.configure(api_key=api_key)
        self.model_name: str = model
        self.system_instruction: str = (
            system_instruction or SYSTEM_PROMPT
        )
        logger.info(f"GeminiLLM adapter initialized: {model}")

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        fnc_ctx: Any = None,
        temperature: float | None = None,
        n: int | None = 1,
        **kwargs: Any,
    ) -> LLMStream:
        """Create a chat completion stream.

        Args:
            chat_ctx: The conversation context.
            fnc_ctx: Optional function context (not supported).
            temperature: Response temperature (0.0 - 1.0).
            n: Number of responses (not supported, kept at 1).
            **kwargs: Additional arguments.

        Returns:
            An LLMStream yielding response chunks.
        """
        return GeminiLLMStream(
            chat_ctx=chat_ctx,
            model_name=self.model_name,
            system_instruction=self.system_instruction,
            temperature=temperature,
        )


class GeminiLLMStream(LLMStream):
    """Streaming response from Gemini for LiveKit integration.

    Converts Gemini's streaming response into LiveKit's
    ChatChunk format for use with VoicePipelineAgent.
    """

    def __init__(
        self,
        *,
        chat_ctx: ChatContext,
        model_name: str,
        system_instruction: str | None,
        temperature: float | None,
    ) -> None:
        """Initialize the Gemini stream.

        Args:
            chat_ctx: Conversation context.
            model_name: Gemini model name.
            system_instruction: System prompt.
            temperature: Response temperature.
        """
        super().__init__(chat_ctx=chat_ctx)
        self._model_name: str = model_name
        self._system_instruction: str | None = system_instruction
        self._temperature: float | None = temperature
        self._queue: asyncio.Queue[ChatChunk | None] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> "GeminiLLMStream":
        """Enter async context and start generation."""
        self._task = asyncio.create_task(self._generate())
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit async context and clean up."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _stream_chunks(self) -> AsyncIterator[ChatChunk]:
        """Yield response chunks from the queue."""
        while True:
            chunk = await self._queue.get()
            if chunk is None:
                break
            yield chunk

    async def _generate(self) -> None:
        """Generate response from Gemini and queue chunks."""
        try:
            # Extract user message and history from chat context
            user_message, history = self._prepare_messages()

            if not user_message:
                self._queue.put_nowait(None)
                return

            # Configure the model and generate
            model = genai.GenerativeModel(
                self._model_name,
                system_instruction=self._system_instruction,
            )

            chat = model.start_chat(history=history)

            # Stream the response
            response = await chat.send_message_async(
                user_message, stream=True
            )

            async for chunk in response:
                if chunk.text:
                    self._queue.put_nowait(
                        ChatChunk(
                            choices=[ChoiceDelta(index=0, delta=chunk.text)]
                        )
                    )

        except Exception as exc:
            logger.error(f"Gemini generation error: {exc}")
        finally:
            self._queue.put_nowait(None)

    def _prepare_messages(self) -> tuple[str, list[dict[str, Any]]]:
        """Extract user message and history from LiveKit ChatContext.

        Converts LiveKit message format to Gemini's expected
        alternating user/model format.

        Returns:
            Tuple of (current_user_message, history_list).
        """
        history: list[dict[str, Any]] = []
        last_user_text: str | None = None

        for msg in self.chat_ctx.messages:
            if msg.role == "system":
                continue  # Handled as system_instruction
            elif msg.role == "user":
                if last_user_text is not None:
                    # Previous user without response — add it
                    history.append({"role": "user", "parts": [last_user_text]})
                last_user_text = msg.text
            elif msg.role == "assistant":
                if last_user_text is not None:
                    history.append({"role": "user", "parts": [last_user_text]})
                    last_user_text = None
                history.append({"role": "model", "parts": [msg.text]})

        current_user = last_user_text or ""
        return current_user, history
