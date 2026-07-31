"""
Gemini service for Friday AI Assistant.

Provides integration with Google's Gemini REST API for conversational AI,
with conversation memory, streaming responses, and error handling.

Uses httpx directly (instead of the deprecated ``google-generativeai`` SDK)
so that TLS certificate verification can be controlled via the
``FRIDAY_SSL_VERIFY`` setting — required on machines where antivirus
software re-signs HTTPS traffic with a root CA that OpenSSL rejects.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from config import GeminiConfig
from logger import get_logger
from prompts import SYSTEM_PROMPT

logger = get_logger("friday.gemini")

_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
_DEFAULT_TIMEOUT = httpx.Timeout(60.0, connect=15.0, read=120.0)


class GeminiService:
    """Gemini API service with conversation memory.

    Handles all communication with Google's Gemini REST API,
    maintains conversation history, and supports streaming responses.

    Attributes:
        model_name: The Gemini model identifier.
        history: Full conversation history as role/parts dicts.
    """

    def __init__(self, config: GeminiConfig, *, verify: bool = True) -> None:
        """Initialize the Gemini service.

        Args:
            config: Gemini configuration with API key and model name.
            verify: Whether to verify TLS certificates (set False when
                FRIDAY_SSL_VERIFY=false because of TLS interception).
        """
        self.api_key: str = config.api_key
        self.model_name: str = config.model
        self._generation_config: dict[str, Any] = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }
        # Conversation memory: list of {"role": "user"|"model", "parts": [...]}
        # where each part is {"text": "..."}.
        self.history: list[dict[str, Any]] = []
        self._client: httpx.AsyncClient = httpx.AsyncClient(
            timeout=_DEFAULT_TIMEOUT,
            verify=verify,
            headers={"Content-Type": "application/json"},
        )
        logger.info(f"Gemini REST model initialized: {self.model_name}")

    # ── Public API ──────────────────────────────────────────────────────────

    async def send_message(self, text: str) -> str:
        """Send a message and receive a complete response.

        Maintains conversation history for context.

        Args:
            text: The user's message text.

        Returns:
            The model's response text.

        Raises:
            RuntimeError: If the Gemini API call fails.
        """
        self.history.append({"role": "user", "parts": [{"text": text}]})
        url = self._endpoint("generateContent")

        try:
            response = await self._client.post(url, json=self._build_payload())
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            self.history.pop()
            logger.error(f"Gemini API error: {exc}")
            raise RuntimeError(f"Gemini API request failed: {exc}") from exc

        self._raise_if_error(data)
        response_text = self._extract_text(data).strip()
        self.history.append({"role": "model", "parts": [{"text": response_text}]})
        logger.info("Gemini response received")
        return response_text

    async def send_message_stream(self, text: str) -> AsyncIterator[str]:
        """Send a message and stream the response chunks.

        Yields text chunks as they are received from the API, and stores
        the complete response in conversation history.

        Args:
            text: The user's message text.

        Yields:
            Text chunks from the streaming response.
        """
        self.history.append({"role": "user", "parts": [{"text": text}]})
        url = self._endpoint("streamGenerateContent", stream=True)

        full_response: list[str] = []
        completed = False

        try:
            async with self._client.stream(
                "POST", url, json=self._build_payload()
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    try:
                        data = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
                    chunk_text = self._extract_text(data)
                    if chunk_text:
                        full_response.append(chunk_text)
                        yield chunk_text
            completed = True
        except httpx.HTTPError as exc:
            logger.error(f"Gemini streaming error: {exc}")
            raise RuntimeError(f"Gemini API request failed: {exc}") from exc
        finally:
            # If the stream was interrupted (error or caller stopped early),
            # drop the unanswered user message so history stays consistent.
            if not completed and self.history and self.history[-1]["role"] == "user":
                self.history.pop()

        if completed:
            complete = "".join(full_response)
            self.history.append({"role": "model", "parts": [{"text": complete}]})
            logger.info("Gemini streaming response complete")

    async def cancel_stream(self) -> None:
        """Cancel any ongoing streaming request."""
        # httpx closes the stream automatically when the generator that
        # iterates it is cancelled, so there is nothing extra to do here.
        logger.debug("Stream cancellation requested")

    def clear_history(self) -> None:
        """Clear conversation history and start fresh."""
        self.history.clear()
        logger.info("Conversation history cleared")

    def truncate_history(self, max_turns: int = 20) -> None:
        """Trim conversation history to prevent context overflow.

        Args:
            max_turns: Maximum number of user turns to keep.
        """
        user_turns = [m for m in self.history if m["role"] == "user"]
        if len(user_turns) <= max_turns:
            return

        excess = len(user_turns) - max_turns
        removed = 0
        while removed < excess and self.history:
            for i, msg in enumerate(self.history):
                if msg["role"] == "user":
                    self.history.pop(i)
                    removed += 1
                    # Remove the corresponding model response if it exists
                    if i < len(self.history) and self.history[i]["role"] == "model":
                        self.history.pop(i)
                    break
        logger.info(f"History truncated to {max_turns} turns")

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        try:
            await self._client.aclose()
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Gemini client close error: {exc}")

    # ── Internal Helpers ────────────────────────────────────────────────────

    def _endpoint(self, method: str, *, stream: bool = False) -> str:
        """Build the REST endpoint URL for the configured model."""
        params = "&alt=sse" if stream else ""
        return (
            f"{_API_BASE_URL}/models/{self.model_name}:{method}"
            f"?key={self.api_key}{params}"
        )

    def _build_payload(self) -> dict[str, Any]:
        """Build the request body, coalescing consecutive same-role turns."""
        contents: list[dict[str, Any]] = []
        for msg in self.history:
            role, parts = msg["role"], msg["parts"]
            if contents and contents[-1]["role"] == role:
                contents[-1]["parts"].extend(parts)
            else:
                contents.append({"role": role, "parts": list(parts)})

        return {
            "contents": contents,
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "generationConfig": {
                "temperature": self._generation_config["temperature"],
                "topP": self._generation_config["top_p"],
                "topK": self._generation_config["top_k"],
                "maxOutputTokens": self._generation_config["max_output_tokens"],
            },
        }

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        """Extract the response text from a Gemini API response dict."""
        text_parts: list[str] = []
        for candidate in data.get("candidates", []):
            content = candidate.get("content") or {}
            for part in content.get("parts", []):
                if "text" in part:
                    text_parts.append(part["text"])
        if text_parts:
            return "".join(text_parts)
        return data.get("text", "")

    @staticmethod
    def _raise_if_error(data: dict[str, Any]) -> None:
        """Raise a clear error if the API response contains an error body."""
        if "error" in data:
            error = data["error"]
            message = error.get("message") or str(error)
            raise RuntimeError(f"Gemini API error: {message}")
