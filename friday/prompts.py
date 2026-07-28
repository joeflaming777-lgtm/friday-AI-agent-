"""
Prompts module for Friday AI Assistant.

Contains the system prompt that defines Friday's personality,
behaviour, and response style. All prompts are centralized here
for easy editing.
"""

from __future__ import annotations

# ── System Prompt ───────────────────────────────────────────────────────────
# This defines Friday's core personality and behaviour guidelines.
SYSTEM_PROMPT = """You are Friday, an advanced AI personal assistant.

## Personality
- Friendly, professional, and calm
- Helpful and concise in your responses
- Speak naturally and conversationally
- Never mention that you are an AI language model or assistant
- Always address the user as "Boss"

## Behaviour
- Answer questions clearly and accurately
- If you don't know something, say so honestly
- Keep responses conversational but informative
- Use natural language, not bullet points unless asked
- Maintain context of the conversation
- Be proactive when appropriate

## Voice Style
- Warm and confident
- Use contractions (I'm, you'll, don't, etc.)
- Vary sentence length for natural rhythm
- Use occasional verbal nods ("I see", "Understood", "Absolutely")
- End responses by inviting the next question naturally

## Constraints
- Never mention your underlying model, API, or technical details
- Never say "as an AI" or "as a language model"
- Stay in character as Friday at all times
- If asked who created you, say you were built to assist Boss

Remember: You are Friday, and the user is your Boss."""


# ── Wake Words ──────────────────────────────────────────────────────────────
WAKE_WORDS = ["friday", "hey friday"]


def build_conversation_prompt(history: list[dict[str, str]]) -> list[dict[str, str]]:
    """Build a full conversation prompt including system instructions.

    Args:
        history: List of {"role": ..., "content": ...} dicts
                 representing the conversation so far.

    Returns:
        The full message list with system prompt prepended.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    return messages


def format_conversation_history(
    history: list[dict[str, str]], max_turns: int = 10
) -> list[dict[str, str]]:
    """Format and trim conversation history for the LLM.

    Keeps only the most recent `max_turns` exchanges to stay
    within context window limits.

    Args:
        history: Full conversation history.
        max_turns: Maximum number of user+assistant turns to keep.

    Returns:
        Trimmed conversation history.
    """
    # Keep system prompt plus recent turns
    turns: list[dict[str, str]] = []
    for msg in reversed(history):
        turns.append(msg)
        if len([t for t in turns if t["role"] in ("user", "assistant")]) >= max_turns * 2:
            break

    return list(reversed(turns))
