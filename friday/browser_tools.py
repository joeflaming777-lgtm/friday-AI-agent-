"""
Browser tools for Friday AI Assistant.

Provides functions to open URLs and websites in the default browser,
including smart URL resolution for popular services like YouTube,
Google, Spotify, GitHub, etc.
"""

from __future__ import annotations

import re
import urllib.parse
import webbrowser

from logger import get_logger

logger = get_logger("friday.browser")

# ── Known site shortcuts ─────────────────────────────────────────────────────

_SITE_MAP: dict[str, str] = {
    "youtube":      "https://www.youtube.com",
    "google":       "https://www.google.com",
    "gmail":        "https://mail.google.com",
    "maps":         "https://maps.google.com",
    "google maps":  "https://maps.google.com",
    "spotify":      "https://open.spotify.com",
    "netflix":      "https://www.netflix.com",
    "github":       "https://github.com",
    "reddit":       "https://www.reddit.com",
    "twitter":      "https://twitter.com",
    "x":            "https://x.com",
    "facebook":     "https://www.facebook.com",
    "instagram":    "https://www.instagram.com",
    "whatsapp":     "https://web.whatsapp.com",
    "wikipedia":    "https://www.wikipedia.org",
    "amazon":       "https://www.amazon.com",
    "chatgpt":      "https://chat.openai.com",
    "openai":       "https://chat.openai.com",
    "linkedin":     "https://www.linkedin.com",
    "stackoverflow": "https://stackoverflow.com",
    "stack overflow": "https://stackoverflow.com",
}

# ── YouTube search / play helpers ────────────────────────────────────────────

def _youtube_search_url(query: str) -> str:
    """Return a YouTube search URL for the given query."""
    encoded = urllib.parse.quote_plus(query)
    return f"https://www.youtube.com/results?search_query={encoded}"


def _google_search_url(query: str) -> str:
    """Return a Google search URL for the given query."""
    encoded = urllib.parse.quote_plus(query)
    return f"https://www.google.com/search?q={encoded}"


# ── Public API ───────────────────────────────────────────────────────────────

def open_url(url: str) -> str:
    """Open a URL in the default browser.

    Args:
        url: The URL to open.

    Returns:
        A human-readable confirmation message.
    """
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    logger.info(f"Opening URL: {url}")
    webbrowser.open(url)
    return f"Opening {url} in your browser, Boss."


def open_site(site_name: str) -> str:
    """Open a well-known website by name.

    Args:
        site_name: Name of the site (e.g. 'YouTube', 'Google').

    Returns:
        A human-readable confirmation message.
    """
    key = site_name.lower().strip()
    url = _SITE_MAP.get(key)

    if url:
        logger.info(f"Opening site '{site_name}': {url}")
        webbrowser.open(url)
        return f"Opening {site_name.title()} for you, Boss."

    # Fall back to a Google search for unknown sites
    search_url = _google_search_url(site_name)
    logger.info(f"Unknown site '{site_name}', falling back to Google search: {search_url}")
    webbrowser.open(search_url)
    return f"I don't have a direct link for {site_name}, so I've searched Google for it, Boss."


def play_on_youtube(query: str) -> str:
    """Search YouTube for a video and open the results.

    Args:
        query: Song, video, or channel name to search for.

    Returns:
        A human-readable confirmation message.
    """
    url = _youtube_search_url(query)
    logger.info(f"YouTube search for '{query}': {url}")
    webbrowser.open(url)
    return f"Searching YouTube for '{query}', Boss."


def google_search(query: str) -> str:
    """Open a Google search for a query.

    Args:
        query: The search query.

    Returns:
        A human-readable confirmation message.
    """
    url = _google_search_url(query)
    logger.info(f"Google search for '{query}': {url}")
    webbrowser.open(url)
    return f"Searching Google for '{query}', Boss."


# ── Action Parser ────────────────────────────────────────────────────────────

# Regex patterns to extract OPEN_BROWSER actions from Gemini's response text.
# Gemini is instructed to embed tags like: [OPEN_BROWSER: https://... ]
_ACTION_PATTERN = re.compile(
    r"\[OPEN_BROWSER:\s*(.+?)\s*\]",
    re.IGNORECASE,
)


def extract_and_execute_browser_actions(text: str) -> tuple[str, list[str]]:
    """Scan response text for [OPEN_BROWSER: ...] tags and execute them.

    Args:
        text: The raw response text from Gemini.

    Returns:
        A tuple of (cleaned_text, list_of_confirmation_messages).
        cleaned_text has the action tags stripped out.
    """
    confirmations: list[str] = []
    matches = _ACTION_PATTERN.findall(text)

    for target in matches:
        target = target.strip()
        if target.startswith(("http://", "https://")):
            msg = open_url(target)
        else:
            msg = open_site(target)
        confirmations.append(msg)
        logger.info(f"Browser action executed: {target}")

    # Strip the action tags from the spoken response
    cleaned = _ACTION_PATTERN.sub("", text).strip()
    # Collapse multiple spaces/newlines left behind
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"  +", " ", cleaned)

    return cleaned, confirmations
