#!/usr/bin/env python3
"""
Friday AI Assistant - Entry Point.

A production-quality real-time AI voice assistant that runs in the
terminal. Uses LiveKit for voice processing, Google Gemini for
intelligence, and supports both voice and text interaction modes.

Usage:
    python main.py                  # Voice mode (default)
    python main.py --text           # Text-only mode
    python main.py --wake-word      # Voice mode with wake word
    python main.py --worker         # LiveKit worker mode

Environment variables must be set in .env (see .env.example).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Friday AI Voice Assistant — Your terminal-based AI companion.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py                  Start in voice mode\n"
            "  python main.py --text           Start in text-only mode\n"
            "  python main.py --wake-word      Voice mode with 'Friday' wake word\n"
            "  python main.py --worker         Run as a LiveKit worker\n"
        ),
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["voice", "text", "worker"],
        default="voice",
        help="Operation mode (default: voice)",
    )
    parser.add_argument(
        "--wake-word",
        "-w",
        action="store_true",
        help="Enable wake word detection ('Friday')",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point. Parses arguments and launches the assistant."""
    args = parse_args()

    if args.mode == "worker":
        # ── LiveKit Worker Mode ────────────────────────────────────────
        try:
            from worker import start_worker
        except ImportError as exc:
            print(f"Error: {exc}")
            print(
                "Worker mode requires livekit-agents and LiveKit plugins.\n"
                "Install with: pip install -r requirements.txt"
            )
            sys.exit(1)
        start_worker()
        return

    # ── Local Mode (voice or text) ─────────────────────────────────────
    if args.mode == "voice":
        # Check for audio dependencies
        try:
            import sounddevice  # noqa: F401
        except ImportError:
            print(
                "Warning: sounddevice not found. Falling back to text mode.\n"
                "Install with: pip install sounddevice numpy"
            )
            args.mode = "text"
        except OSError:
            print(
                "Warning: No audio devices found. Falling back to text mode."
            )
            args.mode = "text"

    # Run the assistant
    try:
        from assistant import FridayAssistant

        assistant = FridayAssistant(mode=args.mode)
        if args.wake_word:
            assistant.enable_wake_word()
        asyncio.run(assistant.run())

    except KeyboardInterrupt:
        print("\nGoodbye Boss!")
    except Exception as exc:
        print(f"Fatal error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    # Ensure we can import from the package
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
