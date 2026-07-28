"""
Logging module for Friday AI Assistant.

Provides rich, colourful logging with a Friday-themed console output.
All log messages are formatted with timestamps and contextual prefixes.
"""

from __future__ import annotations

import codecs
import logging
import sys
from pathlib import Path
from typing import ClassVar

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# Force UTF-8 on Windows so box-drawing / emoji characters don't crash
if sys.platform == "win32":
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, errors="replace")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, errors="replace")

# ── Rich Theme ──────────────────────────────────────────────────────────────
FRIDAY_THEME = Theme(
    {
        "friday.title": "bold blue",
        "friday.user": "bold yellow",
        "friday.bot": "bold cyan",
        "friday.info": "dim white",
        "friday.success": "bold green",
        "friday.warning": "bold yellow",
        "friday.error": "bold red",
        "friday.prompt": "bold magenta",
    }
)

# ── Console ─────────────────────────────────────────────────────────────────
console = Console(
    theme=FRIDAY_THEME,
    highlight=False,
    force_terminal=True,
    file=open(sys.stdout.fileno(), mode="w", encoding="utf-8", errors="replace", closefd=False)
    if sys.platform == "win32" else None,
)


# ── Custom Logger ───────────────────────────────────────────────────────────
class FridayLogger:
    """Friday-themed logging wrapper around Rich logging.

    Provides both standard logging and direct console output
    methods styled for the Friday assistant persona.
    """

    _instances: ClassVar[dict[str, "FridayLogger"]] = {}

    def __init__(self, name: str = "friday", level: str = "INFO") -> None:
        self.name = name
        self._logger = logging.getLogger(name)
        self._logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        self._logger.handlers.clear()

        # Rich handler
        handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(handler)

    @classmethod
    def get_logger(cls, name: str = "friday", level: str = "INFO") -> "FridayLogger":
        """Get or create a FridayLogger instance."""
        if name not in cls._instances:
            cls._instances[name] = cls(name, level)
        return cls._instances[name]

    def debug(self, message: str) -> None:
        """Log a debug message."""
        self._logger.debug(message)

    def info(self, message: str) -> None:
        """Log an info message."""
        self._logger.info(message)

    def warning(self, message: str) -> None:
        """Log a warning message."""
        self._logger.warning(message)

    def error(self, message: str) -> None:
        """Log an error message."""
        self._logger.error(message)

    def critical(self, message: str) -> None:
        """Log a critical message."""
        self._logger.critical(message)

    # ── Direct console helpers ───────────────────────────────────────────

    def print_banner(self) -> None:
        """Print the Friday startup banner."""
        console.print()
        console.print(
            "=" * 50,
            style="friday.title",
        )
        console.print(
            "   ███████╗██████╗ ██╗██████╗  █████╗ ██╗   ██╗",
            style="friday.title",
        )
        console.print(
            "   ██╔════╝██╔══██╗██║██╔══██╗██╔══██╗╚██╗ ██╔╝",
            style="friday.title",
        )
        console.print(
            "   █████╗  ██████╔╝██║██║  ██║███████║ ╚████╔╝ ",
            style="friday.title",
        )
        console.print(
            "   ██╔══╝  ██╔══██╗██║██║  ██║██╔══██║  ╚██╔╝  ",
            style="friday.title",
        )
        console.print(
            "   ██║     ██║  ██║██║██████╔╝██║  ██║   ██║   ",
            style="friday.title",
        )
        console.print(
            "   ╚═╝     ╚═╝  ╚═╝╚═╝╚═════╝ ╚═╝  ╚═╝   ╚═╝   ",
            style="friday.title",
        )
        console.print(
            "   🤖  AI Voice Assistant  🤖",
            style="friday.info",
        )
        console.print("=" * 50, style="friday.title")
        console.print()

    def print_user_message(self, text: str) -> None:
        """Print a formatted user message."""
        console.print(f"\n[bold yellow]You:[/bold yellow] {text}")

    def print_friday_message(self, text: str) -> None:
        """Print a formatted Friday response message."""
        console.print(f"[bold cyan]Friday:[/bold cyan] {text}")

    def print_listening(self) -> None:
        """Print the listening indicator."""
        console.print(
            "\n[dim]🎤 Listening...[/dim]", end=""
        )

    def print_status(self, message: str) -> None:
        """Print a status update."""
        console.print(f"[dim]{'─'*50}[/dim]")
        console.print(f"[dim]ℹ {message}[/dim]")

    def print_success(self, message: str) -> None:
        """Print a success message."""
        console.print(f"[bold green]✓ {message}[/bold green]")

    def print_warning(self, message: str) -> None:
        """Print a warning message."""
        console.print(f"[bold yellow]⚠ {message}[/bold yellow]")

    def print_error(self, message: str) -> None:
        """Print an error message."""
        console.print(f"[bold red]✗ {message}[/bold red]")


# ── Module-level convenience ────────────────────────────────────────────────
def get_logger(name: str = "friday", level: str = "INFO") -> FridayLogger:
    """Get a FridayLogger instance.

    Args:
        name: Logger name (default: "friday").
        level: Log level (default: "INFO").

    Returns:
        Configured FridayLogger instance.
    """
    return FridayLogger.get_logger(name, level)
