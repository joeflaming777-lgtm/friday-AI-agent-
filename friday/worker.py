"""
LiveKit worker mode for Friday AI Assistant.

This entrypoint is used when running Friday as a LiveKit worker.
It connects to a LiveKit server, joins rooms, and processes
audio from room participants using VoicePipelineAgent.

Usage:
    python main.py --worker
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli

from config import load_config
from logger import get_logger, console
from services.livekit_adapter import GeminiLLM
from services.speech_service import create_stt, create_tts, create_vad

logger = get_logger("friday.worker")


async def entrypoint(job: JobContext) -> None:
    """LiveKit worker entrypoint called when a job is assigned.

    Connects to the LiveKit room, creates a VoicePipelineAgent
    with Gemini as the LLM, and starts processing audio from
    room participants.

    Args:
        job: The LiveKit job context providing room access.
    """
    logger.info("Job received, connecting to room...")
    await job.connect()

    config = load_config()

    # Initialize components
    logger.print_status("Initializing Friday worker...")

    try:
        vad = create_vad()
        stt = create_stt(config.stt)
        tts = create_tts(config.tts)
        llm = GeminiLLM(
            api_key=config.gemini.api_key,
            model=config.gemini.model,
        )
    except Exception as exc:
        logger.error(f"Failed to initialize components: {exc}")
        return

    from livekit.agents import VoicePipelineAgent

    # Create the voice pipeline agent
    agent = VoicePipelineAgent(
        vad=vad,
        stt=stt,
        llm=llm,
        tts=tts,
        allow_interruptions=True,
    )

    # Set up event handlers
    @agent.on("user_speech_committed")
    def on_user_speech(agent: VoicePipelineAgent, message: Any) -> None:
        """Handle transcribed user speech."""
        logger.print_user_message(message.text)

    @agent.on("agent_speech_committed")
    def on_agent_speech(agent: VoicePipelineAgent, message: Any) -> None:
        """Handle agent speech response."""
        logger.print_friday_message(message.text)

    # Start the agent in the room
    agent.start(job.room)

    # Greeting
    await agent.say("Hello Boss. How can I help you today?")
    logger.print_success("Friday worker ready")

    # Keep running
    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        logger.info("Worker task cancelled")
    finally:
        logger.info("Worker shutting down")


def start_worker() -> None:
    """Start the LiveKit worker using the standard CLI runner.

    Connects to the LiveKit server configured in .env and
    waits for job assignments.
    """
    console.print("[bold blue]Friday AI Assistant - Worker Mode[/bold blue]")
    console.print("[dim]Connecting to LiveKit server...[/dim]\n")

    # Suppress noisy LiveKit logs
    logging.getLogger("livekit").setLevel(logging.WARNING)

    import sys
    sys.argv = [sys.argv[0], "start"]

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="friday-assistant",
        )
    )
