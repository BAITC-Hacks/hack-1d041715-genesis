"""Environment-backed configuration for the meeting processing pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass


def load_environment() -> None:
    """Load a local ``.env`` file without overriding existing environment variables."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv(override=False)


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings read only from environment variables or ``.env``."""

    mode: str
    transcription_model: str
    diarization_model: str
    text_model: str

    @property
    def is_demo(self) -> bool:
        """Return whether external API calls must be replaced with demo fixtures."""
        return self.mode == "demo"


def get_settings() -> Settings:
    """Load and validate current runtime settings."""
    load_environment()
    legacy_mode = os.getenv("TRANSCRIPTION_MODE")
    mode = os.getenv("APP_MODE", legacy_mode or "demo").strip().lower()
    if mode not in {"demo", "api"}:
        raise ValueError("APP_MODE должен иметь значение 'demo' или 'api'.")

    return Settings(
        mode=mode,
        transcription_model=os.getenv("OPENAI_TRANSCRIPTION_MODEL", "whisper-1"),
        diarization_model=os.getenv(
            "OPENAI_DIARIZATION_MODEL", "gpt-4o-transcribe-diarize"
        ),
        text_model=os.getenv("OPENAI_TEXT_MODEL", "gpt-6-astra"),
    )
