"""Audio transcription through the OpenAI Whisper API."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from .config import get_settings, load_environment
from .diagnostics import log_provider_error


logger = logging.getLogger(__name__)

SUPPORTED_AUDIO_EXTENSIONS = frozenset(
    {".flac", ".m4a", ".mp3", ".mp4", ".mpeg", ".mpga", ".ogg", ".wav", ".webm"}
)
"""Audio filename extensions accepted by the OpenAI transcription endpoint."""

DEMO_TRANSCRIPT = (
    "Айгуль подготовит отчёт по бюджету до пятницы. "
    "Ерлан согласует обновлённый план с командой до 15 октября."
)
"""Predictable transcript returned only when ``TRANSCRIPTION_MODE=demo``."""


class AudioTranscriptionError(RuntimeError):
    """Base exception for an audio transcription failure."""


class AudioFileNotFoundError(AudioTranscriptionError):
    """Raised when the supplied audio path does not point to a file."""


class UnsupportedAudioFormatError(AudioTranscriptionError):
    """Raised when an audio file has an unsupported extension."""


class CorruptedAudioError(AudioTranscriptionError):
    """Raised when the audio is empty or the API cannot decode it."""


class TranscriptionServiceError(AudioTranscriptionError):
    """Raised for authentication, network, or unexpected API failures."""


def transcribe_audio(audio_path: str | Path, client: Any | None = None) -> str:
    """Return a transcript for one supported audio file.

    The default path sends the file to OpenAI's ``whisper-1`` transcription model.
    Set ``TRANSCRIPTION_MODE=demo`` to exercise the interface without an API key;
    demo mode validates the file but returns :data:`DEMO_TRANSCRIPT` instead of
    contacting OpenAI. ``client`` is an optional OpenAI-compatible client used by
    automated tests or by an application that manages its own client instance.

    Args:
        audio_path: Path to a non-empty supported audio file.
        client: Optional object exposing ``audio.transcriptions.create``.

    Returns:
        The non-empty transcription text.

    Raises:
        AudioFileNotFoundError: The path is missing or is not a file.
        UnsupportedAudioFormatError: The file extension is unsupported.
        CorruptedAudioError: The file is empty or cannot be decoded by the API.
        TranscriptionServiceError: API credentials, connection, or service failed.
    """
    path = _validate_audio_file(audio_path)

    settings = get_settings()
    if settings.is_demo:
        return DEMO_TRANSCRIPT

    transcription_client = client if client is not None else _create_openai_client()

    try:
        with path.open("rb") as audio_file:
            response = transcription_client.audio.transcriptions.create(
                model=settings.transcription_model,
                file=audio_file,
                response_format="text",
            )
    except OSError as error:
        logger.error(
            "Audio file read failed: filename=%s error_type=%s message=%s",
            path.name,
            type(error).__name__,
            str(error),
        )
        raise CorruptedAudioError(f"Не удалось прочитать аудиофайл: {path.name}") from error
    except Exception as error:  # API-specific types differ across SDK releases.
        log_provider_error(
            logger,
            operation="OpenAI fallback transcription",
            model=settings.transcription_model,
            error=error,
        )
        if getattr(error, "status_code", None) in {400, 422}:
            raise CorruptedAudioError(
                "OpenAI не смог распознать аудио. Файл может быть повреждён."
            ) from error
        raise TranscriptionServiceError(
            "Не удалось получить транскрипт от OpenAI. Проверьте OPENAI_API_KEY и сеть."
        ) from error

    transcript = response if isinstance(response, str) else getattr(response, "text", None)
    if not isinstance(transcript, str) or not transcript.strip():
        raise CorruptedAudioError("OpenAI вернул пустой транскрипт для этого аудио.")

    return transcript.strip()


def _validate_audio_file(audio_path: str | Path) -> Path:
    """Validate that an audio file exists, has a supported extension, and is non-empty."""
    path = Path(audio_path)

    if not path.is_file():
        raise AudioFileNotFoundError(f"Аудиофайл не найден: {path}")
    if path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        supported_formats = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
        raise UnsupportedAudioFormatError(
            f"Неподдерживаемый формат {path.suffix or 'без расширения'}. "
            f"Поддерживаются: {supported_formats}."
        )
    if path.stat().st_size == 0:
        raise CorruptedAudioError(f"Аудиофайл пустой и, вероятно, повреждён: {path.name}")

    return path


def _create_openai_client() -> Any:
    """Create the SDK client lazily so demo mode works without installed packages."""
    load_environment()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == "your_openai_api_key_here":
        raise TranscriptionServiceError(
            "Не задан OPENAI_API_KEY. Добавьте ключ в environment/.env или включите APP_MODE=demo."
        )

    try:
        from openai import OpenAI
    except ImportError as error:
        raise TranscriptionServiceError(
            "Не установлена библиотека openai. Выполните: pip install -r requirements.txt"
        ) from error

    try:
        return OpenAI()
    except Exception as error:
        raise TranscriptionServiceError("Не удалось создать клиент OpenAI.") from error


def validate_audio_file(audio_path: str | Path) -> Path:
    """Public compatibility wrapper for validation used by other audio providers."""
    return _validate_audio_file(audio_path)


def create_openai_client() -> Any:
    """Public compatibility wrapper for creating the configured OpenAI client."""
    return _create_openai_client()
