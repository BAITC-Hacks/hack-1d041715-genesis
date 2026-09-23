"""Speaker diarization provider with a safe single-speaker fallback."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .config import get_settings
from .demo import load_demo_segments
from .models import TranscriptSegment, TranscriptionResult
from .transcription import (
    AudioTranscriptionError,
    create_openai_client,
    transcribe_audio,
    validate_audio_file,
)


class DiarizationError(RuntimeError):
    """Raised when neither diarization nor transcription fallback can complete."""


def transcribe_with_speakers(
    audio_path: str | None = None,
    client: Any | None = None,
) -> TranscriptionResult:
    """Return speaker segments from API, demo fixture, or single-speaker fallback.

    Demo mode never opens or uploads an audio file. API mode first requests
    ``diarized_json`` and, if that optional feature fails, preserves usability by
    calling the existing plain transcription function and assigning ``SPEAKER_00``.
    """
    settings = get_settings()
    if settings.is_demo:
        return TranscriptionResult(
            segments=load_demo_segments(),
            provider="demo-fixture",
            mode="demo",
            diarization_available=True,
            warning="Демонстрационный транскрипт загружен из fixture; аудио не распознавалось.",
        )

    if not audio_path:
        raise DiarizationError("В API-режиме необходимо выбрать аудиофайл.")

    path = validate_audio_file(audio_path)
    transcription_client = client if client is not None else create_openai_client()
    try:
        with path.open("rb") as audio_file:
            response = transcription_client.audio.transcriptions.create(
                model=settings.diarization_model,
                file=audio_file,
                response_format="diarized_json",
                chunking_strategy="auto",
            )
        segments = _parse_diarized_segments(getattr(response, "segments", ()))
        if not segments:
            raise DiarizationError("API не вернул сегменты говорящих.")
        return TranscriptionResult(
            segments=segments,
            provider=f"openai:{settings.diarization_model}",
            mode="api",
            diarization_available=True,
        )
    except Exception as diarization_error:
        try:
            transcript = transcribe_audio(path, client=transcription_client)
        except AudioTranscriptionError as transcription_error:
            raise DiarizationError(
                "Не удалось выполнить ни diarization, ни резервную транскрипцию."
            ) from transcription_error

        return TranscriptionResult(
            segments=(
                TranscriptSegment(
                    speaker="SPEAKER_00",
                    start=0.0,
                    end=0.0,
                    text=transcript,
                ),
            ),
            provider=f"openai:{settings.transcription_model}",
            mode="api",
            diarization_available=False,
            warning=(
                "Diarization недоступна; использована транскрипция одним говорящим. "
                f"Причина: {type(diarization_error).__name__}."
            ),
        )


def _parse_diarized_segments(raw_segments: Iterable[Any]) -> tuple[TranscriptSegment, ...]:
    """Normalize provider labels to stable ``SPEAKER_XX`` identifiers."""
    speaker_ids: dict[str, str] = {}
    parsed: list[TranscriptSegment] = []

    for raw in raw_segments:
        raw_speaker = str(_field(raw, "speaker", "unknown"))
        normalized = speaker_ids.setdefault(
            raw_speaker, f"SPEAKER_{len(speaker_ids):02d}"
        )
        text = str(_field(raw, "text", "")).strip()
        if not text:
            continue
        parsed.append(
            TranscriptSegment(
                speaker=normalized,
                start=float(_field(raw, "start", 0.0)),
                end=float(_field(raw, "end", 0.0)),
                text=text,
            )
        )

    return tuple(parsed)


def _field(value: Any, name: str, default: Any) -> Any:
    """Read a field from either an SDK object or a plain dictionary."""
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)
