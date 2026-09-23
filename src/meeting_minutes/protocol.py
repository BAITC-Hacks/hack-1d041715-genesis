"""Assemble a complete, auditable meeting protocol object."""

from __future__ import annotations

from pathlib import Path

from .models import (
    ActionItem,
    MeetingProtocol,
    ProtocolMetadata,
    TranscriptSegment,
    TranscriptionResult,
)


def build_protocol(
    transcription: TranscriptionResult,
    transcript: tuple[TranscriptSegment, ...],
    tasks: tuple[ActionItem, ...],
    summary: str,
    source_filename: str | None,
) -> MeetingProtocol:
    """Combine independently generated stages into one serializable protocol."""
    warnings = tuple(
        warning
        for warning in (
            transcription.warning,
            (
                "Demo/mock: транскрипт, поручения и саммари загружены из fixture."
                if transcription.mode == "demo"
                else None
            ),
            (
                "Облачный API-режим не соответствует требованиям закрытого контура."
                if transcription.mode == "api"
                else None
            ),
            (
                "LOCAL_KZ транскрибирует аудио локально, но поручения и саммари "
                "передают текст во внешний OpenAI API; полный закрытый контур не обеспечен."
                if transcription.mode == "local_kz"
                else None
            ),
        )
        if warning
    )
    metadata = ProtocolMetadata(
        mode=transcription.mode,
        source_filename=(
            Path(source_filename).name if source_filename else "demo_meeting.json"
        ),
        transcription_provider=transcription.provider,
        analysis_provider=(
            "demo-fixture" if transcription.mode == "demo" else "openai-responses"
        ),
        diarization_available=transcription.diarization_available,
        warnings=warnings,
    )
    return MeetingProtocol(
        summary=summary,
        tasks=tasks,
        transcript=transcript,
        metadata=metadata,
    )
