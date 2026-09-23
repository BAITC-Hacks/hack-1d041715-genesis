"""End-to-end meeting processing orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .diarization import transcribe_with_speakers
from .models import MeetingProtocol
from .protocol import build_protocol
from .speaker_names import infer_speaker_names
from .summary import generate_summary
from .task_extraction import extract_tasks


def process_meeting(
    audio_path: str | Path | None = None,
    client: Any | None = None,
) -> MeetingProtocol:
    """Run audio through speakers, tasks, summary, and protocol assembly.

    In demo mode ``audio_path`` is intentionally ignored and no external call is
    made. In API mode the same injected client can be reused by all stages.
    """
    path_text = str(audio_path) if audio_path is not None else None
    transcription = transcribe_with_speakers(path_text, client=client)
    named_segments = infer_speaker_names(transcription.segments)
    tasks = extract_tasks(named_segments, client=client)
    summary = generate_summary(named_segments, client=client)
    return build_protocol(
        transcription=transcription,
        transcript=named_segments,
        tasks=tasks,
        summary=summary,
        source_filename=path_text,
    )
