"""Small serializable data models shared by all pipeline stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    """One time-bounded utterance attributed to a speaker label."""

    speaker: str
    start: float
    end: float
    text: str
    speaker_name: str | None = None

    @property
    def display_speaker(self) -> str:
        """Return a contextual name with its auditable technical label."""
        if self.speaker_name:
            return f"{self.speaker_name} ({self.speaker})"
        return self.speaker

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ActionItem:
    """A meeting assignment without invented assignee or deadline values."""

    task: str
    assignee: str | None
    deadline: str | None
    speaker: str | None = None
    source_text: str | None = None
    confidence: float | None = None
    status: str = "open"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """Transcript plus provider and fallback transparency metadata."""

    segments: tuple[TranscriptSegment, ...]
    provider: str
    mode: str
    diarization_available: bool
    warning: str | None = None

    @property
    def text(self) -> str:
        """Join segment text into the legacy plain-transcript form."""
        return " ".join(segment.text.strip() for segment in self.segments if segment.text.strip())


@dataclass(frozen=True, slots=True)
class ProtocolMetadata:
    """Auditable metadata describing how the protocol was produced."""

    mode: str
    source_filename: str
    transcription_provider: str
    analysis_provider: str
    diarization_available: bool
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MeetingProtocol:
    """Complete result consumed by exporters and the user interface."""

    summary: str
    tasks: tuple[ActionItem, ...]
    transcript: tuple[TranscriptSegment, ...]
    metadata: ProtocolMetadata

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable protocol object."""
        return {
            "summary": self.summary,
            "tasks": [task.to_dict() for task in self.tasks],
            "transcript": [segment.to_dict() for segment in self.transcript],
            "metadata": self.metadata.to_dict(),
        }
