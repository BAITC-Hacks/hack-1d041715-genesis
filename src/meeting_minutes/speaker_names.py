"""Contextual speaker-name inference without biometric identification."""

from __future__ import annotations

import re
from dataclasses import replace

from .models import TranscriptSegment


_NAME_TOKEN = r"[А-ЯЁӘҒҚҢӨҰҮҺІ][а-яёәғқңөұүһі]+"
_DIRECT_ADDRESS = re.compile(rf"\b({_NAME_TOKEN}(?:\s+{_NAME_TOKEN}){{0,2}}),")
_SELF_INTRODUCTION = re.compile(
    rf"\b(?:менің\s+атым|я\s+|мен\s+)({_NAME_TOKEN}(?:\s+{_NAME_TOKEN}){{0,2}})",
    re.IGNORECASE,
)
_ADDRESS_SIGNALS = (
    "вам слово",
    "сөз сізде",
    "проведите",
    "подготовьте",
    "өткізіңіз",
    "дайындаңыз",
    "жіберіңіз",
)
_NOT_NAMES = {"коллеги", "рахмет", "принято", "түсінікті"}


def infer_speaker_names(
    segments: tuple[TranscriptSegment, ...],
) -> tuple[TranscriptSegment, ...]:
    """Attach names inferred from self-introductions and adjacent direct address.

    The function deliberately keeps the technical ``speaker`` label unchanged.
    It only fills ``speaker_name`` when the conversational evidence is strong;
    otherwise the UI continues to show ``SPEAKER_XX``.
    """
    names: dict[str, str] = {}

    for index, segment in enumerate(segments):
        self_match = _SELF_INTRODUCTION.search(segment.text)
        if self_match:
            names[segment.speaker] = self_match.group(1).strip()

        if index + 1 >= len(segments):
            continue
        next_segment = segments[index + 1]
        if next_segment.speaker == segment.speaker:
            continue

        addressed_name = _find_directly_addressed_name(segment.text)
        if addressed_name:
            names.setdefault(next_segment.speaker, addressed_name)

    return tuple(
        replace(segment, speaker_name=names.get(segment.speaker, segment.speaker_name))
        for segment in segments
    )


def _find_directly_addressed_name(text: str) -> str | None:
    """Find a name followed by an instruction or invitation to speak."""
    for match in _DIRECT_ADDRESS.finditer(text):
        candidate = match.group(1).strip()
        if candidate.lower() in _NOT_NAMES:
            continue
        following_text = text[match.end() : match.end() + 180].lower()
        if any(signal in following_text for signal in _ADDRESS_SIGNALS):
            return candidate
    return None
