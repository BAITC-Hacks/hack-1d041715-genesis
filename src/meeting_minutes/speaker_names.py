"""Contextual speaker-name inference without biometric identification."""

from __future__ import annotations

import re
from dataclasses import replace

from .models import TranscriptSegment


_NAME_TOKEN = r"[А-ЯЁӘҒҚҢӨҰҮҺІ][а-яёәғқңөұүһі]+"
_DIRECT_ADDRESS = re.compile(rf"\b({_NAME_TOKEN}(?:\s+{_NAME_TOKEN}){{0,2}}),")
_SELF_INTRODUCTION = re.compile(
    rf"\b(?:(?i:менің\s+атым|я|мен)\s+)"
    rf"({_NAME_TOKEN}(?:\s+{_NAME_TOKEN}){{0,2}})",
)
_ADDRESS_SIGNALS = (
    "вам слово",
    "сөз сізде",
    "что у вас",
    "что предлагаете",
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
    self_names: dict[str, set[str]] = {}
    addressed_names: dict[str, set[str]] = {}
    turn_names: dict[int, str] = {}

    for segment in segments:
        self_match = _SELF_INTRODUCTION.search(segment.text)
        if self_match:
            self_names.setdefault(segment.speaker, set()).add(
                self_match.group(1).strip()
            )

    index = 0
    while index < len(segments):
        turn_end = _turn_end(segments, index)
        if turn_end >= len(segments):
            break

        turn_text = " ".join(segment.text for segment in segments[index:turn_end])
        addressed_name = _find_directly_addressed_name(turn_text)
        if addressed_name:
            response_speaker = segments[turn_end].speaker
            response_end = _turn_end(segments, turn_end)
            addressed_names.setdefault(response_speaker, set()).add(addressed_name)
            for response_index in range(turn_end, response_end):
                turn_names[response_index] = addressed_name

        index = turn_end

    global_names: dict[str, str] = {}
    all_speakers = set(self_names) | set(addressed_names)
    for speaker in all_speakers:
        explicit = self_names.get(speaker, set())
        contextual = addressed_names.get(speaker, set())
        if len(explicit) == 1:
            global_names[speaker] = next(iter(explicit))
        elif not explicit and len(contextual) == 1:
            global_names[speaker] = next(iter(contextual))

    return tuple(
        replace(
            segment,
            speaker_name=(
                turn_names.get(index)
                or global_names.get(segment.speaker)
                or segment.speaker_name
            ),
        )
        for index, segment in enumerate(segments)
    )


def _find_directly_addressed_name(text: str) -> str | None:
    """Find a name followed by an instruction or invitation to speak."""
    addressed_name: str | None = None
    for match in _DIRECT_ADDRESS.finditer(text):
        candidate = match.group(1).strip()
        if candidate.lower() in _NOT_NAMES:
            continue
        following_text = text[match.end() : match.end() + 180].lower()
        has_signal = any(signal in following_text for signal in _ADDRESS_SIGNALS)
        is_direct_question = len(candidate.split()) >= 2 and "?" in following_text
        if has_signal or is_direct_question:
            addressed_name = candidate
    return addressed_name


def _turn_end(segments: tuple[TranscriptSegment, ...], start: int) -> int:
    """Return the first index after a contiguous turn by one technical speaker."""
    speaker = segments[start].speaker
    index = start + 1
    while index < len(segments) and segments[index].speaker == speaker:
        index += 1
    return index
