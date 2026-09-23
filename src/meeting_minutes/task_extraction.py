"""Extract evidence-backed action items from a structured transcript."""

from __future__ import annotations

import json
import re
from typing import Any

from .config import get_settings
from .demo import load_demo_tasks
from .models import ActionItem, TranscriptSegment
from .transcription import create_openai_client


class TaskExtractionError(RuntimeError):
    """Raised when action-item extraction cannot return a valid result."""


_TASK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "task": {"type": "string"},
                    "assignee": {"type": ["string", "null"]},
                    "deadline": {"type": ["string", "null"]},
                    "speaker": {"type": ["string", "null"]},
                    "source_text": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": [
                    "task",
                    "assignee",
                    "deadline",
                    "speaker",
                    "source_text",
                    "confidence",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["tasks"],
    "additionalProperties": False,
}


def extract_tasks(
    segments: tuple[TranscriptSegment, ...],
    client: Any | None = None,
) -> tuple[ActionItem, ...]:
    """Return only explicit assignments, preserving unknown values as ``None``."""
    settings = get_settings()
    if settings.is_demo:
        return load_demo_tasks()
    if not segments:
        return ()

    analysis_client = client if client is not None else create_openai_client()
    transcript_payload = [
        {
            "speaker": segment.speaker,
            "speaker_name": segment.speaker_name,
            "text": segment.text,
        }
        for segment in segments
    ]
    instructions = (
        "Извлеки только явно сформулированные поручения из стенограммы RU/KZ. "
        "Определи исполнителя по смыслу поручения, а не автоматически по говорящему. "
        "Сохраняй относительный срок в исходной формулировке. Если исполнитель или срок "
        "не указаны, верни null. Не придумывай поручения. source_text должен быть точной "
        "непрерывной цитатой из одной реплики. speaker — техническая метка реплики-источника."
    )

    try:
        response = analysis_client.responses.create(
            model=settings.text_model,
            input=[
                {"role": "system", "content": instructions},
                {
                    "role": "user",
                    "content": json.dumps(transcript_payload, ensure_ascii=False),
                },
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "meeting_action_items",
                    "strict": True,
                    "schema": _TASK_SCHEMA,
                }
            },
        )
        payload = json.loads(response.output_text)
    except Exception as error:
        raise TaskExtractionError(
            "Не удалось извлечь поручения. Проверьте API-ключ, модель и сеть."
        ) from error

    source_by_speaker = {
        (segment.speaker, _normalize_text(segment.text)) for segment in segments
    }
    tasks: list[ActionItem] = []
    for item in payload.get("tasks", []):
        source_text = str(item.get("source_text", "")).strip()
        speaker = item.get("speaker")
        if not source_text or not _has_source_evidence(
            source_by_speaker, speaker, source_text
        ):
            continue
        task_text = str(item.get("task", "")).strip()
        if not task_text:
            continue
        tasks.append(
            ActionItem(
                task=task_text,
                assignee=_optional_text(item.get("assignee")),
                deadline=_optional_text(item.get("deadline")),
                speaker=_optional_text(speaker),
                source_text=source_text,
                confidence=float(item.get("confidence", 0.0)),
            )
        )
    return tuple(tasks)


def _has_source_evidence(
    sources: set[tuple[str, str]], speaker: Any, source_text: str
) -> bool:
    """Require every extracted task to point to a literal transcript quotation."""
    normalized_quote = _normalize_text(source_text)
    return any(
        (speaker is None or source_speaker == speaker) and normalized_quote in source
        for source_speaker, source in sources
    )


def _normalize_text(value: str) -> str:
    """Normalize spacing and case for conservative source-quote comparison."""
    return re.sub(r"\s+", " ", value).strip().casefold()


def _optional_text(value: Any) -> str | None:
    """Convert empty strings and textual null markers to ``None``."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.casefold() in {"null", "не указан", "не указано", "unknown"}:
        return None
    return text
