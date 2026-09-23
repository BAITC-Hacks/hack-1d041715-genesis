"""Generate a concise evidence-bound management summary."""

from __future__ import annotations

import json
from typing import Any

from .config import get_settings
from .demo import load_demo_summary
from .models import TranscriptSegment
from .transcription import create_openai_client


class SummaryError(RuntimeError):
    """Raised when summary generation fails or returns empty text."""


def generate_summary(
    segments: tuple[TranscriptSegment, ...],
    client: Any | None = None,
) -> str:
    """Return a short management summary without adding external facts."""
    settings = get_settings()
    if settings.is_demo:
        return load_demo_summary()
    if not segments:
        return "Обсуждение отсутствует."

    analysis_client = client if client is not None else create_openai_client()
    payload = [
        {
            "speaker": segment.display_speaker,
            "text": segment.text,
        }
        for segment in segments
    ]
    instructions = (
        "Составь краткое управленческое саммари совещания на русском языке: основные "
        "темы, проблемы, решения и важные числовые показатели. Используй только факты "
        "из стенограммы, не добавляй предположений. Объём — 4–7 коротких предложений."
    )
    try:
        response = analysis_client.responses.create(
            model=settings.text_model,
            input=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        summary = str(response.output_text).strip()
    except Exception as error:
        raise SummaryError(
            "Не удалось сформировать саммари. Проверьте API-ключ, модель и сеть."
        ) from error

    if not summary:
        raise SummaryError("Модель вернула пустое саммари.")
    return summary
