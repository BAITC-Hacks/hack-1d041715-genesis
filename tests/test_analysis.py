"""Tests for contextual speaker names, tasks, and summaries."""

import json
import os
import unittest
from types import SimpleNamespace

from meeting_minutes.demo import load_demo_segments
from meeting_minutes.models import TranscriptSegment
from meeting_minutes.speaker_names import infer_speaker_names
from meeting_minutes.summary import generate_summary
from meeting_minutes.task_extraction import extract_tasks


class FakeResponsesClient:
    """Minimal Responses API fake returning configured output text."""

    def __init__(self, output_text: str) -> None:
        self.output_text = output_text
        self.responses = SimpleNamespace(create=self.create_response)

    def create_response(self, **_: object) -> object:
        """Return a small SDK-like response."""
        return SimpleNamespace(output_text=self.output_text)


class AnalysisTests(unittest.TestCase):
    """Cover evidence-based names and action-item uncertainty."""

    def setUp(self) -> None:
        self.previous_mode = os.environ.get("APP_MODE")

    def tearDown(self) -> None:
        if self.previous_mode is None:
            os.environ.pop("APP_MODE", None)
        else:
            os.environ["APP_MODE"] = self.previous_mode

    def test_infers_names_from_direct_address_and_next_reply(self) -> None:
        """A direct address is attached only to the next responding speaker."""
        named = infer_speaker_names(load_demo_segments())
        name_by_speaker = {segment.speaker: segment.speaker_name for segment in named}

        self.assertEqual(name_by_speaker["SPEAKER_01"], "Гульмира Сериковна")
        self.assertEqual(name_by_speaker["SPEAKER_02"], "Нурлан Сагатович")
        self.assertEqual(name_by_speaker["SPEAKER_03"], "Айгуль")

    def test_demo_tasks_preserve_unknown_assignee_and_deadline(self) -> None:
        """The fixture demonstrates null values instead of invented facts."""
        os.environ["APP_MODE"] = "demo"

        tasks = extract_tasks(load_demo_segments())

        self.assertTrue(any(task.deadline is None for task in tasks))
        self.assertTrue(any(task.assignee is None for task in tasks))

    def test_api_tasks_require_literal_source_evidence(self) -> None:
        """Items without a transcript quotation are discarded as unsupported."""
        os.environ["APP_MODE"] = "api"
        segments = (
            TranscriptSegment("SPEAKER_00", 0, 3, "Нужно обновить план."),
        )
        payload = {
            "tasks": [
                {
                    "task": "Обновить план",
                    "assignee": None,
                    "deadline": None,
                    "speaker": "SPEAKER_00",
                    "source_text": "Нужно обновить план.",
                    "confidence": 0.8,
                },
                {
                    "task": "Позвонить клиенту",
                    "assignee": "Анна",
                    "deadline": "завтра",
                    "speaker": "SPEAKER_00",
                    "source_text": "Анна позвонит клиенту завтра.",
                    "confidence": 0.9,
                },
            ]
        }

        tasks = extract_tasks(
            segments,
            FakeResponsesClient(json.dumps(payload, ensure_ascii=False)),
        )

        self.assertEqual(len(tasks), 1)
        self.assertIsNone(tasks[0].assignee)
        self.assertIsNone(tasks[0].deadline)

    def test_demo_summary_is_explicit_fixture_content(self) -> None:
        """Offline summary generation returns the packaged mock summary."""
        os.environ["APP_MODE"] = "demo"

        summary = generate_summary(load_demo_segments())

        self.assertIn("92", summary)
        self.assertIn("не назван", summary)
