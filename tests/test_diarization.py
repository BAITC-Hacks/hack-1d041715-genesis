"""Tests for API diarization normalization and resilient fallback behavior."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from meeting_minutes.diarization import transcribe_with_speakers
from meeting_minutes.local_kz import (
    LocalKazakhDependenciesError,
    LocalKazakhResult,
)


class FakeDiarizationClient:
    """OpenAI-compatible fake with configurable diarization and fallback calls."""

    def __init__(
        self,
        diarization_error: Exception | None = None,
        segments: list[object] | None = None,
    ) -> None:
        self.diarization_error = diarization_error
        self.segments = segments
        self.audio = SimpleNamespace(
            transcriptions=SimpleNamespace(create=self.create_transcription)
        )

    def create_transcription(self, **kwargs: object) -> object:
        """Return realistic SDK-like responses for the requested response format."""
        if kwargs.get("response_format") == "diarized_json":
            if self.diarization_error:
                raise self.diarization_error
            return SimpleNamespace(
                segments=self.segments or [
                    SimpleNamespace(speaker="A", start=0.0, end=1.5, text="Здравствуйте."),
                    SimpleNamespace(speaker="B", start=1.5, end=3.0, text="Сәлеметсіз бе."),
                ]
            )
        return "Резервный транскрипт."


class DiarizationTests(unittest.TestCase):
    """Verify demo, real provider parsing, and optional-provider fallback."""

    def setUp(self) -> None:
        self.previous_mode = os.environ.get("APP_MODE")
        self.temp_dir = tempfile.TemporaryDirectory()
        self.audio_path = Path(self.temp_dir.name) / "meeting.wav"
        self.audio_path.write_bytes(b"test-audio")

    def tearDown(self) -> None:
        if self.previous_mode is None:
            os.environ.pop("APP_MODE", None)
        else:
            os.environ["APP_MODE"] = self.previous_mode
        self.temp_dir.cleanup()

    def test_demo_returns_multispeaker_fixture_without_audio(self) -> None:
        """Offline demo mode exposes explicit multi-speaker mock segments."""
        os.environ["APP_MODE"] = "demo"

        result = transcribe_with_speakers()

        self.assertEqual(result.mode, "demo")
        self.assertTrue(result.diarization_available)
        self.assertGreater(len({segment.speaker for segment in result.segments}), 1)

    def test_api_normalizes_provider_speaker_labels(self) -> None:
        """Provider-specific labels become stable SPEAKER_XX identifiers."""
        os.environ["APP_MODE"] = "api"

        result = transcribe_with_speakers(self.audio_path, FakeDiarizationClient())

        self.assertEqual([segment.speaker for segment in result.segments], [
            "SPEAKER_00",
            "SPEAKER_01",
        ])

    def test_arbitrary_provider_text_never_becomes_internal_speaker_id(self) -> None:
        """Unexpected raw labels are deterministically mapped to safe identifiers."""
        os.environ["APP_MODE"] = "api"
        raw_segments = [
            SimpleNamespace(speaker="A", start=0.0, end=1.0, text="Первый."),
            SimpleNamespace(
                speaker="сейчас не отдам",
                start=1.0,
                end=2.0,
                text="Второй.",
            ),
            SimpleNamespace(
                speaker="сейчас не отдам",
                start=2.0,
                end=3.0,
                text="Третий.",
            ),
        ]

        result = transcribe_with_speakers(
            self.audio_path,
            FakeDiarizationClient(segments=raw_segments),
        )

        self.assertEqual(
            [segment.speaker for segment in result.segments],
            ["SPEAKER_00", "SPEAKER_01", "SPEAKER_01"],
        )
        self.assertTrue(
            all(segment.speaker.startswith("SPEAKER_") for segment in result.segments)
        )

    def test_missing_provider_labels_are_not_merged(self) -> None:
        """Unlabelled turns remain distinct instead of collapsing into one speaker."""
        os.environ["APP_MODE"] = "api"
        raw_segments = [
            SimpleNamespace(speaker=None, start=0.0, end=1.0, text="Первый."),
            SimpleNamespace(speaker="unknown", start=1.0, end=2.0, text="Второй."),
        ]

        result = transcribe_with_speakers(
            self.audio_path,
            FakeDiarizationClient(segments=raw_segments),
        )

        self.assertEqual(
            [segment.speaker for segment in result.segments],
            ["SPEAKER_00", "SPEAKER_01"],
        )

    def test_diarization_failure_falls_back_to_single_speaker(self) -> None:
        """Optional diarization failure does not abort a valid transcription."""
        os.environ["APP_MODE"] = "api"

        with self.assertLogs("meeting_minutes.diarization", level="ERROR") as logs:
            result = transcribe_with_speakers(
                self.audio_path,
                FakeDiarizationClient(diarization_error=RuntimeError("unavailable")),
            )

        self.assertFalse(result.diarization_available)
        self.assertEqual(result.segments[0].speaker, "SPEAKER_00")
        self.assertIn("Резервный", result.text)
        self.assertIn("OpenAI diarization failed", "\n".join(logs.output))

    def test_local_kz_selects_local_provider_without_openai_stt(self) -> None:
        """LOCAL_KZ uses one safe speaker and never calls the OpenAI STT client."""
        os.environ["APP_MODE"] = "local_kz"
        local_result = LocalKazakhResult(
            text="Қазақша жыл санау.",
            device="cuda:0",
            duration_seconds=78.43,
        )

        with (
            patch(
                "meeting_minutes.diarization.transcribe_kazakh_audio",
                return_value=local_result,
            ) as transcribe_local,
            patch("meeting_minutes.diarization.create_openai_client") as create_client,
        ):
            result = transcribe_with_speakers(self.audio_path)

        transcribe_local.assert_called_once()
        create_client.assert_not_called()
        self.assertEqual(result.mode, "local_kz")
        self.assertFalse(result.diarization_available)
        self.assertEqual(result.segments[0].speaker, "SPEAKER_00")
        self.assertEqual(result.segments[0].text, "Қазақша жыл санау.")
        self.assertEqual(result.segments[0].end, 78.43)

    def test_local_kz_reports_missing_optional_dependencies(self) -> None:
        """A missing optional install produces an actionable error without fallback."""
        os.environ["APP_MODE"] = "local_kz"

        with patch(
            "meeting_minutes.diarization.transcribe_kazakh_audio",
            side_effect=LocalKazakhDependenciesError("Install .[local-kz]"),
        ):
            with self.assertRaisesRegex(
                LocalKazakhDependenciesError,
                "local-kz",
            ):
                transcribe_with_speakers(self.audio_path)
