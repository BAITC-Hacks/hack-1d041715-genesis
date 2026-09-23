"""Tests for API diarization normalization and resilient fallback behavior."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from meeting_minutes.diarization import transcribe_with_speakers


class FakeDiarizationClient:
    """OpenAI-compatible fake with configurable diarization and fallback calls."""

    def __init__(self, diarization_error: Exception | None = None) -> None:
        self.diarization_error = diarization_error
        self.audio = SimpleNamespace(
            transcriptions=SimpleNamespace(create=self.create_transcription)
        )

    def create_transcription(self, **kwargs: object) -> object:
        """Return realistic SDK-like responses for the requested response format."""
        if kwargs.get("response_format") == "diarized_json":
            if self.diarization_error:
                raise self.diarization_error
            return SimpleNamespace(
                segments=[
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
