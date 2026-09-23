"""Unit tests for the isolated audio transcription module."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from meeting_minutes.transcription import (
    CorruptedAudioError,
    UnsupportedAudioFormatError,
    transcribe_audio,
)


class FakeBadRequest(Exception):
    """Minimal API-like validation error used to simulate corrupted audio."""

    status_code = 400


class FakeTranscriptionClient:
    """OpenAI-compatible fake client that returns a known transcription response."""

    def __init__(self, response: object | None = None, error: Exception | None = None) -> None:
        self.response = response or SimpleNamespace(text="Тестовый транскрипт.")
        self.error = error
        self.audio = SimpleNamespace(
            transcriptions=SimpleNamespace(create=self.create_transcription)
        )

    def create_transcription(self, **_: object) -> object:
        """Return the configured response or raise the configured fake API error."""
        if self.error:
            raise self.error
        return self.response


class TranscribeAudioTests(unittest.TestCase):
    """Verify successful and invalid-file transcription paths."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.audio_path = Path(self.temp_dir.name) / "meeting.wav"
        self.audio_path.write_bytes(b"not-real-audio")
        self.previous_mode = os.environ.pop("TRANSCRIPTION_MODE", None)

    def tearDown(self) -> None:
        if self.previous_mode is not None:
            os.environ["TRANSCRIPTION_MODE"] = self.previous_mode
        self.temp_dir.cleanup()

    def test_returns_text_from_openai_compatible_client(self) -> None:
        """A valid API response exposes its text to the caller."""
        result = transcribe_audio(self.audio_path, FakeTranscriptionClient())

        self.assertEqual(result, "Тестовый транскрипт.")

    def test_rejects_unsupported_extension(self) -> None:
        """Extensions outside the documented API list fail before the network call."""
        document_path = Path(self.temp_dir.name) / "notes.txt"
        document_path.write_text("not audio", encoding="utf-8")

        with self.assertRaises(UnsupportedAudioFormatError):
            transcribe_audio(document_path, FakeTranscriptionClient())

    def test_maps_decode_error_to_corrupted_audio_error(self) -> None:
        """A validation error from the API is presented as a damaged audio file."""
        client = FakeTranscriptionClient(error=FakeBadRequest())

        with self.assertRaises(CorruptedAudioError):
            transcribe_audio(self.audio_path, client)

    def test_demo_mode_does_not_need_an_api_client(self) -> None:
        """Demo mode permits a no-account project check with any supported file."""
        os.environ["TRANSCRIPTION_MODE"] = "demo"

        result = transcribe_audio(self.audio_path)

        self.assertIn("Айгуль", result)


if __name__ == "__main__":
    unittest.main()
