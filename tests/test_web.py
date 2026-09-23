"""Flask UI tests for the autonomous demonstration flow."""

import os
import unittest

from meeting_minutes.web import create_app


class WebTests(unittest.TestCase):
    """Verify the primary jury-facing UI path and DOCX download."""

    def setUp(self) -> None:
        self.previous_mode = os.environ.get("APP_MODE")
        os.environ["APP_MODE"] = "demo"
        self.app = create_app()
        self.app.config.update(TESTING=True)
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        if self.previous_mode is None:
            os.environ.pop("APP_MODE", None)
        else:
            os.environ["APP_MODE"] = self.previous_mode

    def test_home_marks_demo_as_mock(self) -> None:
        """The start page never presents fixture data as real transcription."""
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Демонстрационный режим".encode(), response.data)
        self.assertIn("Аудио не распознаётся".encode(), response.data)

    def test_demo_processes_protocol_and_downloads_docx(self) -> None:
        """One button reaches the result page and a valid DOCX response."""
        response = self.client.post("/process")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Протокол совещания".encode(), response.data)
        self.assertIn("Demo/mock".encode(), response.data)
        result_id = next(iter(self.app.config["DOCX_RESULTS"]))
        download = self.client.get(f"/download/{result_id}")
        self.assertEqual(download.status_code, 200)
        self.assertTrue(download.data.startswith(b"PK"))

    def test_api_mode_requires_audio_upload(self) -> None:
        """Real mode reports a clear missing-file error rather than a traceback."""
        os.environ["APP_MODE"] = "api"

        response = self.client.post("/process")

        self.assertEqual(response.status_code, 400)
        self.assertIn("Выберите аудиофайл".encode(), response.data)
