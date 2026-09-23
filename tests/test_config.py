"""Tests for environment-backed provider selection."""

import os
import unittest

from meeting_minutes.config import get_settings


class ConfigTests(unittest.TestCase):
    """Verify that LOCAL_KZ is opt-in and keeps its documented default model."""

    def setUp(self) -> None:
        self.previous_mode = os.environ.get("APP_MODE")
        self.previous_model = os.environ.get("LOCAL_KZ_MODEL")

    def tearDown(self) -> None:
        for name, value in (
            ("APP_MODE", self.previous_mode),
            ("LOCAL_KZ_MODEL", self.previous_model),
        ):
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_local_kz_mode_uses_specialized_default_model(self) -> None:
        """Selecting LOCAL_KZ does not require another configuration switch."""
        os.environ["APP_MODE"] = "local_kz"
        os.environ.pop("LOCAL_KZ_MODEL", None)

        settings = get_settings()

        self.assertTrue(settings.is_local_kz)
        self.assertEqual(
            settings.local_kz_model,
            "shyngys879/kazakh-whisper-large-v3-turbo",
        )
