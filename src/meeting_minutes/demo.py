"""Load explicit, offline-only demonstration fixtures."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

from .models import ActionItem, TranscriptSegment


def load_demo_fixture() -> dict[str, Any]:
    """Return the packaged RU/KZ demo meeting fixture."""
    fixture_path = files("meeting_minutes").joinpath("fixtures/demo_meeting.json")
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def load_demo_segments() -> tuple[TranscriptSegment, ...]:
    """Return speaker-labelled transcript segments from the demo fixture."""
    return tuple(TranscriptSegment(**item) for item in load_demo_fixture()["transcript"])


def load_demo_tasks() -> tuple[ActionItem, ...]:
    """Return explicitly mocked action items from the demo fixture."""
    return tuple(ActionItem(**item) for item in load_demo_fixture()["tasks"])


def load_demo_summary() -> str:
    """Return the explicitly mocked management summary."""
    return str(load_demo_fixture()["summary"])
