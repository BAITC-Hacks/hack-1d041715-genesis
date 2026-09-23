"""Minimal Flask interface for the end-to-end meeting protocol pipeline."""

from __future__ import annotations

import os
import tempfile
from collections import OrderedDict
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from flask import Flask, abort, render_template, request, send_file
from werkzeug.utils import secure_filename

from .config import get_settings
from .export_docx import build_protocol_docx
from .pipeline import process_meeting


def create_app() -> Flask:
    """Create the web application without requiring an API key in demo mode."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
    app.config["DOCX_RESULTS"] = OrderedDict()

    @app.get("/")
    def index() -> str:
        """Show an upload form or an explicit offline demo launcher."""
        settings = get_settings()
        return render_template("index.html", mode=settings.mode)

    @app.post("/process")
    def process() -> tuple[str, int] | str:
        """Run the configured pipeline and render a human-readable protocol."""
        settings = get_settings()
        try:
            if settings.is_demo:
                protocol = process_meeting()
            else:
                upload = request.files.get("audio")
                if upload is None or not upload.filename:
                    raise ValueError("Выберите аудиофайл для обработки в API-режиме.")
                filename = secure_filename(upload.filename)
                suffix = Path(filename).suffix or Path(upload.filename).suffix
                with tempfile.TemporaryDirectory(prefix="meeting-minutes-") as temp_dir:
                    audio_path = Path(temp_dir) / f"meeting{suffix.lower()}"
                    upload.save(audio_path)
                    protocol = process_meeting(audio_path)

            result_id = uuid4().hex
            results: OrderedDict[str, bytes] = app.config["DOCX_RESULTS"]
            results[result_id] = build_protocol_docx(protocol)
            while len(results) > 5:
                results.popitem(last=False)
            return render_template(
                "result.html",
                protocol=protocol,
                result_id=result_id,
                mode=settings.mode,
            )
        except (RuntimeError, ValueError) as error:
            return render_template(
                "index.html",
                mode=settings.mode,
                error=str(error),
            ), 400

    @app.get("/download/<result_id>")
    def download(result_id: str):
        """Download one recently generated protocol as a DOCX file."""
        content = app.config["DOCX_RESULTS"].get(result_id)
        if content is None:
            abort(404)
        return send_file(
            BytesIO(content),
            mimetype=(
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ),
            as_attachment=True,
            download_name="meeting_protocol.docx",
        )

    @app.errorhandler(413)
    def file_too_large(_: object) -> tuple[str, int]:
        """Replace Flask's generic upload error with an actionable message."""
        settings = get_settings()
        return render_template(
            "index.html",
            mode=settings.mode,
            error="Файл превышает лимит 25 МБ.",
        ), 413

    return app


def main() -> None:
    """Run the local demonstration server with environment-based settings."""
    app = create_app()
    app.run(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "5000")),
        debug=False,
    )


if __name__ == "__main__":
    main()
