"""Unit tests for LOCAL_KZ device selection without loading the real model."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from meeting_minutes.local_kz import (
    LocalKazakhSTTError,
    _run_chunked_pipeline,
    transcribe_kazakh_audio,
)


class LocalKazakhProviderTests(unittest.TestCase):
    """Verify CPU selection and CUDA failure fallback in isolation."""

    def test_uses_cpu_when_cuda_is_unavailable(self) -> None:
        """Machines without CUDA get a clear warning and the float32 CPU path."""
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            float16="float16",
            float32="float32",
        )
        dependencies = {
            "torch": fake_torch,
            "soundfile": object(),
            "resample": object(),
            "model_class": object(),
            "processor_class": object(),
            "pipeline_factory": object(),
        }

        with (
            patch("meeting_minutes.local_kz._load_dependencies", return_value=dependencies),
            patch(
                "meeting_minutes.local_kz._load_audio",
                return_value=("samples", 12.5),
            ),
            patch(
                "meeting_minutes.local_kz._run_chunked_pipeline",
                return_value="Қазақша мәтін.",
            ) as run_pipeline,
        ):
            result = transcribe_kazakh_audio("meeting.ogg", model_id="model")

        self.assertEqual(result.device, "cpu")
        self.assertIn("CUDA is unavailable", result.warning or "")
        self.assertEqual(run_pipeline.call_args.kwargs["device"], "cpu")
        self.assertEqual(run_pipeline.call_args.kwargs["dtype"], "float32")

    def test_cuda_runtime_error_retries_on_cpu(self) -> None:
        """A CUDA inference failure falls back to CPU instead of aborting."""
        fake_cuda = SimpleNamespace(
            is_available=lambda: True,
            empty_cache=lambda: None,
        )
        fake_torch = SimpleNamespace(
            cuda=fake_cuda,
            float16="float16",
            float32="float32",
        )
        dependencies = {
            "torch": fake_torch,
            "soundfile": object(),
            "resample": object(),
            "model_class": object(),
            "processor_class": object(),
            "pipeline_factory": object(),
        }

        with (
            patch("meeting_minutes.local_kz._load_dependencies", return_value=dependencies),
            patch(
                "meeting_minutes.local_kz._load_audio",
                return_value=("samples", 12.5),
            ),
            patch(
                "meeting_minutes.local_kz._run_chunked_pipeline",
                side_effect=[RuntimeError("out of memory"), "Қазақша мәтін."],
            ) as run_pipeline,
        ):
            result = transcribe_kazakh_audio("meeting.ogg", model_id="model")

        self.assertEqual(result.device, "cpu")
        self.assertIn("retried on CPU", result.warning or "")
        self.assertEqual(run_pipeline.call_count, 2)
        self.assertEqual(run_pipeline.call_args_list[0].kwargs["device"], "cuda:0")
        self.assertEqual(run_pipeline.call_args_list[1].kwargs["device"], "cpu")

    def test_chunked_pipeline_uses_benchmarked_parameters(self) -> None:
        """The integration preserves Kazakh, 30-second, text-only inference."""
        calls: dict[str, object] = {}
        fake_model = SimpleNamespace(
            to=lambda device: calls.setdefault("model_device", device),
            eval=lambda: None,
        )
        model_class = SimpleNamespace(
            from_pretrained=lambda model_id, **kwargs: fake_model
        )
        processor_class = SimpleNamespace(
            from_pretrained=lambda model_id: SimpleNamespace(
                tokenizer="tokenizer",
                feature_extractor="feature_extractor",
            )
        )

        def pipeline_factory(task: str, **kwargs: object):
            calls["pipeline_task"] = task
            calls["pipeline_kwargs"] = kwargs

            def recognize(audio: object, **recognize_kwargs: object) -> dict[str, str]:
                calls["audio"] = audio
                calls["recognize_kwargs"] = recognize_kwargs
                return {"text": " Қазақша мәтін. "}

            return recognize

        fake_torch = SimpleNamespace(inference_mode=lambda: nullcontext())

        text = _run_chunked_pipeline(
            samples="samples",
            model_id="model",
            device="cuda:0",
            dtype="float16",
            torch=fake_torch,
            model_class=model_class,
            processor_class=processor_class,
            pipeline_factory=pipeline_factory,
        )

        pipeline_kwargs = calls["pipeline_kwargs"]
        recognize_kwargs = calls["recognize_kwargs"]
        self.assertEqual(text, "Қазақша мәтін.")
        self.assertEqual(calls["pipeline_task"], "automatic-speech-recognition")
        self.assertEqual(pipeline_kwargs["chunk_length_s"], 30)
        self.assertEqual(pipeline_kwargs["batch_size"], 1)
        self.assertEqual(recognize_kwargs["generate_kwargs"], {
            "language": "kk",
            "task": "transcribe",
        })
        self.assertFalse(recognize_kwargs["return_timestamps"])

    def test_missing_model_has_actionable_error(self) -> None:
        """An absent cache or failed first download names the configured model."""
        model_class = SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: (_ for _ in ()).throw(
                OSError("not cached")
            )
        )

        with self.assertRaisesRegex(LocalKazakhSTTError, "Check the model cache"):
            _run_chunked_pipeline(
                samples="samples",
                model_id="missing-model",
                device="cpu",
                dtype="float32",
                torch=SimpleNamespace(inference_mode=lambda: nullcontext()),
                model_class=model_class,
                processor_class=object(),
                pipeline_factory=object(),
            )
