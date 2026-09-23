"""Optional local Kazakh speech-to-text provider."""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16_000
CHUNK_LENGTH_SECONDS = 30


class LocalKazakhSTTError(RuntimeError):
    """Raised when local Kazakh transcription cannot be completed."""


class LocalKazakhDependenciesError(LocalKazakhSTTError):
    """Raised when optional local STT dependencies are unavailable."""


@dataclass(frozen=True, slots=True)
class LocalKazakhResult:
    """Text and runtime metadata returned by the local Kazakh provider."""

    text: str
    device: str
    duration_seconds: float
    warning: str | None = None


def transcribe_kazakh_audio(
    audio_path: str | Path,
    *,
    model_id: str,
) -> LocalKazakhResult:
    """Transcribe Kazakh audio locally with the benchmarked chunked pipeline."""

    dependencies = _load_dependencies()
    torch = dependencies["torch"]
    samples, duration_seconds = _load_audio(
        Path(audio_path),
        soundfile=dependencies["soundfile"],
        torch=torch,
        resample=dependencies["resample"],
    )
    pipeline_arguments = {
        "samples": samples,
        "model_id": model_id,
        "torch": torch,
        "model_class": dependencies["model_class"],
        "processor_class": dependencies["processor_class"],
        "pipeline_factory": dependencies["pipeline_factory"],
    }

    if torch.cuda.is_available():
        try:
            text = _run_chunked_pipeline(
                **pipeline_arguments,
                device="cuda:0",
                dtype=torch.float16,
            )
            return LocalKazakhResult(text, "cuda:0", duration_seconds)
        except RuntimeError as error:
            logger.warning(
                "LOCAL_KZ CUDA inference failed (%s); retrying on CPU.",
                type(error).__name__,
            )
            gc.collect()
            torch.cuda.empty_cache()
            warning = (
                "LOCAL_KZ: CUDA inference failed; transcription was retried on CPU."
            )
    else:
        logger.warning("LOCAL_KZ: CUDA is unavailable; using the slower CPU fallback.")
        warning = "LOCAL_KZ: CUDA is unavailable; transcription is running on CPU."

    try:
        text = _run_chunked_pipeline(
            **pipeline_arguments,
            device="cpu",
            dtype=torch.float32,
        )
    except LocalKazakhSTTError:
        raise
    except Exception as error:
        raise LocalKazakhSTTError(
            f"LOCAL_KZ CPU transcription failed: {type(error).__name__}: {error}"
        ) from error

    return LocalKazakhResult(text, "cpu", duration_seconds, warning)


def _load_dependencies() -> dict[str, Any]:
    """Import heavy optional dependencies only when LOCAL_KZ is selected."""

    try:
        import soundfile
        import torch
        from torchaudio.functional import resample
        from transformers import (
            AutoModelForSpeechSeq2Seq,
            AutoProcessor,
            pipeline,
        )
    except (ImportError, ModuleNotFoundError, OSError) as error:
        raise LocalKazakhDependenciesError(
            "LOCAL_KZ optional dependencies are not installed. "
            'Install them with: python -m pip install -e ".[local-kz]"'
        ) from error

    return {
        "soundfile": soundfile,
        "torch": torch,
        "resample": resample,
        "model_class": AutoModelForSpeechSeq2Seq,
        "processor_class": AutoProcessor,
        "pipeline_factory": pipeline,
    }


def _load_audio(
    audio_path: Path,
    *,
    soundfile: Any,
    torch: Any,
    resample: Any,
) -> tuple[Any, float]:
    """Decode audio to mono 16 kHz float samples expected by Whisper."""

    try:
        samples, sample_rate = soundfile.read(
            str(audio_path),
            dtype="float32",
            always_2d=True,
        )
    except Exception as error:
        raise LocalKazakhSTTError(
            "LOCAL_KZ could not decode the audio file. Use a valid OGG, WAV, or FLAC file."
        ) from error

    if samples.size == 0 or sample_rate <= 0:
        raise LocalKazakhSTTError("LOCAL_KZ received an empty audio file.")

    mono_samples = samples.mean(axis=1)
    duration_seconds = len(mono_samples) / float(sample_rate)
    tensor = torch.from_numpy(mono_samples)
    if sample_rate != TARGET_SAMPLE_RATE:
        tensor = resample(tensor, sample_rate, TARGET_SAMPLE_RATE)

    return tensor.cpu().numpy(), duration_seconds


def _run_chunked_pipeline(
    *,
    samples: Any,
    model_id: str,
    device: str,
    dtype: Any,
    torch: Any,
    model_class: Any,
    processor_class: Any,
    pipeline_factory: Any,
) -> str:
    """Run 30-second chunks with Kazakh language and text-only output."""

    try:
        model = model_class.from_pretrained(
            model_id,
            dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(device)
        model.eval()
        processor = processor_class.from_pretrained(model_id)
        recognizer = pipeline_factory(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            dtype=dtype,
            device=device,
            chunk_length_s=CHUNK_LENGTH_SECONDS,
            batch_size=1,
            ignore_warning=True,
        )
        with torch.inference_mode():
            output = recognizer(
                {"array": samples, "sampling_rate": TARGET_SAMPLE_RATE},
                generate_kwargs={"language": "kk", "task": "transcribe"},
                return_timestamps=False,
            )
    except OSError as error:
        raise LocalKazakhSTTError(
            f"LOCAL_KZ could not load model '{model_id}'. "
            "Check the model cache and network access for the first download."
        ) from error

    text = str(output.get("text", "")).strip()
    if not text:
        raise LocalKazakhSTTError("LOCAL_KZ returned an empty transcript.")
    return text
