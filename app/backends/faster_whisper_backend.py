from __future__ import annotations

from importlib import import_module
from typing import Optional, Union

from .base import Segment, Transcriber, TranscriptionResult


class FasterWhisperTranscriber(Transcriber):
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = 0,
        temperature: Union[float, tuple[float, ...], None] = None,
    ):
        # Imported here so tests and CI can run without the heavy dependency installed.
        WhisperModel = import_module("faster_whisper").WhisperModel

        self.name = f"faster-whisper-{model_size}"
        self.cpu_threads = cpu_threads
        self.temperature = temperature
        # faster-whisper forwards cpu_threads to CTranslate2 as intra_threads,
        # the number of threads used inside the device. On CUDA that is GPU-side
        # parallelism rather than CPU threads, so 0 (auto) is the safe default.
        self._model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
        )

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> TranscriptionResult:
        transcribe_kwargs = {"language": language, "vad_filter": True}
        if self.temperature is not None:
            transcribe_kwargs["temperature"] = self.temperature
            # Keep decoding deterministic when a fixed temperature is requested
            transcribe_kwargs["beam_size"] = 5
        segments_iter, info = self._model.transcribe(audio_path, **transcribe_kwargs)
        segments = [Segment(s.start, s.end, s.text.strip()) for s in segments_iter]  # generator: consume it
        text = " ".join(s.text for s in segments).strip()
        return TranscriptionResult(
            text=text,
            language=info.language,
            language_probability=float(info.language_probability),
            duration_s=float(info.duration),
            segments=segments,
        )
