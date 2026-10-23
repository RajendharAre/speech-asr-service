"""Backend interface. Any speech engine implements `Transcriber`, so the API
layer and the evaluation script never depend on a specific model library."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    text: str
    language: str
    language_probability: float
    duration_s: float
    segments: List[Segment] = field(default_factory=list)


class Transcriber(ABC):
    name: str = "unnamed"

    @abstractmethod
    def transcribe(self, audio_path: str, language: Optional[str] = None) -> TranscriptionResult:
        """Transcribe the audio file at `audio_path`. `language=None` means auto-detect."""
