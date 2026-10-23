from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional


@dataclass(frozen=True)
class Settings:
    model_size: str = "base"
    device: str = "cpu"
    compute_type: str = "int8"
    max_upload_mb: float = 25.0
    # Number of threads CTranslate2 uses inside the device for inference.
    # 0 means "auto" (CTranslate2 picks the value), which is the default and
    # preserves the behaviour of versions that had no such option.
    cpu_threads: int = 0
    # Temperature for decoding. None uses faster-whisper defaults (fallback enabled).
    # Set to 0.0 for deterministic, beam-search only decoding (no temperature fallback).
    temperature: Optional[float] = None

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)


def parse_cpu_threads(raw: str) -> int:
    """Parses the ASR_CPU_THREADS value.

    0 keeps CTranslate2's automatic thread selection. 1 is the useful setting
    for controlled experiments, because it removes thread-scheduling as a
    variable. Note that single-threaded execution does not by itself make
    decoding reproducible: faster-whisper falls back to temperature sampling
    when the model is uncertain, and sampling is stochastic regardless of the
    thread count. See scripts/reproducibility.py.
    """
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        raise ValueError(f"ASR_CPU_THREADS must be an integer, got {raw!r}") from None
    if value < 0:
        raise ValueError(f"ASR_CPU_THREADS must be >= 0 (0 means auto), got {value}")
    return value


def parse_temperature(raw: str) -> float:
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        raise ValueError(f"ASR_TEMPERATURE must be a float, got {raw!r}") from None
    if value < 0.0:
        raise ValueError(f"ASR_TEMPERATURE must be >= 0, got {value}")
    return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    temp_raw = os.environ.get("ASR_TEMPERATURE")
    temperature = parse_temperature(temp_raw) if temp_raw not in (None, "") else None
    return Settings(
        model_size=os.environ.get("ASR_MODEL_SIZE", "base"),
        device=os.environ.get("ASR_DEVICE", "cpu"),
        compute_type=os.environ.get("ASR_COMPUTE_TYPE", "int8"),
        max_upload_mb=float(os.environ.get("ASR_MAX_UPLOAD_MB", "25")),
        cpu_threads=parse_cpu_threads(os.environ.get("ASR_CPU_THREADS", "0")),
        temperature=temperature,
    )
