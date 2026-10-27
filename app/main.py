from __future__ import annotations

import logging
import os
import re
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import RedirectResponse

from .backends.base import Transcriber
from .config import Settings, get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("speech_service")

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}
LANGUAGE_RE = re.compile(r"[a-z]{2,3}")
CHUNK_BYTES = 1024 * 1024
MAX_PROMPT_CHARS = 200

app = FastAPI(
    title="Speech ASR Service",
    version="0.1.0",
    description="Upload audio, get a transcript, timing and real-time factor.",
)


@lru_cache(maxsize=1)
def get_transcriber() -> Transcriber:
    """Loads the model once per process, on first use."""
    from .backends.faster_whisper_backend import FasterWhisperTranscriber

    s = get_settings()
    logger.info(
        "Loading model=%s device=%s compute_type=%s cpu_threads=%s temperature=%s",
        s.model_size,
        s.device,
        s.compute_type,
        s.cpu_threads,
        s.temperature,
    )
    return FasterWhisperTranscriber(
        s.model_size,
        s.device,
        s.compute_type,
        s.cpu_threads,
        temperature=s.temperature,
    )


@app.get("/")
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health(settings: Settings = Depends(get_settings)):
    return {"status": "ok", "model_size": settings.model_size, "device": settings.device}


@app.post("/transcribe")
def transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = Form(default=None),
    initial_prompt: Optional[str] = Form(default=None),
    settings: Settings = Depends(get_settings),
    transcriber: Transcriber = Depends(get_transcriber),
):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(415, f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}")

    if language is not None:
        language = language.strip().lower()
        if not LANGUAGE_RE.fullmatch(language):
            raise HTTPException(400, "language must be a 2-3 letter code such as 'en', 'hi' or 'te'")

    if initial_prompt is not None:
        initial_prompt = initial_prompt.strip() or None
    if initial_prompt is not None and len(initial_prompt) > MAX_PROMPT_CHARS:
        raise HTTPException(400, f"initial_prompt must be at most {MAX_PROMPT_CHARS} characters")

    tmp_path: Optional[str] = None
    try:
        total = 0
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp_path = tmp.name
            while True:
                chunk = file.file.read(CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > settings.max_upload_bytes:
                    raise HTTPException(413, f"File too large (limit {settings.max_upload_mb} MB)")
                tmp.write(chunk)
        if total == 0:
            raise HTTPException(400, "Uploaded file is empty")

        started = time.perf_counter()
        try:
            result = transcriber.transcribe(tmp_path, language=language, initial_prompt=initial_prompt)
        except HTTPException:
            raise
        except ValueError:
            logger.exception("Transcription failed for %s", file.filename)
            raise HTTPException(400, "Transcription failed. The audio may be corrupt or unsupported.")
        except Exception:
            logger.exception("Transcription failed for %s", file.filename)
            raise HTTPException(500, "Transcription failed. The audio may be corrupt or unsupported.")
        elapsed = time.perf_counter() - started
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    rtf = elapsed / result.duration_s if result.duration_s > 0 else None
    return {
        "text": result.text,
        "language": result.language,
        "language_probability": round(result.language_probability, 3),
        "duration_s": round(result.duration_s, 2),
        "processing_time_s": round(elapsed, 3),
        "real_time_factor": round(rtf, 3) if rtf is not None else None,
        "model": transcriber.name,
        "segments": [
            {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text} for s in result.segments
        ],
    }
