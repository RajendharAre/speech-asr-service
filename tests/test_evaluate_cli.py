import csv

import pytest

from app.backends.base import Transcriber, TranscriptionResult
from scripts import evaluate


class FakeBackend(Transcriber):
    def __init__(self, size, device, compute_type):
        self.name = f"fake-{size}"

    def transcribe(self, audio_path, language=None):
        return TranscriptionResult("hello world", language or "en", 0.9, 4.0, [])


def make_manifest(tmp_path, with_files=True):
    if with_files:
        (tmp_path / "a.wav").write_bytes(b"x")
        (tmp_path / "b.wav").write_bytes(b"x")
    m = tmp_path / "m.csv"
    m.write_text("audio_path,reference,language\na.wav,hello world,en\nb.wav,hello there,en\n", encoding="utf-8")
    return m


def test_cli_writes_per_file_csv_and_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "FasterWhisperTranscriber", FakeBackend)
    out = tmp_path / "out"
    evaluate.main(["--manifest", str(make_manifest(tmp_path)), "--model-sizes", "tiny", "base", "--out-dir", str(out)])

    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "| tiny | 2 |" in summary and "| base | 2 |" in summary
    assert "25.0%" in summary  # 1 wrong word out of 4 reference words

    with open(out / "per_file.csv", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4
    assert {r["model"] for r in rows} == {"tiny", "base"}


def test_cli_fails_cleanly_when_audio_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "FasterWhisperTranscriber", FakeBackend)
    with pytest.raises(SystemExit):
        evaluate.main(["--manifest", str(make_manifest(tmp_path, with_files=False)), "--out-dir", str(tmp_path / "out")])
