import csv

import pytest

from app.backends.base import Transcriber, TranscriptionResult
from scripts import evaluate


class FakeBackend(Transcriber):
    instances = []

    def __init__(self, size, device, compute_type):
        self.name = f"fake-{size}"
        self.initial_prompts = []
        FakeBackend.instances.append(self)

    def transcribe(self, audio_path, language=None, initial_prompt=None):
        self.initial_prompts.append(initial_prompt)
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


def test_cli_writes_script_share_and_source_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "FasterWhisperTranscriber", FakeBackend)
    (tmp_path / "a.wav").write_bytes(b"x")
    m = tmp_path / "m.csv"
    m.write_text(
        "audio_path,reference,language,source\na.wav,hello world,en,fleurs\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    evaluate.main(["--manifest", str(m), "--out-dir", str(out)])

    with open(out / "per_file.csv", newline="", encoding="utf-8-sig") as f:
        row = next(csv.DictReader(f))
    assert row["source"] == "fleurs"
    assert row["script_share"] == ""  # English has no expected non-Latin script


def test_cli_prompt_variant_reuses_one_model_and_writes_prompt_rows(tmp_path, monkeypatch):
    FakeBackend.instances = []
    monkeypatch.setattr(evaluate, "FasterWhisperTranscriber", FakeBackend)
    (tmp_path / "a.wav").write_bytes(b"x")
    (tmp_path / "b.wav").write_bytes(b"x")
    m = tmp_path / "m.csv"
    m.write_text(
        "audio_path,reference,language,source\na.wav,hello world,hi,fleurs\nb.wav,hello world,hi,fleurs\n",
        encoding="utf-8",
    )
    prompts = tmp_path / "prompts.json"
    prompts.write_text('{"hi": "यह एक हिंदी वाक्य है।"}', encoding="utf-8")
    out = tmp_path / "out"

    evaluate.main(
        [
            "--manifest", str(m),
            "--model-sizes", "tiny",
            "--prompts-json", str(prompts),
            "--out-dir", str(out),
        ]
    )

    assert len(FakeBackend.instances) == 1
    assert "यह एक हिंदी वाक्य है।" in FakeBackend.instances[0].initial_prompts

    with open(out / "per_file.csv", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert {r["model"] for r in rows} == {"tiny", "tiny+prompt"}
    assert len(rows) == 4  # 2 clips x 2 variants

    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "| tiny | 2 |" in summary and "| tiny+prompt | 2 |" in summary
    assert "## Source: fleurs" in summary


def test_load_prompts_strips_blank_entries(tmp_path):
    p = tmp_path / "p.json"
    p.write_text('{"hi": "  नमस्ते  ", "te": "   "}', encoding="utf-8")
    assert evaluate.load_prompts(str(p)) == {"hi": "नमस्ते"}


def test_load_prompts_rejects_overlong_prompt(tmp_path):
    p = tmp_path / "p.json"
    p.write_text('{"hi": "' + "x" * 201 + '"}', encoding="utf-8")
    with pytest.raises(ValueError, match="200"):
        evaluate.load_prompts(str(p))
