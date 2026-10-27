import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from app.backends.base import Segment, Transcriber, TranscriptionResult
from app.config import Settings, get_settings
from app.main import app, get_transcriber

AUDIO = b"RIFF" + b"\x00" * 2000


class FakeTranscriber(Transcriber):
    name = "fake-model"

    def __init__(self, fail=False):
        self.fail = fail
        self.paths = []
        self.language = "unset"
        self.prompt = "unset"

    def transcribe(self, audio_path, language=None, initial_prompt=None):
        self.paths.append(audio_path)
        self.language = language
        self.prompt = initial_prompt
        assert os.path.exists(audio_path), "temp file must exist while transcribing"
        if self.fail:
            raise RuntimeError("secret internal detail")
        return TranscriptionResult(
            text="hello world",
            language=language or "en",
            language_probability=0.98,
            duration_s=2.0,
            segments=[Segment(0.0, 2.0, "hello world")],
        )


@pytest.fixture
def fake():
    return FakeTranscriber()


@pytest.fixture
def client(fake, tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))  # so we can assert nothing is left behind
    app.dependency_overrides[get_transcriber] = lambda: fake
    app.dependency_overrides[get_settings] = lambda: Settings(max_upload_mb=1.0)
    yield TestClient(app)
    app.dependency_overrides.clear()


def post(client, name="clip.wav", content=AUDIO, **data):
    return client.post("/transcribe", files={"file": (name, content, "audio/wav")}, data=data)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_transcribe_returns_expected_fields(client):
    r = post(client)
    body = r.json()
    assert r.status_code == 200
    assert body["text"] == "hello world"
    assert body["model"] == "fake-model"
    assert body["real_time_factor"] is not None
    assert body["segments"] == [{"start": 0.0, "end": 2.0, "text": "hello world"}]


def test_language_is_forwarded_and_normalised(client, fake):
    r = post(client, language=" TE ")
    assert r.status_code == 200
    assert fake.language == "te"
    assert r.json()["language"] == "te"


def test_no_language_means_auto_detect(client, fake):
    post(client)
    assert fake.language is None


def test_initial_prompt_is_forwarded_stripped(client, fake):
    r = post(client, language="hi", initial_prompt="  यह एक हिंदी वाक्य है।  ")
    assert r.status_code == 200
    assert fake.prompt == "यह एक हिंदी वाक्य है।"


def test_missing_initial_prompt_means_none(client, fake):
    post(client)
    assert fake.prompt is None


def test_blank_initial_prompt_means_none(client, fake):
    r = post(client, initial_prompt="   ")
    assert r.status_code == 200
    assert fake.prompt is None


def test_overlong_initial_prompt_rejected_before_transcribing(client, fake):
    r = post(client, initial_prompt="x" * 201)
    assert r.status_code == 400
    assert "200" in r.json()["detail"]
    assert fake.paths == []


def test_initial_prompt_at_limit_is_accepted(client, fake):
    r = post(client, initial_prompt="x" * 200)
    assert r.status_code == 200
    assert fake.prompt == "x" * 200


def test_invalid_language_rejected_before_transcribing(client, fake):
    r = post(client, language="english-us")
    assert r.status_code == 400
    assert fake.paths == []


def test_unsupported_extension_rejected(client, fake):
    r = post(client, name="notes.txt")
    assert r.status_code == 415
    assert fake.paths == []


def test_empty_file_rejected(client, fake, tmp_path):
    r = post(client, content=b"")
    assert r.status_code == 400
    assert fake.paths == []
    assert list(tmp_path.iterdir()) == []


def test_oversize_rejected_and_cleaned_up(client, fake, tmp_path):
    app.dependency_overrides[get_settings] = lambda: Settings(max_upload_mb=0.001)  # ~1 KB
    r = post(client)
    assert r.status_code == 413
    assert fake.paths == []
    assert list(tmp_path.iterdir()) == []


def test_temp_file_removed_after_success(client, fake, tmp_path):
    post(client)
    assert len(fake.paths) == 1
    assert not os.path.exists(fake.paths[0])
    assert list(tmp_path.iterdir()) == []


def test_model_failure_returns_generic_500_and_cleans_up(client, fake, tmp_path):
    fake.fail = True
    r = post(client)
    assert r.status_code == 500
    assert "secret internal detail" not in r.text
    assert list(tmp_path.iterdir()) == []


def test_corrupt_audio_returns_400(client, fake, tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise ValueError("av could not open container")

    fake.transcribe = boom
    r = post(client)
    assert r.status_code == 400
    assert "corrupt" in r.json()["detail"].lower()
    assert list(tmp_path.iterdir()) == []
