from types import SimpleNamespace

import pytest

from app.backends import faster_whisper_backend


class RecordingWhisperModel:
    """Stand-in for faster_whisper.WhisperModel that records constructor kwargs."""

    instances = []

    def __init__(self, model_size_or_path, **kwargs):
        self.model_size_or_path = model_size_or_path
        self.kwargs = kwargs
        RecordingWhisperModel.instances.append(self)


@pytest.fixture(autouse=True)
def fake_faster_whisper(monkeypatch):
    RecordingWhisperModel.instances = []
    fake_module = SimpleNamespace(WhisperModel=RecordingWhisperModel)
    monkeypatch.setattr(faster_whisper_backend, "import_module", lambda name: fake_module)
    return fake_module


def test_cpu_threads_is_forwarded_to_the_model():
    faster_whisper_backend.FasterWhisperTranscriber("small", "cpu", "int8", cpu_threads=1)
    model = RecordingWhisperModel.instances[-1]
    assert model.kwargs["cpu_threads"] == 1


def test_cpu_threads_defaults_to_auto():
    faster_whisper_backend.FasterWhisperTranscriber("small")
    assert RecordingWhisperModel.instances[-1].kwargs["cpu_threads"] == 0


def test_cpu_settings_are_preserved():
    transcriber = faster_whisper_backend.FasterWhisperTranscriber("tiny", "cpu", "int8", cpu_threads=2)
    model = RecordingWhisperModel.instances[-1]
    assert model.model_size_or_path == "tiny"
    assert model.kwargs["device"] == "cpu"
    assert model.kwargs["compute_type"] == "int8"
    assert transcriber.cpu_threads == 2
    assert transcriber.name == "faster-whisper-tiny"


def test_gpu_configuration_is_still_passed_through():
    faster_whisper_backend.FasterWhisperTranscriber("small", "cuda", "float16")
    model = RecordingWhisperModel.instances[-1]
    assert model.kwargs["device"] == "cuda"
    assert model.kwargs["compute_type"] == "float16"
    assert model.kwargs["cpu_threads"] == 0


def test_model_is_constructed_once_per_transcriber():
    faster_whisper_backend.FasterWhisperTranscriber("base")
    faster_whisper_backend.FasterWhisperTranscriber("base", cpu_threads=1)
    assert len(RecordingWhisperModel.instances) == 2


def test_get_transcriber_forwards_configured_cpu_threads(monkeypatch):
    import app.main as main
    from app.config import Settings

    captured = {}

    class FakeTranscriber:
        name = "fake"

        def __init__(self, model_size, device, compute_type, cpu_threads, temperature=None):
            captured["args"] = (model_size, device, compute_type, cpu_threads)
            captured["temperature"] = temperature

    monkeypatch.setattr(faster_whisper_backend, "FasterWhisperTranscriber", FakeTranscriber)
    monkeypatch.setattr(
        main,
        "get_settings",
        lambda: Settings(model_size="small", device="cpu", compute_type="int8", cpu_threads=1, temperature=0.0),
    )
    main.get_transcriber.cache_clear()
    try:
        main.get_transcriber()
    finally:
        main.get_transcriber.cache_clear()

    assert captured["args"] == ("small", "cpu", "int8", 1)
    assert captured["temperature"] == 0.0

