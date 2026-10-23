import pytest

from app.config import Settings, get_settings, parse_cpu_threads


@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_cpu_threads_defaults_to_auto(monkeypatch):
    monkeypatch.delenv("ASR_CPU_THREADS", raising=False)
    assert get_settings().cpu_threads == 0


def test_cpu_threads_can_be_overridden(monkeypatch):
    monkeypatch.setenv("ASR_CPU_THREADS", "1")
    assert get_settings().cpu_threads == 1


def test_cpu_threads_accepts_surrounding_whitespace(monkeypatch):
    monkeypatch.setenv("ASR_CPU_THREADS", " 4 ")
    assert get_settings().cpu_threads == 4


@pytest.mark.parametrize("raw", ["four", "", "1.5", "2x"])
def test_cpu_threads_rejects_non_integers(monkeypatch, raw):
    monkeypatch.setenv("ASR_CPU_THREADS", raw)
    with pytest.raises(ValueError, match="ASR_CPU_THREADS must be an integer"):
        get_settings()


def test_cpu_threads_rejects_negative(monkeypatch):
    monkeypatch.setenv("ASR_CPU_THREADS", "-1")
    with pytest.raises(ValueError, match="must be >= 0"):
        get_settings()


def test_parse_cpu_threads_is_directly_usable():
    assert parse_cpu_threads("0") == 0
    assert parse_cpu_threads("8") == 8


def test_settings_defaults_preserve_existing_behaviour():
    s = Settings()
    assert (s.model_size, s.device, s.compute_type) == ("base", "cpu", "int8")
    assert s.cpu_threads == 0
    assert s.max_upload_bytes == int(25.0 * 1024 * 1024)
