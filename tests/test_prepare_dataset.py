import pytest

from scripts import prepare_dataset as prep


def row(row_id, samples=10 * 16000, **extra):
    r = {"id": row_id, "num_samples": samples}
    r.update(extra)
    return r


def test_select_rows_is_deterministic_for_a_seed():
    rows = [row(i) for i in range(1, 51)]
    a = prep.select_rows(rows, 8, seed=0, max_seconds=20)
    b = prep.select_rows(list(reversed(rows)), 8, seed=0, max_seconds=20)
    assert [r["id"] for r in a] == [r["id"] for r in b]
    assert len(a) == 8


def test_select_rows_drops_clips_over_the_limit():
    rows = [row(1, samples=25 * 16000), row(2, samples=20 * 16000), row(3, samples=5 * 16000)]
    assert [r["id"] for r in prep.select_rows(rows, 8, seed=0, max_seconds=20)] == [2, 3]


def test_select_rows_dedupes_ids_keeping_the_first_row():
    rows = [row(7, gender="FEMALE"), row(7, gender="MALE"), row(9)]
    chosen = prep.select_rows(rows, 8, seed=0, max_seconds=20)
    assert [r["id"] for r in chosen] == [7, 9]
    assert chosen[0]["gender"] == "FEMALE"


def test_select_rows_returns_everything_when_fewer_than_requested():
    assert len(prep.select_rows([row(1)], 8, seed=0, max_seconds=20)) == 1


def test_clip_filename_zero_pads_the_id():
    assert prep.clip_filename("hi_in", 179) == "hi_in_00179.wav"


def test_reference_text_prefers_raw_transcription():
    row_with_both = {"id": 1, "raw_transcription": " नमस्ते। ", "transcription": "नमस्ते"}
    assert prep.reference_text(row_with_both) == "नमस्ते।"
    assert prep.reference_text({"id": 1, "transcription": "नमस्ते"}) == "नमस्ते"


def test_reference_text_rejects_rows_without_text():
    with pytest.raises(ValueError, match="no transcription"):
        prep.reference_text({"id": 3})


def test_row_duration_prefers_num_samples_then_duration():
    assert prep.row_duration({"num_samples": 160000}) == 10.0
    assert prep.row_duration({"duration": 7.5}) == 7.5
    assert prep.row_duration({}) is None


def test_audio_bytes_accepts_bytes_and_paths(tmp_path):
    assert prep.audio_bytes({"bytes": b"abc"}) == b"abc"
    p = tmp_path / "a.wav"
    p.write_bytes(b"xyz")
    assert prep.audio_bytes({"path": str(p)}) == b"xyz"
    with pytest.raises(ValueError):
        prep.audio_bytes({"array": [0, 1]})
