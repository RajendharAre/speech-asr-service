import pytest

from app.evaluation import (
    FileResult,
    corpus_scores,
    file_error_rates,
    load_manifest,
    normalize,
    summarize,
    to_markdown,
)


def test_normalize_lowercases_and_strips_punctuation():
    assert normalize("Hello,  World!") == "hello world"


def test_normalize_keeps_indic_vowel_signs_and_strips_danda():
    assert normalize("తెలుగు, భాష!") == "తెలుగు భాష"
    assert normalize("नमस्ते। दुनिया") == "नमस्ते दुनिया"


def make(ref, hyp, lang=None, dur=10.0, proc=5.0):
    return FileResult("a.wav", lang, ref, hyp, dur, proc)


def test_wer_single_substitution():
    s = corpus_scores([make("the cat sat", "the cat sit")])
    assert s["wer"] == pytest.approx(1 / 3)


def test_wer_is_corpus_level_not_mean_of_files():
    perfect = make("a b", "a b")                                   # 0 errors / 2 words
    bad = make("a b c d e f g h", "a b c d x y z w")                # 4 errors / 8 words
    s = corpus_scores([perfect, bad])
    assert s["wer"] == pytest.approx(4 / 10)                        # a mean of file WERs would give 0.25


def test_empty_hypothesis_counts_as_total_miss():
    s = corpus_scores([make("one two three", "")])
    assert s["wer"] == pytest.approx(1.0)


def test_file_error_rates_empty_hypothesis_is_total_miss():
    assert file_error_rates("one two three", "") == (1.0, 1.0)


def test_insertions_can_push_wer_above_100_percent():
    s = corpus_scores([make("a b", "a b c d e f")])
    assert s["wer"] == pytest.approx(2.0)


@pytest.mark.parametrize("reference", ["", "   ", "!!! ..."])
def test_empty_reference_is_rejected(reference):
    # jiwer returns inconsistent values for an empty reference (WER 1 but CER
    # equal to the hypothesis length), so scoring must refuse it explicitly.
    with pytest.raises(ValueError, match="empty after normalisation"):
        file_error_rates(reference, "hello")


def test_corpus_scores_rejects_empty_reference():
    with pytest.raises(ValueError, match="non-empty"):
        corpus_scores([make("a b", "a b"), make("", "x")])


def test_corpus_scores_rejects_no_results():
    with pytest.raises(ValueError, match="at least one"):
        corpus_scores([])


def test_rtf_is_none_when_audio_duration_is_zero():
    s = corpus_scores([make("a", "a", dur=0, proc=1)])
    assert s["rtf"] is None


def test_rtf_is_total_processing_over_total_audio():
    s = corpus_scores([make("a", "a", dur=10, proc=5), make("b", "b", dur=30, proc=5)])
    assert s["rtf"] == pytest.approx(10 / 40)


def test_summary_groups_by_language():
    s = summarize([make("a b", "a b", "en"), make("c d", "c x", "te")])
    assert set(s["by_language"]) == {"en", "te"}
    assert s["by_language"]["en"]["wer"] == 0
    assert s["by_language"]["te"]["wer"] == pytest.approx(0.5)


def test_markdown_contains_model_and_percentages():
    md = to_markdown({"base": summarize([make("a b", "a b", "en"), make("c d", "c x", "te")])}, "note")
    assert "| base | 2 |" in md and "%" in md and "note" in md and "WER by language" in md


def test_load_manifest_resolves_relative_paths(tmp_path):
    m = tmp_path / "m.csv"
    m.write_text("audio_path,reference,language\nclips/a.wav,hello there,EN\n", encoding="utf-8")
    rows = load_manifest(str(m))
    assert rows[0]["audio_path"] == str(tmp_path / "clips" / "a.wav")
    assert rows[0]["language"] == "en"


def test_load_manifest_rejects_missing_column(tmp_path):
    m = tmp_path / "m.csv"
    m.write_text("audio_path\na.wav\n", encoding="utf-8")
    with pytest.raises(ValueError, match="reference"):
        load_manifest(str(m))


def test_load_manifest_rejects_empty_reference(tmp_path):
    m = tmp_path / "m.csv"
    m.write_text("audio_path,reference\na.wav,  \n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_manifest(str(m))
