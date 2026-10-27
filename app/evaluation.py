"""Scoring helpers for the evaluation script: text normalisation, WER/CER,
real-time factor, and a markdown summary. Kept free of any model code so it
can be unit-tested without downloading weights."""
from __future__ import annotations

import csv
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jiwer


def normalize(text: str) -> str:
    """NFC, lowercase, punctuation and symbols -> space.

    Combining marks are deliberately kept: for Indic scripts they carry vowel
    signs, and a plain `\\w` regex would strip them and corrupt the text.
    """
    text = unicodedata.normalize("NFC", text).lower()
    text = "".join(" " if unicodedata.category(c)[0] in ("P", "S") else c for c in text)
    return " ".join(text.split())


@dataclass
class FileResult:
    audio_path: str
    language: Optional[str]
    reference: str
    hypothesis: str
    duration_s: float
    processing_s: float
    source: Optional[str] = None


# Languages whose expected output script is non-Latin. The value is the prefix of
# the Unicode character name, e.g. "DEVANAGARI LETTER KA".
EXPECTED_SCRIPTS = {"hi": "DEVANAGARI", "te": "TELUGU"}
_ZERO_WIDTH = {"\u200b", "\u200c", "\u200d", "\ufeff"}


def _is_letter(ch: str) -> bool:
    # L* letters and M* marks; marks carry Indic vowel signs and must count.
    return unicodedata.category(ch)[0] in ("L", "M")


def _is_zero_width(ch: str) -> bool:
    return ch in _ZERO_WIDTH or unicodedata.name(ch, "").startswith("VARIATION SELECTOR")


def _script_counts(text: str, language: Optional[str]) -> Optional[Tuple[int, int]]:
    """(letters in the expected script, all letters) or None if the language has
    no expected non-Latin script."""
    prefix = EXPECTED_SCRIPTS.get((language or "").strip().lower())
    if prefix is None:
        return None
    matched = total = 0
    for ch in unicodedata.normalize("NFC", text):
        if not _is_letter(ch) or _is_zero_width(ch):
            continue
        total += 1
        if unicodedata.name(ch, "").startswith(prefix):
            matched += 1
    return matched, total


def script_share(text: str, language: Optional[str]) -> Optional[float]:
    """Fraction of letters (and vowel-sign marks) that use the expected script.

    Hindi expects Devanagari, Telugu expects Telugu script. Other languages return
    None, as does text with no letters. Digits, punctuation, whitespace and
    zero-width characters are ignored, so a low value means the engine produced
    the right language in the wrong writing system.
    """
    counts = _script_counts(text, language)
    if counts is None:
        return None
    matched, total = counts
    return matched / total if total else None


def file_error_rates(reference: str, hypothesis: str) -> Tuple[float, float]:
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        # jiwer silently returns inconsistent numbers for an empty reference
        # (WER 1 but CER equal to the hypothesis length), so reject it instead.
        raise ValueError("Reference transcript is empty after normalisation")
    if not hyp:  # model returned nothing: every reference word/char is missed
        return 1.0, 1.0
    return jiwer.wer(ref, hyp), jiwer.cer(ref, hyp)


def corpus_scores(results: List[FileResult]) -> Dict:
    """Corpus-level WER/CER (total errors / total reference tokens), not a mean of
    per-file rates, so long clips weigh more than short ones."""
    if not results:
        raise ValueError("corpus_scores needs at least one result")
    refs = [normalize(r.reference) for r in results]
    hyps = [normalize(r.hypothesis) for r in results]
    if any(not ref for ref in refs):
        raise ValueError("All reference transcripts must be non-empty after normalisation")
    audio = sum(r.duration_s for r in results)
    proc = sum(r.processing_s for r in results)
    script_matched = script_total = 0
    for r in results:
        counts = _script_counts(r.hypothesis, r.language)
        if counts is not None:
            script_matched += counts[0]
            script_total += counts[1]
    return {
        "n_files": len(results),
        "wer": jiwer.wer(refs, hyps),
        "cer": jiwer.cer(refs, hyps),
        "script_share": script_matched / script_total if script_total else None,
        "audio_s": audio,
        "processing_s": proc,
        "rtf": proc / audio if audio > 0 else None,
    }


def summarize(results: List[FileResult]) -> Dict:
    overall = corpus_scores(results)
    groups: Dict[str, List[FileResult]] = defaultdict(list)
    for r in results:
        groups[r.language or "unspecified"].append(r)
    overall["by_language"] = {lang: corpus_scores(rs) for lang, rs in sorted(groups.items())}
    return overall


def summarize_by_source(results: List[FileResult]) -> Dict[str, Dict]:
    """Per-source summaries. Sources are never merged: FLEURS and your own
    recordings differ in speakers, conditions and difficulty."""
    groups: Dict[str, List[FileResult]] = defaultdict(list)
    for r in results:
        groups[r.source or "unspecified"].append(r)
    return {source: summarize(rs) for source, rs in sorted(groups.items())}


def load_manifest(path: str) -> List[Dict]:
    """CSV columns: audio_path, reference, language (optional), source (optional).
    Relative audio paths are resolved against the manifest's folder. A missing
    source is reported as "unspecified"."""
    base = Path(path).parent
    rows: List[Dict] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = {"audio_path", "reference"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing column(s): {sorted(missing)}")
        for line_no, row in enumerate(reader, start=2):
            reference = (row["reference"] or "").strip()
            if not normalize(reference):
                raise ValueError(f"Manifest line {line_no}: reference transcript is empty")
            audio = Path(row["audio_path"].strip())
            if not audio.is_absolute():
                audio = base / audio
            rows.append(
                {
                    "audio_path": str(audio),
                    "reference": reference,
                    "language": (row.get("language") or "").strip().lower() or None,
                    "source": (row.get("source") or "").strip().lower() or None,
                }
            )
    if not rows:
        raise ValueError("Manifest has no rows")
    return rows


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _share(x: Optional[float]) -> str:
    return "n/a" if x is None else _pct(x)


def _cell(value: Optional[float], n_files: int) -> str:
    """A metric cell that always shows the number of files it was computed from."""
    return f"{_pct(value)} ({n_files})" if value is not None else f"n/a ({n_files})"


def _warnings(variants: Dict[str, Dict]) -> List[str]:
    lines: List[str] = []
    for variant, summary in variants.items():
        for lang, s in summary["by_language"].items():
            ss = s["script_share"]
            if ss is not None and ss < 0.5 and s["wer"] > 0.8:
                lines.append(
                    f"- {variant}, {lang}: Script% {_pct(ss)} (< 50%) and WER {_pct(s['wer'])} (> 80%) "
                    f"over {s['n_files']} file(s). WER is dominated by script mismatch, not "
                    "recognition quality."
                )
    return lines


def to_markdown(sources: Dict[str, Dict[str, Dict]], footer: str = "") -> str:
    """Render one section per source.

    `sources` maps a source name ("fleurs", "own", "unspecified") to
    {variant name: summarize() output}. Sources are never combined into a single
    WER, because they differ in speakers and recording conditions.
    """
    lines: List[str] = []
    for source, variants in sources.items():
        lines += [f"## Source: {source}", ""]
        lines += ["| Variant | Files | WER | CER | Script% | RTF |", "|---|---|---|---|---|---|"]
        for variant, s in variants.items():
            rtf = f"{s['rtf']:.2f}" if s["rtf"] is not None else "n/a"
            lines.append(
                f"| {variant} | {s['n_files']} | {_pct(s['wer'])} | {_pct(s['cer'])} | "
                f"{_share(s['script_share'])} | {rtf} |"
            )

        langs = sorted({lang for s in variants.values() for lang in s["by_language"]})
        if langs:
            header = "| Variant | " + " | ".join(langs) + " |"
            rule = "|---|" + "---|" * len(langs)
            lines += ["", "WER by language (files in parentheses):", "", header, rule]
            for variant, s in variants.items():
                cells = [
                    _cell(s["by_language"][lang]["wer"], s["by_language"][lang]["n_files"])
                    if lang in s["by_language"]
                    else "-"
                    for lang in langs
                ]
                lines.append(f"| {variant} | " + " | ".join(cells) + " |")

            lines += ["", "Script% by language (files in parentheses):", "", header, rule]
            for variant, s in variants.items():
                cells = [
                    _cell(s["by_language"][lang]["script_share"], s["by_language"][lang]["n_files"])
                    if lang in s["by_language"]
                    else "-"
                    for lang in langs
                ]
                lines.append(f"| {variant} | " + " | ".join(cells) + " |")

        warnings = _warnings(variants)
        if warnings:
            lines += ["", "Warnings:", ""] + warnings
        lines.append("")

    if footer:
        lines += [footer]
    return "\n".join(lines).strip("\n") + "\n"
