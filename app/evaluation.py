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
    return {
        "n_files": len(results),
        "wer": jiwer.wer(refs, hyps),
        "cer": jiwer.cer(refs, hyps),
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


def load_manifest(path: str) -> List[Dict]:
    """CSV columns: audio_path, reference, language (optional). Relative audio paths
    are resolved against the manifest's folder."""
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
                }
            )
    if not rows:
        raise ValueError("Manifest has no rows")
    return rows


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def to_markdown(summaries: Dict[str, Dict], footer: str = "") -> str:
    lines = ["| Model | Files | WER | CER | RTF |", "|---|---|---|---|---|"]
    for model, s in summaries.items():
        rtf = f"{s['rtf']:.2f}" if s["rtf"] is not None else "n/a"
        lines.append(f"| {model} | {s['n_files']} | {_pct(s['wer'])} | {_pct(s['cer'])} | {rtf} |")

    langs = sorted({lang for s in summaries.values() for lang in s["by_language"]})
    if len(langs) > 1:
        lines += ["", "WER by language:", "", "| Model | " + " | ".join(langs) + " |", "|---|" + "---|" * len(langs)]
        for model, s in summaries.items():
            cells = [_pct(s["by_language"][l]["wer"]) if l in s["by_language"] else "-" for l in langs]
            lines.append(f"| {model} | " + " | ".join(cells) + " |")

    if footer:
        lines += ["", footer]
    return "\n".join(lines) + "\n"
