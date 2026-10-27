"""Export a small, reproducible sample of Google FLEURS test clips.

    python -m scripts.prepare_dataset --per-language 8 --seed 0

Downloads the test split for the requested language configs, keeps clips of at
most --max-seconds, samples them deterministically from a sorted id list, writes
16 kHz mono 16-bit WAV files to data/audio/fleurs/ and a manifest with
source=fleurs.

The first run downloads the whole test split for each language (a few hundred MB
per language); later runs reuse the Hugging Face cache.

Needs the data extras (not the API runtime):

    pip install -r requirements-data.txt

Field names were checked against the dataset's own test.tsv and its Croissant
metadata: id, num_samples, audio, transcription, raw_transcription, gender.
raw_transcription (punctuated) is used as the reference when present.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from scripts.fetch_clips import whisper_language

DATASET = "google/fleurs"
SAMPLE_RATE = 16000


def row_duration(row: Dict, sample_rate: int = SAMPLE_RATE) -> Optional[float]:
    samples = row.get("num_samples")
    if isinstance(samples, int) and samples > 0:
        return samples / sample_rate
    duration = row.get("duration")
    if isinstance(duration, (int, float)):
        return float(duration)
    return None


def reference_text(row: Dict) -> str:
    for key in ("raw_transcription", "transcription"):
        text = (row.get(key) or "").strip()
        if text:
            return text
    raise ValueError(f"Row {row.get('id')!r} has no transcription")


def clip_filename(config: str, row_id) -> str:
    return f"{config}_{int(row_id):05d}.wav"


def select_rows(
    rows: Sequence[Dict],
    per_language: int,
    seed: int,
    max_seconds: float,
) -> List[Dict]:
    """Deterministic sample: drop duplicate ids and clips over the limit, sort by
    id, then draw with `random.Random(seed)`.

    Sorting first keeps the draw stable for a given dataset revision instead of
    depending on parquet row order.
    """
    seen = set()
    candidates: List[Dict] = []
    for row in rows:
        row_id = row.get("id")
        if row_id is None or row_id in seen:
            continue
        duration = row_duration(row)
        if duration is None or duration > max_seconds:
            continue
        seen.add(row_id)
        candidates.append(row)

    candidates.sort(key=lambda r: r["id"])
    if len(candidates) <= per_language:
        return candidates
    picked = random.Random(seed).sample(candidates, per_language)
    return sorted(picked, key=lambda r: r["id"])


def audio_bytes(audio) -> bytes:
    """Audio(decode=False) yields {"bytes": ...} or {"path": ...}."""
    if isinstance(audio, dict):
        if audio.get("bytes"):
            return audio["bytes"]
        if audio.get("path"):
            return Path(audio["path"]).read_bytes()
    raise ValueError("Unsupported audio column; expected datasets.Audio(decode=False)")


def write_wav(raw: bytes, dest: Path) -> float:
    """Re-encode to 16 kHz mono 16-bit PCM WAV and return its duration."""
    import av

    dest.parent.mkdir(parents=True, exist_ok=True)
    with av.open(io.BytesIO(raw)) as container:
        resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
        frames = []
        for frame in container.decode(audio=0):
            frames.extend(resampler.resample(frame))
        frames.extend(resampler.resample(None) or [])
        total = sum(f.samples for f in frames)

        with av.open(str(dest), mode="w") as out:
            out_stream = out.add_stream("pcm_s16le", rate=SAMPLE_RATE)
            out_stream.layout = "mono"
            for frame in frames:
                for packet in out_stream.encode(frame):
                    out.mux(packet)
            for packet in out_stream.encode(None):
                out.mux(packet)
    return total / SAMPLE_RATE


def load_split(config: str, split: str):
    """Load one FLEURS config/split, falling back to the parquet mirror.

    Newer `datasets` releases no longer run the dataset's Python loading script,
    so `load_dataset("google/fleurs", config)` can fail. The repo also publishes
    pre-converted parquet files; loading those directly is the fallback.
    """
    from datasets import Audio, load_dataset

    try:
        ds = load_dataset(DATASET, config, split=split)
    except Exception as exc:  # noqa: BLE001 - report the real reason and fall back
        print(f"  load_dataset({DATASET}, {config}) failed: {exc}")
        parquet = f"hf://datasets/{DATASET}/parquet-data/{config}/{split}-00000-of-00001.parquet"
        print(f"  falling back to {parquet}")
        ds = load_dataset("parquet", data_files=parquet, split="train")

    # Keep audio as encoded bytes: decoding every clip up front would cost far
    # more memory than the sampled clips need.
    return ds.cast_column("audio", Audio(decode=False))


def _meta_columns(ds) -> List[str]:
    cols = ["id"]
    for name in ("num_samples", "duration"):
        if name in ds.column_names:
            cols.append(name)
    return cols


def main(argv=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--languages", nargs="+", default=["hi_in", "te_in"], help="FLEURS configs")
    p.add_argument("--per-language", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", default="test", choices=["test", "validation", "train"])
    p.add_argument("--max-seconds", type=float, default=20.0)
    p.add_argument("--out-dir", default="data/audio/fleurs")
    p.add_argument("--manifest", default="data/manifest_fleurs.csv")
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    audio_prefix = Path(os.path.relpath(out_dir, manifest_path.parent)).as_posix()

    records: List[List[str]] = []
    for config in args.languages:
        language = whisper_language(config)
        print(f"\n== {config} (language={language}) ==")
        ds = load_split(config, args.split)
        meta = ds.select_columns(_meta_columns(ds)).to_list()
        chosen = select_rows(meta, args.per_language, args.seed, args.max_seconds)
        if not chosen:
            print(f"  no clips <= {args.max_seconds}s found, skipping")
            continue
        if len(chosen) < args.per_language:
            print(f"  only {len(chosen)} clip(s) available (asked for {args.per_language})")

        index_by_id: Dict = {}
        for i, row in enumerate(meta):
            index_by_id.setdefault(row["id"], i)

        for n, row in enumerate(chosen, start=1):
            full = ds[index_by_id[row["id"]]]
            name = clip_filename(config, row["id"])
            dest = out_dir / name
            duration = write_wav(audio_bytes(full["audio"]), dest)
            text = reference_text(full)
            print(f"  [{n}/{len(chosen)}] {name}  {duration:.1f}s  {text[:48]}")
            records.append([f"{audio_prefix}/{name}", text, language, "fleurs"])

    if not records:
        p.error("No clips were exported")

    with open(manifest_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["audio_path", "reference", "language", "source"])
        w.writerows(records)

    print(f"\nWrote {len(records)} clips to {out_dir} and manifest {manifest_path}")
    print("Attribution: Google FLEURS, CC-BY-4.0. See the README.")


if __name__ == "__main__":
    main()
