"""Download short speech clips with reference transcripts and build a manifest.

Source: FLEURS (`google/fleurs` on Hugging Face) via the public datasets-server
API. Only the individual clips you ask for are fetched, so this stays in the
tens of megabytes rather than the multi-gigabyte full-corpus download.

    python -m scripts.fetch_clips --languages te_in hi_in en_us --per-language 5

Writes 16 kHz mono 16-bit WAV files plus a manifest.csv that
`python -m scripts.evaluate --manifest ...` can consume directly.

Requires the ASR extras (PyAV ships with faster-whisper):
    pip install -r requirements-asr.txt
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

DATASET = "google/fleurs"
API = "https://datasets-server.huggingface.co"
SAMPLE_RATE = 16000
PAGE_SIZE = 100

# FLEURS configs are <lang>_<region>. Whisper wants the bare ISO 639-1 code.
# These are the few configs whose prefix is not directly a Whisper language code.
LANG_OVERRIDES = {
    "cmn_hans_cn": "zh",
    "cmn_hant_tw": "zh",
    "yue_hant_hk": "zh",
    "sr_latn_rs": "sr",
    "az_az": "az",
}


def _get_json(url: str, retries: int = 4) -> dict:
    """GET with retries. The datasets-server returns 500/timeout under load."""
    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "speech-asr-service/0.1"})
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Could not read {url}: {last}")


def _get_bytes(url: str, retries: int = 4) -> bytes:
    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "speech-asr-service/0.1"})
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Could not download {url}: {last}")


def list_rows(config: str, split: str) -> List[dict]:
    """All rows of one (config, split), paged 100 at a time."""
    rows: List[dict] = []
    offset = 0
    while True:
        url = (
            f"{API}/rows?dataset={DATASET}&config={config}&split={split}"
            f"&offset={offset}&length={PAGE_SIZE}"
        )
        data = _get_json(url)
        batch = data.get("rows", [])
        rows.extend(batch)
        total = data.get("num_rows_total", len(rows))
        offset += len(batch)
        if not batch or offset >= total:
            break
    return rows


def pick(rows: List[dict], count: int, min_s: float, max_s: float) -> List[dict]:
    """Clips whose duration falls in [min_s, max_s], spread evenly across the split."""
    candidates = []
    for r in rows:
        row = r.get("row", {})
        samples = row.get("num_samples")
        if not samples:
            continue
        duration = samples / SAMPLE_RATE
        if min_s <= duration <= max_s:
            candidates.append((r["row_idx"], row, duration))
    if not candidates:
        return []
    candidates.sort(key=lambda x: x[0])
    if len(candidates) <= count:
        chosen = candidates
    else:
        stride = len(candidates) / count
        chosen = [candidates[int(i * stride)] for i in range(count)]
    return [{"row_idx": i, "row": row, "duration_s": d} for i, row, d in chosen]


def write_wav(src_url: str, dest: Path) -> float:
    """Download the clip and re-encode to 16 kHz mono 16-bit PCM WAV."""
    import av

    raw = _get_bytes(src_url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(raw)

    with av.open(str(tmp)) as container:
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

    tmp.unlink(missing_ok=True)
    return total / SAMPLE_RATE


def whisper_language(config: str) -> str:
    if config in LANG_OVERRIDES:
        return LANG_OVERRIDES[config]
    code = config.split("_")[0].lower()
    if not 2 <= len(code) <= 3:
        raise ValueError(f"Cannot map FLEURS config '{config}' to a Whisper language code")
    return code


def main(argv=None) -> None:
    # Reference text is often non-Latin; the default Windows console codec cannot
    # print it and would raise UnicodeEncodeError.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--languages", nargs="+", default=["te_in", "hi_in", "en_us"],
                   help="FLEURS configs, e.g. te_in hi_in ta_in en_us")
    p.add_argument("--per-language", type=int, default=5, help="Clips to keep per language")
    p.add_argument("--split", default="validation", choices=["validation", "train", "test"])
    p.add_argument("--min-seconds", type=float, default=5.0)
    p.add_argument("--max-seconds", type=float, default=20.0)
    p.add_argument("--out-dir", default="data/audio")
    p.add_argument("--manifest", default="data/manifest.csv")
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    # `load_manifest` resolves relative audio paths against the manifest's folder,
    # so store them relative to it rather than to the current directory.
    audio_prefix = Path(os.path.relpath(out_dir, manifest_path.parent)).as_posix()

    records: List[List[str]] = []
    for config in args.languages:
        language = whisper_language(config)
        print(f"\n== {config} (language={language}) ==")
        rows = list_rows(config, args.split)
        print(f"  {len(rows)} rows in '{args.split}'")
        chosen = pick(rows, args.per_language, args.min_seconds, args.max_seconds)
        if not chosen:
            print(f"  no clips between {args.min_seconds}s and {args.max_seconds}s, skipping")
            continue

        for n, item in enumerate(chosen, start=1):
            row = item["row"]
            src = row["audio"][0]["src"]
            name = f"{config}_{item['row_idx']:05d}.wav"
            dest = out_dir / name
            duration = write_wav(src, dest)
            text = row["transcription"].strip()
            print(f"  [{n}/{len(chosen)}] {name}  {duration:.1f}s  {text[:48]}")
            records.append([f"{audio_prefix}/{name}", text, language])

    if not records:
        p.error("No clips were downloaded")

    with open(manifest_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["audio_path", "reference", "language"])
        w.writerows(records)

    print(f"\nWrote {len(records)} clips to {out_dir} and manifest {manifest_path}")
    print(f"Next: python -m scripts.evaluate --manifest {manifest_path.as_posix()} --model-sizes tiny base")


if __name__ == "__main__":
    main()
