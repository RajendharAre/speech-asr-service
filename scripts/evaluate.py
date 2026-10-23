"""Measure WER, CER and real-time factor of one or more Whisper model sizes
on your own recordings.

    python -m scripts.evaluate --manifest data/manifest.csv --model-sizes tiny base small
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from app.backends.faster_whisper_backend import FasterWhisperTranscriber
from app.evaluation import FileResult, file_error_rates, load_manifest, summarize, to_markdown


def run_model(size: str, rows: list, device: str, compute_type: str) -> list:
    print(f"\n== {size} ==")
    transcriber = FasterWhisperTranscriber(size, device, compute_type)
    # One untimed decode so model/kernel warm-up doesn't inflate the first file's timing.
    transcriber.transcribe(rows[0]["audio_path"], language=rows[0]["language"])

    results = []
    for i, row in enumerate(rows, start=1):
        start = time.perf_counter()
        out = transcriber.transcribe(row["audio_path"], language=row["language"])
        elapsed = time.perf_counter() - start
        results.append(
            FileResult(
                audio_path=row["audio_path"],
                language=row["language"],
                reference=row["reference"],
                hypothesis=out.text,
                duration_s=out.duration_s,
                processing_s=elapsed,
            )
        )
        print(f"  [{i}/{len(rows)}] {Path(row['audio_path']).name}: {elapsed:.2f}s for {out.duration_s:.1f}s of audio")
    return results


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, help="CSV with audio_path, reference, [language]")
    p.add_argument("--model-sizes", nargs="+", default=["base"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--compute-type", default="int8")
    p.add_argument("--out-dir", default="results")
    args = p.parse_args(argv)

    rows = load_manifest(args.manifest)
    missing = [r["audio_path"] for r in rows if not Path(r["audio_path"]).exists()]
    if missing:
        p.error(f"{len(missing)} audio file(s) not found, e.g. {missing[0]}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries, per_file = {}, []
    for size in args.model_sizes:
        results = run_model(size, rows, args.device, args.compute_type)
        summaries[size] = summarize(results)
        for r in results:
            wer, cer = file_error_rates(r.reference, r.hypothesis)
            per_file.append(
                [size, r.audio_path, r.language or "", f"{wer:.4f}", f"{cer:.4f}",
                 f"{r.duration_s:.2f}", f"{r.processing_s:.3f}", r.reference, r.hypothesis]
            )

    with open(out_dir / "per_file.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["model", "audio_path", "language", "wer", "cer", "duration_s", "processing_s", "reference", "hypothesis"])
        w.writerows(per_file)

    footer = f"Device: {args.device} ({args.compute_type}). Files: {len(rows)}. Lower is better for all columns."
    markdown = to_markdown(summaries, footer)
    (out_dir / "summary.md").write_text(markdown, encoding="utf-8")
    print("\n" + markdown)
    print(f"Wrote {out_dir / 'per_file.csv'} and {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
