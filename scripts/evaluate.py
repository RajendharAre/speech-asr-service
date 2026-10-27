"""Measure WER, CER, script share and real-time factor of one or more Whisper
model sizes on your own recordings, optionally with per-language initial prompts.

    python -m scripts.evaluate --manifest data/manifest_fleurs.csv --model-sizes base small
    python -m scripts.evaluate --manifest data/manifest_fleurs.csv --model-sizes base \
        --prompts-json data/prompts.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from app.backends.faster_whisper_backend import FasterWhisperTranscriber
from app.evaluation import (
    FileResult,
    file_error_rates,
    load_manifest,
    script_share,
    summarize_by_source,
    to_markdown,
)

# Same limit as the API's initial_prompt field.
MAX_PROMPT_CHARS = 200


def load_prompts(path: str) -> Dict[str, str]:
    """Read a JSON object mapping language code to prompt text."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("prompts JSON must be an object mapping language code to prompt text")

    prompts: Dict[str, str] = {}
    for language, prompt in data.items():
        code = str(language).strip().lower()
        text = str(prompt).strip()
        if not text:
            continue
        if len(text) > MAX_PROMPT_CHARS:
            raise ValueError(f"prompt for '{code}' is longer than {MAX_PROMPT_CHARS} characters")
        prompts[code] = text
    if not prompts:
        raise ValueError(f"{path} contains no prompts")
    return prompts


def transcribe_rows(
    transcriber,
    rows: List[dict],
    prompt_map: Optional[Dict[str, str]] = None,
    label: str = "",
) -> List[FileResult]:
    results: List[FileResult] = []
    for i, row in enumerate(rows, start=1):
        prompt = prompt_map.get(row["language"]) if prompt_map else None
        started = time.perf_counter()
        out = transcriber.transcribe(row["audio_path"], language=row["language"], initial_prompt=prompt)
        elapsed = time.perf_counter() - started
        results.append(
            FileResult(
                audio_path=row["audio_path"],
                language=row["language"],
                reference=row["reference"],
                hypothesis=out.text,
                duration_s=out.duration_s,
                processing_s=elapsed,
                source=row["source"],
            )
        )
        prefix = f"{label}: " if label else ""
        print(
            f"  [{i}/{len(rows)}] {prefix}{Path(row['audio_path']).name}: "
            f"{elapsed:.2f}s for {out.duration_s:.1f}s of audio"
        )
    return results


def environment_footer(device: str, compute_type: str) -> str:
    cpu = platform.processor() or platform.machine() or "unknown CPU"
    return (
        f"Hardware: {platform.platform()}; {cpu}; {os.cpu_count()} logical CPUs. "
        f"Device: {device}; compute type: {compute_type}. "
        "RTF values are wall-clock on this machine; timings from the Render free-tier "
        "deployment are not comparable to laptop timings."
    )


def main(argv=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, help="CSV with audio_path, reference, [language], [source]")
    p.add_argument("--model-sizes", nargs="+", default=["base"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--compute-type", default="int8")
    p.add_argument(
        "--prompts-json",
        default=None,
        help="JSON object mapping language code to initial prompt; each size is then run twice",
    )
    p.add_argument("--out-dir", default="results")
    args = p.parse_args(argv)

    rows = load_manifest(args.manifest)
    missing = [r["audio_path"] for r in rows if not Path(r["audio_path"]).exists()]
    if missing:
        p.error(f"{len(missing)} audio file(s) not found, e.g. {missing[0]}")

    prompts = None
    if args.prompts_json:
        try:
            prompts = load_prompts(args.prompts_json)
        except (OSError, ValueError, json.JSONDecodeError) as e:
            p.error(str(e))
        unused = sorted(set(prompts) - {r["language"] for r in rows})
        if unused:
            print(f"Warning: prompts for {unused} match no manifest rows")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: Dict[str, Dict[str, Dict]] = {}
    per_file: List[List] = []

    for size in args.model_sizes:
        print(f"\n== {size} ==")
        # One model instance per size; the prompt variant reuses it.
        transcriber = FasterWhisperTranscriber(size, args.device, args.compute_type)
        # One untimed decode so model/kernel warm-up doesn't inflate the first file's timing.
        transcriber.transcribe(rows[0]["audio_path"], language=rows[0]["language"])

        variants = [(size, None)]
        if prompts:
            variants.append((f"{size}+prompt", prompts))

        for variant, prompt_map in variants:
            results = transcribe_rows(transcriber, rows, prompt_map, label=variant)
            for source, summary in summarize_by_source(results).items():
                summaries.setdefault(source, {})[variant] = summary
            for r in results:
                wer, cer = file_error_rates(r.reference, r.hypothesis)
                share = script_share(r.hypothesis, r.language)
                per_file.append(
                    [
                        variant,
                        r.source or "unspecified",
                        r.audio_path,
                        r.language or "",
                        f"{wer:.4f}",
                        f"{cer:.4f}",
                        "" if share is None else f"{share:.4f}",
                        f"{r.duration_s:.2f}",
                        f"{r.processing_s:.3f}",
                        r.reference,
                        r.hypothesis,
                    ]
                )

    with open(out_dir / "per_file.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model",
                "source",
                "audio_path",
                "language",
                "wer",
                "cer",
                "script_share",
                "duration_s",
                "processing_s",
                "reference",
                "hypothesis",
            ]
        )
        w.writerows(per_file)

    markdown = to_markdown(summaries, environment_footer(args.device, args.compute_type))
    (out_dir / "summary.md").write_text(markdown, encoding="utf-8")
    print("\n" + markdown)
    print(f"Wrote {out_dir / 'per_file.csv'} and {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
