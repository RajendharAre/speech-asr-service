"""Repeatability experiment: transcribe the same clips several times and report
whether the transcripts are byte-identical.

    python -m scripts.reproducibility --manifest data/manifest.csv --model-size small --runs 5

Everything except the variable under test is held fixed: the audio file, the
language, the model, the compute type, the decoding parameters and the
software versions. The script varies three axes:

  decoding   `default`      faster-whisper defaults, i.e. beam_size=5 plus the
                             temperature fallback list [0.0, 0.2, ... 1.0]
               `no-fallback` temperature=0.0, so only beam search is used and
                             the stochastic sampling path is never reached
  threads    `auto`         cpu_threads=0 (CTranslate2 picks the count)
               `single`      cpu_threads=1
  language   `forced`       the language column of the manifest is passed in
               `detect`      language=None, so the model detects it

This script drives faster-whisper directly instead of going through
app.backends, because the production backend deliberately exposes only the
language. The `default` decoding configuration reproduces the production
backend's call exactly (same beam size, same temperature list, same
vad_filter=True).
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from app.evaluation import load_manifest

# faster-whisper's own default list. Passing it explicitly documents the
# experiment and keeps it stable across faster-whisper upgrades.
DEFAULT_TEMPERATURES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

DECODING_CONFIGS: Dict[str, Dict] = {
    "default": {
        "beam_size": 5,
        "temperature": DEFAULT_TEMPERATURES,
        "description": "faster-whisper defaults; temperature fallback enabled",
    },
    "no-fallback": {
        "beam_size": 5,
        "temperature": 0.0,
        "description": "temperature=0.0; beam search only, no sampling",
    },
}

THREAD_CONFIGS: Dict[str, int] = {"auto": 0, "single": 1}


def software_versions() -> Dict[str, str]:
    import av
    import ctranslate2
    import faster_whisper

    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "faster_whisper": faster_whisper.__version__,
        "ctranslate2": ctranslate2.__version__,
        "av": av.__version__,
    }
    try:
        import importlib.metadata as md

        versions["jiwer"] = md.version("jiwer")
    except Exception:
        versions["jiwer"] = "unknown"
    return versions


def select_clips(rows: List[Dict], languages: List[str], per_language: int) -> List[Dict]:
    chosen: List[Dict] = []
    for language in languages:
        matching = [r for r in rows if (r["language"] or "").lower() == language.lower()]
        chosen.extend(matching[:per_language])
    return chosen


def run_once(model, audio_path: str, language: Optional[str], decoding: Dict) -> str:
    segments_iter, _ = model.transcribe(
        audio_path,
        language=language,
        beam_size=decoding["beam_size"],
        temperature=decoding["temperature"],
        vad_filter=True,
    )
    return " ".join(s.text for s in segments_iter).strip()


def main(argv=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, help="CSV with audio_path, reference, [language]")
    p.add_argument("--model-size", default="small")
    p.add_argument("--device", default="cpu")
    p.add_argument("--compute-type", default="int8")
    p.add_argument("--runs", type=int, default=5, help="Transcriptions per clip and configuration")
    p.add_argument("--languages", nargs="+", default=["te", "hi", "en"])
    p.add_argument("--per-language", type=int, default=1, help="Clips per language")
    p.add_argument("--decodings", nargs="+", default=list(DECODING_CONFIGS), choices=list(DECODING_CONFIGS))
    p.add_argument("--threads", nargs="+", default=list(THREAD_CONFIGS), choices=list(THREAD_CONFIGS))
    p.add_argument("--language-modes", nargs="+", default=["forced"], choices=["forced", "detect"])
    p.add_argument("--out-dir", default="results")
    args = p.parse_args(argv)

    from faster_whisper import WhisperModel

    rows = load_manifest(args.manifest)
    clips = select_clips(rows, args.languages, args.per_language)
    if not clips:
        p.error(f"No clips matched languages {args.languages}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    versions = software_versions()
    print("Software versions")
    for key, value in versions.items():
        print(f"  {key:16s} {value}")
    print(f"\nModel {args.model_size} on {args.device} ({args.compute_type}), {args.runs} runs per configuration")
    print(f"Clips: {len(clips)}\n")

    models: Dict[int, object] = {}
    report: List[Dict] = []

    for thread_name in args.threads:
        cpu_threads = THREAD_CONFIGS[thread_name]
        if cpu_threads not in models:
            models[cpu_threads] = WhisperModel(
                args.model_size,
                device=args.device,
                compute_type=args.compute_type,
                cpu_threads=cpu_threads,
            )
        model = models[cpu_threads]
        # One untimed warm-up decode per model instance, so the first measured
        # run does not pay for lazy initialisation of the model or the kernels.
        run_once(model, clips[0]["audio_path"], clips[0]["language"], DECODING_CONFIGS[args.decodings[0]])

        for decoding_name in args.decodings:
            decoding = DECODING_CONFIGS[decoding_name]

            for language_mode in args.language_modes:
                for clip in clips:
                    audio_path = clip["audio_path"]
                    language = None if language_mode == "detect" else clip["language"]

                    transcripts: List[str] = []
                    timings: List[float] = []
                    for _ in range(args.runs):
                        started = time.perf_counter()
                        transcripts.append(run_once(model, audio_path, language, decoding))
                        timings.append(time.perf_counter() - started)

                    unique = sorted(set(transcripts))
                    identical = len(unique) == 1
                    record = {
                        "clip": Path(audio_path).name,
                        "language": clip["language"],
                        "language_mode": language_mode,
                        "threads": thread_name,
                        "cpu_threads": cpu_threads,
                        "decoding": decoding_name,
                        "runs": args.runs,
                        "identical": identical,
                        "unique_outputs": len(unique),
                        "mean_processing_s": round(sum(timings) / len(timings), 3),
                        "transcripts": transcripts,
                    }
                    report.append(record)

                    status = "IDENTICAL" if identical else f"{len(unique)} UNIQUE"
                    print(
                        f"  {record['clip']:22s} lang={record['language']:3s} "
                        f"mode={language_mode:6s} threads={thread_name:6s} decoding={decoding_name:11s} "
                        f"-> {status:12s} ({record['mean_processing_s']:.2f}s/run)"
                    )
                    if not identical:
                        for text in unique:
                            print(f"        {text[:88]!r}")

    summary = {
        "model_size": args.model_size,
        "device": args.device,
        "compute_type": args.compute_type,
        "runs_per_configuration": args.runs,
        "versions": versions,
        "decoding_configs": {k: v["description"] for k, v in DECODING_CONFIGS.items()},
        "results": report,
    }
    report_path = out_dir / "reproducibility.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    identical_count = sum(1 for r in report if r["identical"])
    print(f"\n{identical_count}/{len(report)} configurations produced byte-identical transcripts")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
