# Speech ASR Service

A small, self-hostable speech-to-text API built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper), plus an evaluation script that measures word/character error rate and speed on your own recordings.

The goal is to understand how ASR models behave in deployment, especially on Indian languages and accents, instead of only calling a model and trusting the output.

## Features

- `POST /transcribe`: upload audio, optionally set a language code, get back text, segments, detected language, processing time and real-time factor.
- `GET /health`: liveness check (does not load the model).
- Input validation: extension allow-list, upload size cap, language-code check. Temp files are always deleted.
- A backend interface (`app/backends/base.py`), so another engine, or a text-to-speech counterpart, can be added without touching the API layer.
- Evaluation CLI: corpus-level WER and CER plus real-time factor, per model size and per language.
- Configurable CPU threading (`ASR_CPU_THREADS`) and a run-to-run reproducibility experiment (`scripts/reproducibility.py`).
- Tests use a fake backend, so CI never downloads model weights.

## Quickstart

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-asr.txt -r requirements-dev.txt
uvicorn app.main:app --reload
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-asr.txt -r requirements-dev.txt
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs for the interactive API page. The first transcription request downloads the model weights, so it needs an internet connection and is slow.

```powershell
curl.exe -F "file=@sample.wav" -F "language=hi" http://127.0.0.1:8000/transcribe
```

Response fields: `text`, `language`, `language_probability`, `duration_s`, `processing_time_s`, `real_time_factor` (processing time ÷ audio duration, lower is faster), `model`, and `segments` (`start`, `end`, `text`).

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Tests use a fake backend, so no model weights are downloaded.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `ASR_MODEL_SIZE` | `base` | `tiny`, `base`, `small`, `medium`, `large-v3`, ... |
| `ASR_DEVICE` | `cpu` | `cpu` or `cuda` |
| `ASR_COMPUTE_TYPE` | `int8` | `int8` on CPU, `float16` on GPU |
| `ASR_CPU_THREADS` | `0` | Threads per model; `0` lets ctranslate2 choose. Set `1` for single-threaded runs. |
| `ASR_MAX_UPLOAD_MB` | `25` | Upload size limit |

## Docker

```bash
docker build -t speech-asr-service .
docker run -p 8000:8000 -v hf_cache:/models speech-asr-service
```

The volume keeps downloaded weights between runs.

## Evaluation methodology

### Data

`scripts/fetch_clips.py` downloads the evaluation clips from [FLEURS](https://huggingface.co/datasets/google/fleurs) (CC-BY-4.0) and writes `data/manifest.csv`. The run below uses 12 clips: 4 `en_us`, 4 `hi_in` and 4 `te_in`, each 6–18 seconds of read speech.

```powershell
.\.venv\Scripts\python.exe -m scripts.fetch_clips
```

To score your own audio instead: record 15 to 30 clips (5 to 20 seconds) with different speakers and conditions, write the exact reference transcript for each, and fill in `data/manifest.csv`. Columns are `audio_path`, `reference`, `language` (optional); audio paths are relative to the manifest.

### Running

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate --manifest data/manifest.csv --model-sizes tiny base small medium
```

This writes `results/per_file.csv` and `results/summary.md`.

### Method

- WER and CER are corpus-level (total errors ÷ total reference words or characters), not an average of per-file scores.
- Both sides are normalised before scoring: Unicode NFC, lowercase, punctuation removed. Combining marks are kept, which matters for Indic scripts. CER counts Unicode code points.
- A reference that is empty after normalisation is rejected rather than scored as a bogus rate.
- Each model does one untimed warm-up decode before timing starts.
- If a row has a `language`, that language is forced; otherwise the model auto-detects.
- RTF is processing time ÷ audio duration, measured on CPU. It is wall-clock, so it depends on machine load and is not comparable across machines.
- The tables below use faster-whisper's default decoding (beam size 5 with temperature fallback). That fallback can sample, so Hindi/Telugu transcripts — and therefore their WER/CER — are not identical from run to run; see [Reproducibility](#reproducibility).

Recorded environment: Python 3.14.7, faster-whisper 1.2.1, ctranslate2 4.8.3, PyAV 18.1.0, jiwer 4.0.0, Windows 11, CPU only (`int8`).

## Reproducibility

Identical audio can produce different text across runs. `scripts/reproducibility.py` isolates the cause by re-running the same clips under controlled settings (thread count × decoding):

```powershell
.\.venv\Scripts\python.exe -m scripts.reproducibility --languages te hi en --runs 5 --out-dir results
```

It writes `results/reproducibility.json` with the transcripts for every configuration, whether they were identical, and the mean processing time.

### `ASR_CPU_THREADS`

`ASR_CPU_THREADS` is passed to faster-whisper as `cpu_threads` (ctranslate2's `intra_threads`). The default is `0`, which lets ctranslate2 pick the thread count.

```powershell
$env:ASR_CPU_THREADS = "1"          # single-threaded decode
uvicorn app.main:app
```

Docker: `docker run -e ASR_CPU_THREADS=1 -p 8000:8000 speech-asr-service`.

### What we found

Nondeterminism is **not** caused by thread count. It comes from faster-whisper's temperature fallback in `generate_with_fallback` (`faster_whisper/transcribe.py`): when a decode looks low-confidence (compression ratio > 2.4 or average log-probability < -1.0), it retries at temperatures 0.2, 0.4, … 1.0 using **sampling**, which is random.

`base` model, 5 runs per configuration:

| Clip | Threads | Decoding | Identical? | Unique outputs | Mean time |
|---|---|---|---|---|---|
| te | auto | default | no | 5 | 42.6 s |
| te | single | default | no | 5 | 54.0 s |
| te | auto | `temperature=0` | yes | 1 | 7.2 s |
| te | single | `temperature=0` | yes | 1 | 12.2 s |
| hi | auto / single | either | yes | 1 | 3.0–7.3 s |
| en | auto / single | either | yes | 1 | 1.6–3.3 s |

Telugu is low-confidence, so it falls back to sampling at both thread counts (5 unique transcripts each). Disabling the fallback (`--decoding no-fallback`, i.e. `temperature=0`) makes it byte-identical. Hindi and English did not trigger the fallback and were identical even with default decoding. Pre-seeding `ctranslate2` (`set_random_seed`) did not fix it on its own.

### Interpreting single-thread runs

- `ASR_CPU_THREADS=1` does **not** make output deterministic; it only changes ctranslate2's intra-op parallelism (about 1.3–1.7× slower on this machine).
- For byte-reproducible output, disable the temperature fallback (`temperature=0`). That trades away the robustness the fallback was designed to add, so decide per use case.
- With the default `ASR_CPU_THREADS=0`, ctranslate2 may use all cores; set it explicitly to trade latency against competing processes.

## Results

Newly measured on the machine described above, 12 FLEURS clips, default decoding, CPU `int8`, single run.

| Model | Files | WER | CER | RTF |
|---|---|---|---|---|
| tiny | 12 | 76.2% | 72.3% | 0.49 |
| base | 12 | 78.2% | 70.4% | 0.87 |
| small | 12 | 54.4% | 37.1% | 10.01 |
| medium | 12 | 46.0% | 29.9% | 7.80 |

WER / CER by language:

| Model | en | hi | te |
|---|---|---|---|
| tiny | 4.2% / 1.0% | 104.7% / 100.5% | 100.0% / 95.9% |
| base | 2.8% / 0.2% | 100.0% / 93.8% | 119.4% / 99.6% |
| small | 5.6% / 1.2% | 54.7% / 18.8% | 109.7% / 95.0% |
| medium | 0.0% / 0.0% | 43.8% / 12.2% | 103.2% / 81.7% |

Caveats:

- WER above 100% means the hypothesis has more insertions than the reference has words.
- Hindi and Telugu come from a single run of a nondeterministic decoder, so treat those numbers as indicative, not exact. Use `temperature=0` (see [Reproducibility](#reproducibility)) to reproduce them exactly.
- RTF for small/medium is dominated by the Telugu fallback loop and was sampled while the CPU was also busy, so it is not a clean throughput benchmark.
- Whisper sometimes emits Hindi in Urdu/Arabic script and Telugu in Latin script. Those count as errors even when the acoustics are partly right; that is a model limitation, not a scoring bug.
- Not measured: GPU (`cuda`/`float16`), `large-v3`, and per-speaker or per-noise breakdowns.

## Design notes

- The model is loaded lazily, once per process.
- Uploads are streamed to a temp file in 1 MB chunks, size-capped, and removed in a `finally` block. Tests cover the success, failure and oversize paths.
- Model errors are logged on the server and returned to the client as a generic 500, so internals are not leaked.

## Project layout

```
app/
  main.py                      FastAPI routes
  config.py                    Settings from environment variables
  evaluation.py                Normalisation, WER/CER, RTF, markdown summary
  backends/
    base.py                    Transcriber interface
    faster_whisper_backend.py  faster-whisper implementation
scripts/
  evaluate.py                  Evaluation CLI
  fetch_clips.py               FLEURS clip downloader + manifest builder
  reproducibility.py           Reproducibility experiment CLI
tests/                         API, scoring, CLI and configuration tests
```

## Limitations

- Defaults target CPU. The GPU path is configurable (`ASR_DEVICE=cuda`) but **not verified** on this machine (ctranslate2 had no CUDA devices). `ASR_CPU_THREADS` affects CPU-side parallelism and still applies via ctranslate2 even on GPU, but its effect was not measured here.
- No authentication, rate limiting or streaming. One process holds one model in memory.
- Whisper accuracy varies by language and model size, and lower-resource languages can be noticeably worse. Measure with the evaluation script instead of assuming.
- With default decoding, Telugu (and some low-confidence cases) can be nondeterministic because of faster-whisper's temperature fallback; see [Reproducibility](#reproducibility).
- Results above use 12 FLEURS clips only (4/4/4 for en/hi/te). They are not a comprehensive benchmark.
- Corrupt audio currently returns a 500 instead of a 4xx.

## Roadmap (not implemented)

- Chunked or streaming transcription
- Compare against Indic fine-tuned checkpoints
- A text-to-speech backend following the same interface pattern
- Authentication and rate limiting
