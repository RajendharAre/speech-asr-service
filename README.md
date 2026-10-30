# Speech ASR Service

A small, self-hostable speech-to-text API built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper), plus an evaluation harness that measures word/character error rate, script fidelity and speed on your own recordings.

The goal is to understand how ASR models behave in deployment, especially on Indian languages and accents, instead of only calling a model and trusting the output. Whisper can return, for example, Hindi words written in Roman letters; the harness below measures how often that happens instead of assuming it.

## Features

- `POST /transcribe`: upload audio, optionally set a language code and an `initial_prompt`, get back text, segments, detected language, processing time and real-time factor.
- `GET /health`: liveness check (does not load the model).
- Input validation: extension allow-list, upload size cap, language-code check, `initial_prompt` trimmed and capped at 200 characters. Temp files are always deleted.
- A backend interface (`app/backends/base.py`), so another engine, or a text-to-speech counterpart, can be added without touching the API layer.
- Evaluation CLI: corpus-level WER and CER, script share, real-time factor, per source, with optional per-language initial prompts run as A/B variants.
- Dataset preparation: a small, reproducible FLEURS test-split sample with references.
- Configurable CPU threading (`ASR_CPU_THREADS`) and a run-to-run reproducibility experiment (`scripts/reproducibility.py`).
- Tests use a fake backend, so CI never downloads model weights.

## Quickstart (Windows, PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-asr.txt -r requirements-dev.txt
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs for the interactive API page. The first transcription request downloads the model weights, so it needs an internet connection and is slow.

```powershell
curl.exe -F "file=@sample.wav" -F "language=hi" http://127.0.0.1:8000/transcribe
```

`/transcribe` accepts three form fields:

| Field | Required | Meaning |
|---|---|---|
| `file` | yes | Audio file (`.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, `.webm`) |
| `language` | no | 2–3 letter code such as `hi`, `te`, `en`. Omitted means auto-detect |
| `initial_prompt` | no | Short hint, at most 200 characters. Trimmed; blank means "not set" |

`initial_prompt` is passed to faster-whisper unchanged. Setting it is optional and omitting it behaves exactly as before. For non-Latin scripts, prefer sending it from code or from `data/prompts.json` rather than typing it into `cmd`, which may mangle the encoding.

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
| `ASR_TEMPERATURE` | unset | Unset keeps faster-whisper's temperature fallback; `0` disables it for deterministic beam search. |
| `ASR_MAX_UPLOAD_MB` | `25` | Upload size limit |

## Docker

```powershell
docker build -t speech-asr-service .
docker run -p 8000:8000 -v hf_cache:/models speech-asr-service
```

The volume keeps downloaded weights between runs.

## Evaluation methodology

### Sources

Results are always reported per source. FLEURS clips and your own recordings differ in speakers, microphones and difficulty, so the summary never combines them into one WER.

**FLEURS test clips.** `scripts/prepare_dataset.py` exports a deterministic sample of the FLEURS test split:

```powershell
pip install -r requirements-data.txt
.\.venv\Scripts\python.exe -m scripts.prepare_dataset --per-language 8 --seed 0
```

It keeps clips of 20 seconds or less, writes 16 kHz mono 16-bit WAV files to `data/audio/fleurs/` (named like `hi_in_00179.wav`) and writes `data/manifest_fleurs.csv` with `source=fleurs`. Selection is stable for a given dataset revision: duplicate ids are dropped, rows are sorted by id, then `random.Random(seed)` draws the sample. The dataset's `raw_transcription` field is used as the reference (it keeps punctuation), falling back to `transcription`.

The first run downloads the whole test split per language, which is a few hundred megabytes each; later runs reuse the Hugging Face cache. `requirements-data.txt` is separate from the API and CI dependencies.

The checked-in `data/manifest.csv` is an older 12-clip FLEURS sample with `source=fleurs`.

**Your own recordings.** See [data/README.md](data/README.md): 6 clips per language, 5–20 seconds, reading a written sentence aloud so the written text is the reference, saved under `data/audio/own/` with `source=own`. Never generate the reference by running an ASR model on your recording; that is circular. Read speech is easier than spontaneous speech, so own-recording scores are optimistic relative to real usage.

### Initial prompts

Copy the example and edit it in UTF-8 (Notepad is fine):

```powershell
Copy-Item data\prompts.example.json data\prompts.json
```

```json
{
  "hi": "यह एक हिंदी वाक्य है।",
  "te": "ఇది ఒక తెలుగు వాక్యం।"
}
```

With `--prompts-json`, each model size is evaluated twice: as `<size>` with no prompt and as `<size>+prompt` with the prompt for the row's language. Each model is loaded once and reused for both variants. Rows whose language has no prompt run without one.

### Running

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate --manifest data/manifest.csv --model-sizes base small --prompts-json data/prompts.json
```

`medium` is slow on CPU and each low-confidence clip can trigger several decoding retries, so start with `base` and `small`. To evaluate your own recordings:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate --manifest data/manifest_own.csv --model-sizes base small --prompts-json data/prompts.json
```

Outputs:

- `results/per_file.csv`: `model`, `source`, `audio_path`, `language`, `wer`, `cer`, `script_share`, `duration_s`, `processing_s`, `reference`, `hypothesis`.
- `results/summary.md`: one section per source, each with `Variant | Files | WER | CER | Script% | RTF`, WER and Script% by language, and warnings. The footer records the hardware, device and compute type, and states that Render free-tier timings are not comparable to laptop timings.

### Method

- WER and CER are corpus-level (total errors ÷ total reference words or characters), not an average of per-file scores.
- Both sides are normalised before scoring: Unicode NFC, lowercase, punctuation removed. Combining marks are kept, which matters for Indic scripts. CER counts Unicode code points.
- **Script%** is the fraction of letters and vowel-sign marks in the hypothesis that use the language's expected script: Devanagari for `hi`, Telugu for `te`. Digits, punctuation, whitespace and zero-width characters are ignored. It is `n/a` for languages with no single expected script (such as `en`) and for empty output. It answers "did the model write the right language in the right writing system?", separately from "were the words right?".
- A language row with Script% below 50% and WER above 80% produces a warning in `summary.md`: WER is dominated by script mismatch, not recognition quality.
- A reference that is empty after normalisation is rejected rather than scored as a bogus rate.
- Each model does one untimed warm-up decode before timing starts.
- If a row has a `language`, that language is forced; otherwise the model auto-detects.
- RTF is processing time ÷ audio duration. It is wall-clock, so it depends on machine load and is not comparable across machines.
- Decoding uses faster-whisper's defaults (beam size 5 with temperature fallback). That fallback can sample, so Hindi/Telugu transcripts — and therefore their metrics — are not identical from run to run; see [Reproducibility](#reproducibility).

## Reproducibility

Identical audio can produce different text across runs. `scripts/reproducibility.py` isolates the cause by re-running the same clips under controlled settings (thread count × decoding):

```powershell
.\.venv\Scripts\python.exe -m scripts.reproducibility --manifest data/manifest.csv --languages te hi en --runs 5 --out-dir results
```

It writes `results/reproducibility.json` with the transcripts for every configuration, whether they were identical, and the mean processing time. The JSON is the source of truth; do not restate its numbers in prose.

Recorded conclusion: nondeterminism is **not** caused by thread count. It comes from faster-whisper's temperature fallback in `generate_with_fallback` (`faster_whisper/transcribe.py`): when a decode looks low-confidence (compression ratio > 2.4 or average log-probability < -1.0), it retries at temperatures 0.2, 0.4, … 1.0 using **sampling**, which is random. On Telugu, the fallback triggered at both thread counts and produced a different transcript on every run; disabling it (`--decoding no-fallback`, i.e. `temperature=0`) made the output byte-identical. Hindi and English did not trigger the fallback in that experiment. Pre-seeding `ctranslate2` (`set_random_seed`) did not fix it on its own.

`ASR_CPU_THREADS=1` does **not** make output deterministic; it only changes ctranslate2's intra-op parallelism and makes decoding slower. For byte-reproducible output, disable the temperature fallback (`temperature=0`). That trades away the robustness the fallback was designed to add, so decide per use case.

## Results

First evaluation: 12 clips from Google FLEURS (4 English, 4 Hindi, 4 Telugu), one run per variant, CPU only (int8).

[paste the tables from results/summary.md here]

What this shows:
- Without a prompt, base put 0.0% of its Hindi and Telugu letters in the expected script. A native-script prompt raised that to 100% (Hindi) and 76.1% (Telugu) on base.
- Script and recognition are separate problems. Telugu WER stayed at or above 100% even when small+prompt had 100% correct script, while Hindi WER fell from 91.4% (base+prompt) to 53.1% (small+prompt), which points to model size.

Limits:
- 4 clips per language, one run, no confidence intervals.
- FLEURS is read speech and may overlap Whisper's training data. Own recordings are planned as a second test.
- Hindi and Telugu WER is sensitive to spelling variants and normalization.
- RTF was timed once on one laptop. Prompting raised base's RTF from 0.51 to 1.74; I have not investigated why. Render free-tier timings are not comparable.

## Design notes

- The model is loaded lazily, once per process.
- `initial_prompt` is trimmed at the API boundary; blank becomes `None`, and anything over 200 characters is rejected with 400 before transcription starts.
- Uploads are streamed to a temp file in 1 MB chunks, size-capped, and removed in a `finally` block. Tests cover the success, failure and oversize paths.
- Model errors are logged on the server and returned to the client as a generic 500, so internals are not leaked.
- The evaluation harness keeps the model and the prompt variant in one process, so a prompt A/B run does not reload weights.

## Project layout

```
app/
  main.py                      FastAPI routes
  config.py                    Settings from environment variables
  evaluation.py                Normalisation, WER/CER, script share, RTF, markdown summary
  backends/
    base.py                    Transcriber interface
    faster_whisper_backend.py  faster-whisper implementation
scripts/
  prepare_dataset.py           Reproducible FLEURS test-split export
  evaluate.py                  Evaluation CLI, prompt variants, per-source summary
  fetch_clips.py               Earlier FLEURS downloader + manifest builder
  reproducibility.py           Reproducibility experiment CLI
data/
  README.md                    How to record and reference your own clips
  prompts.example.json         Neutral prompt per language
  manifest.csv                 Checked-in 12-clip FLEURS sample
tests/                         API, scoring, CLI, dataset-prep and configuration tests

```



## Limitations

- Defaults target CPU. The GPU path is configurable (`ASR_DEVICE=cuda`) but **not verified** on this machine (ctranslate2 had no CUDA devices).
- No authentication, rate limiting or streaming. One process holds one model in memory.
- Whisper accuracy varies by language and model size, and lower-resource languages can be noticeably worse. Measure with the evaluation script instead of assuming.
- Script% is a heuristic: it detects the writing system, not whether the words are correct. A high Script% with high WER means the model wrote the right script and the wrong words.
- With default decoding, low-confidence cases can be nondeterministic because of faster-whisper's temperature fallback; see [Reproducibility](#reproducibility).
- FLEURS is read speech from a small number of speakers per language; it is not a comprehensive benchmark, and it is easier than spontaneous conversation.
- The `datasets` loader API changes between releases; `prepare_dataset.py` falls back to the repo's parquet files, but a future release may still need a version pin.
- Corrupt audio returns 400 rather than 500. Truly unexpected model failures still return 500.

## Roadmap (not implemented)

- Chunked or streaming transcription
- Compare against Indic fine-tuned checkpoints
- A text-to-speech backend following the same interface pattern
- Authentication and rate limiting

## Data attribution

The evaluation clips come from **FLEURS** (Few-shot Learning Evaluation of Universal Representations of Speech), Google, described in *FLEURS: Few-Shot Cross-Lingual Universal Speech Recognition* (Conneau et al., 2022, arXiv:2205.12446) and published as [`google/fleurs`](https://huggingface.co/datasets/google/fleurs) under **CC-BY-4.0** (license tag on the dataset card). Clip transcripts are © Google and the FLEURS contributors.
