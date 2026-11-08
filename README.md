# Audiobookey

Turn PDFs into audiobooks, entirely on your own machine. No cloud APIs, no
accounts: a local pipeline that extracts the text, synthesizes it with a
neural TTS engine, and packages chaptered `m4b` plus `mp3` output,
loudness-normalized. Two synthesis engines, both GPU-accelerated when a CUDA
device is present: **Kokoro-82M** for fast drafts and **Chatterbox V3** for
final, clone-grade audio.

## Features

- **PDF to m4b + mp3** — one chapter per PDF page, loudness-normalized output
- **Two engines** — Kokoro-82M ONNX (draft) and Chatterbox (final), selectable per run via quality tiers
- **Quality tiers** — `low` / `medium` / `high` (see table below)
- **Web reader** — browser UI with live conversion progress, per-page text, and a PDF pane rendered next to the audio
- **Voice cloning** — add a named voice from a 5–15 s WAV sample and assign it anywhere
- **Diarization** — quotes in the text get a separate dialogue voice (`--diarize`)
- **OCR** — scanned, text-less PDF pages via Tesseract (optional extra)
- **Batch conversion** — queue several PDFs into one run through the CLI or the server
- **Job controls** — pause, cancel, and requeue (resume) running conversions; paused books resume from the chunk cache, so already-synthesized audio is never re-rendered. Cancel and delete reclaim everything from disk immediately (partial audio, chunk cache, uploaded PDF)
- **Durable chunk cache** — chunk WAVs are keyed by text + voice config + engine identity; re-runs with unchanged inputs never synthesize twice

## Install

Requires Python 3.13 (the project targets `>=3.13,<3.14`) and
[uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

The base install includes both engines and the CUDA 12 GPU stack on Windows.
For scanned PDFs, add the OCR extra:

```bash
uv sync --extra ocr
```

OCR uses the `pytesseract` library only — the Tesseract binary itself is a
system install and is **not** provided by the extra. Check `tesseract
--version` before relying on `--ocr`.

## Model setup

Models live in a `models/` directory, resolved in this order: the
`--models-dir` flag, the `AUDIOBOOKEY_MODELS_DIR` environment variable, the
current working directory, then upward from the package location. A valid
models directory contains a `voices/` subdirectory.

- **Kokoro (draft)**: the ONNX files `models/model_fp16.onnx` and
  `models/model_q4.onnx`, plus voice style bins under `models/voices/`. These
  are fetched separately and placed there; the engine loads them directly.
- **Chatterbox (final)**: checkpoints under `models/chatterbox/`
  (`ve.safetensors`, `t3_cfg.safetensors`, `s3gen.safetensors`,
  `tokenizer.json`, `conds.pt`). These are **auto-downloaded** on first use
  from Hugging Face (`ResembleAI/chatterbox`) via `huggingface_hub`,
  resumably — an interrupted download continues where it stopped.

```
models/
  model_fp16.onnx  model_q4.onnx     # Kokoro (draft)
  voices/*.bin                      # Kokoro voice styles
  chatterbox/                       # Chatterbox checkpoints (auto-downloaded)
```

## Quickstart

Convert a PDF from the command line:

```bash
# default: medium quality (Chatterbox final), outputs to ./audiobook
uv run audiobookey convert book.pdf

# fastest draft voice, custom output dir
uv run audiobookey convert book.pdf --quality low --out my-book

# several PDFs in one sequential run
uv run audiobookey convert vol1.pdf vol2.pdf --out series
```

Useful `convert` flags (all verified against `cli.py`):

| Flag | Meaning |
|---|---|
| `--quality low\|medium\|high` | Synthesis tier (see table below; `draft`/`final` also accepted) |
| `--out DIR` | Output directory (default `audiobook`) |
| `--voice NAME` | Kokoro voice id, or a name from `voices.json` (default `af_heart`) |
| `--speed X` | Speech speed (draft engine only; final rejects non-1.0) |
| `--diarize` | Give quoted spans a separate dialogue voice |
| `--ocr` / `--ocr-lang LL` | OCR text-less pages / Tesseract language (default `eng`) |
| `--lexicon FILE` | `pattern => replacement` lines, `r:` prefix for regex |
| `--voice-map FILE` | JSON `{regex: voice-name}`; first match wins |
| `--gap SEC` | Sentence gap, draft engine (default `0.3`) |
| `--cache-root DIR` | Shared chunk cache: `<cache-root>/<book-id>/chunks` |
| `--lang LL` | Phonemization language (default `en-us`) |
| `--models-dir DIR` | Models directory override |
| `--reporter rich\|lines` | `lines` emits machine-readable progress events |

Serve the web reader:

```bash
uv run audiobookey serve            # http://127.0.0.1:8765
uv run audiobookey serve --port 9000
```

The server exposes the API under `/api/*` (books, pages, audio, covers,
bookmarks, progress, voices, previews, single and batch conversion, job
status, pause/cancel) and serves the frontend from `web/` at `/`.

## GPU notes

On Windows the GPU stack is pinned deliberately: `onnxruntime-gpu==1.22.0`
(because 1.23+ needs CUDA 13, which has no complete Windows wheels yet), and
cuBLAS 12 — which has no Windows wheel on PyPI at all — is dropped via a uv
override, with `engine.py` registering DLL directories from the pip
`nvidia-*` wheels, torch's bundled DLLs, CUDA toolkit installs, or
`AUDIOBOOKEY_CUDA_DLL_DIR` instead. At load time each engine runs a canary
synthesis and falls back through providers; the fp16 Kokoro graph can load
on CUDA but silently produce zero samples on current onnxruntime builds,
which is why the engine prefers the **q4** quantized model on CUDA. See the
comments in `pyproject.toml` — they are the source of truth.

## Voices and cloning

Named voices live in `data/voices.json` (schema:
`data/voices.schema.json`; `data/` is runtime data and git-ignored):

```bash
# draft voice: a Kokoro voice id
uv run audiobookey voices add narrator --voice-id af_heart

# cloned voice: from a 5-15 s WAV sample (Chatterbox)
uv run audiobookey voices add me --clone data/clone-sample.wav --exaggeration 0.5

uv run audiobookey voices list
uv run audiobookey voices preview me "Hello from my cloned voice." --out me.wav
```

`--voice` on `convert` accepts any name from `voices.json`, and a raw Kokoro
voice id for the draft engine. Books can mix engines — each voice carries its
own. Inline markup switches voice per region:

```
[voice:me]Only this sentence uses the cloned voice.[/voice] Narration resumes.
```

## Quality tiers

The user-facing tiers map onto the two engines; `medium` and `high` share
identical chunk synthesis and cache keys — only the final encode differs.

| Tier | Engine | Output encode | Speed control | Notes |
|---|---|---|---|---|
| `low` | Kokoro-82M (kokoro-onnx, q4/fp16 ONNX) | standard | `--speed`, `--gap` | fastest; stable on CUDA |
| `medium` | Chatterbox (PyTorch CUDA) | standard | not supported | clone-grade voice |
| `high` | Chatterbox (PyTorch CUDA) | best bitrate | not supported | same synthesis as medium, better encode |

Engines are validated at load with a real canary synthesis (draft:
fp16+CUDA → q4+CUDA → fp16+CPU; final: CUDA → CPU), because a graph that
loads without error can still produce silence. Both engines synthesize at
24 kHz; the pipeline resamples uniformly before concatenation, so draft and
final outputs are interchangeable.

## Project layout

```
src/audiobookey/        CLI, engines, pipeline, FastAPI server
  cli.py                argument parsers, entrypoint
  engine*.py            Kokoro ONNX + Chatterbox engines, load canaries
  audiobook.py          convert pipeline (chunk, cache, encode)
  server/               FastAPI app, job manager, db, previews
web/                    browser reader (served by `audiobookey serve`)
tests/                  pytest suite (fake engines; TTS smoke test when q4 model present)
data/                   runtime data: sqlite db, voices.json, clones, uploads, outputs
models/                 TTS weights (fetched/auto-downloaded, git-ignored)
```

```bash
uv run pytest    # run the test suite
```

## License

MIT — see [LICENSE](LICENSE).
