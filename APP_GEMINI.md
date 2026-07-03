# app_gemini.py — Architecture & Operations Guide

`app_gemini.py` is a Discord bot that translates message text — including text
inside images (via OCR) and linked tweets — by reacting to a message with a
flag emoji. It talks to Google's Gemini API directly (as opposed to `app.py`,
which routes through OpenRouter).

## 1. Overview

| | |
|---|---|
| Entry point | `app_gemini.py` |
| OCR module | `ocr.py` |
| Trigger | React to any message with a flag emoji |
| Translation engine | Google Gemini (`gemini-3.1-flash-lite`) |
| OCR engine | PaddleOCR (CPU, 3 pipelines) |
| Config | `.env` (`DISCORD_TOKEN`, `GEMINI_API_KEY`, `DEBUG_MODE`, `OCR_BACKEND`, ...) |

## 2. Startup sequence

```
__main__
 └─ ocr.init_ocr()        # blocking; builds 3 PaddleOCR pipelines
 └─ client.run(DISCORD_TOKEN)
```

`ocr.init_ocr()` is called once, synchronously, before the Discord client
connects. This is a deliberate design choice: PaddleOCR pipeline construction
downloads/loads model weights and is too slow to do lazily on the first
reaction (it would make the bot appear to hang on first use). The tradeoff is
a slower cold start — expect the process to sit for several seconds to a
couple of minutes on first run while models download and cache locally.

## 3. Runtime flow (`on_raw_reaction_add`)

```
reaction added
 ├─ is it our own reaction?              → ignore
 ├─ is it 🏓 (ping)?                     → reply with latency, stop
 ├─ is the emoji a known flag?           → LANGUAGES dict lookup, else ignore
 ├─ already translated this (msg, emoji)? → dedup via translated_cache, stop
 ├─ fetch the original message
 ├─ is the author a bot?                 → ignore
 │
 ├─ build text_to_translate:
 │   ├─ message.content, run through extract_tweet_text()
 │   │     (appends tweet body if a twitter/x/vxtwitter/etc. link is found)
 │   └─ first image attachment, run through ocr.extract_image_text()
 │         (appended as a labeled "[Extracted Image Text]" block)
 │
 ├─ nothing to translate at all?         → stop
 ├─ call Gemini (model.generate_content_async)
 └─ reply to the original message with an Embed containing the translation
```

Key implementation details:

- **Dedup cache** (`translated_cache`): an in-memory `set` of
  `(message_id, emoji)` pairs. Prevents double-processing the same
  message+flag reaction (e.g. from Discord retry/gateway replay). This is
  **not persisted** — restarting the bot clears it and the same emoji can
  trigger a fresh translation reply.
- **Multiple flags → multiple replies**: since the cache key includes the
  emoji, reacting with 🇯🇵 then 🇰🇷 on the same message produces two separate
  translation replies, not one.
- **Only the first image attachment is processed.** If a message has several
  image attachments, the rest are ignored.
- **Errors are swallowed**: the whole handler body is wrapped in a
  `try/except Exception`, which logs to stdout (and `debug_log` if
  `DEBUG_MODE=true`) but does not notify the channel. If the Gemini call or
  OCR step throws, the reaction simply produces no reply.

## 4. `LANGUAGES` — flag → target language

```python
LANGUAGES = {
    "🇺🇸": "English",  "🇬🇧": "English",
    "🇯🇵": "Japanese",
    "🇨🇳": "Chinese",  "🇹🇼": "Chinese",
    "🇰🇷": "Korean",
    "🇫🇷": "French",
    "🇪🇸": "Spanish",  "🇲🇽": "Spanish",
    "🇩🇪": "German",
    "🇷🇺": "Russian",
}
```

This dict drives both the Gemini translation target *and* which emojis the
bot reacts to at all — any other emoji is ignored. Note this list is broader
than the OCR language set (see below); OCR just can't *read* French, German,
or Russian source images, but the bot will happily translate French/German/
Russian **text or tweet content** into those languages.

## 5. OCR pipeline (`ocr.py`)

`ocr.py` is a **backend dispatcher** — it supports three interchangeable OCR
engines, selected once at startup via the `OCR_BACKEND` env var:

| `OCR_BACKEND` | Engine | Cost | Required env vars |
|---|---|---|---|
| `paddleocr` (default) | Local PaddleOCR, CPU | Free, self-hosted | none |
| `ocrspace` | OCR.space cloud API | Paid/free tier, per-request | `OCR_SPACE_API_KEY` |
| `gcv` | Google Cloud Vision API | Paid, per-request | `GOOGLE_VISION_API_KEY` |

Only one backend is active at a time — there's no fallback chain between
them. `extract_image_text(image_bytes, content_type=None)` is the single
entry point; it dispatches internally based on `OCR_BACKEND` and always
returns a blocking call regardless of which engine is selected, so the
caller in `app_gemini.py` doesn't need to know or care which backend is
configured.

Language coverage differs by backend:

- **`paddleocr`** only reads the 5 originally-required languages, split
  across 3 pipeline instances because no single PP-OCRv5 model covers all
  of them:

  | Pipeline key | `lang=` | Covers |
  |---|---|---|
  | `default` | `"ch"` | English, Japanese, Simplified Chinese, Traditional Chinese |
  | `korean` | `"korean"` | Korean |
  | `es` | `"es"` | Spanish |

  `_extract_paddleocr()`: decodes bytes to a BGR array via `cv2.imdecode`,
  runs the `default` pipeline first (cheapest — covers 4 of 5 languages), and
  falls through to `korean` then `es` if `default`'s average confidence is
  below `0.6` or it found no usable text, keeping whichever pipeline scored
  highest. Whitespace-only recognitions are filtered out before scoring — a
  pipeline can "detect" text regions in a script it can't read and report a
  misleadingly high confidence for blank/garbage output, which would
  otherwise short-circuit the fallback to the correct pipeline. All 3
  pipelines are built with `use_doc_orientation_classify=False`,
  `use_doc_unwarping=False`, `use_textline_orientation=False` (skip
  preprocessing not needed for typical screenshots/photos of text) and
  `enable_mkldnn=False` (works around a paddlepaddle 3.3.x CPU inference
  crash — see Troubleshooting below).

- **`ocrspace`** and **`gcv`** both auto-detect the source language natively
  in a single API call — no per-language pipeline selection needed, and both
  can read languages *beyond* the original 5 (whatever their respective
  auto-detect model supports). `OCR_SPACE_LANGUAGE` (default `"auto"`)
  controls OCR.space's `language` parameter if you ever need to pin it to a
  specific code instead of auto-detecting.

Returns `""` if no engine found usable text (caller treats this as "no text
to translate from the image"). Cloud backend requests use a 30s timeout and
return `""` on any request/API error (logged via `print`) rather than
raising, so a transient cloud outage degrades to "no image text found"
instead of crashing the reaction handler.

`paddleocr`/`numpy`/`cv2` are imported lazily inside the PaddleOCR-specific
functions rather than at module load time — so `ocr.py` imports cleanly (and
the `ocrspace`/`gcv` backends work) even in an environment that never
installs `paddleocr`, `paddlepaddle`, or `opencv`.

OCR inference (all three backends) is **blocking** — local PaddleOCR because
it's CPU-bound, and the cloud backends because they use synchronous
`requests` calls rather than `aiohttp` (this keeps `extract_image_text()` a
plain blocking function regardless of backend, so the dispatch pattern below
doesn't need to change depending on configuration). It's dispatched via

```python
loop.run_in_executor(None, ocr.extract_image_text, image_bytes, image_attachment.content_type)
```

from `app_gemini.py` so it doesn't stall the asyncio event loop (which would
otherwise freeze the Discord gateway heartbeat and other concurrent
reactions).

## 6. `extract_tweet_text` — link expansion

If `message.content` contains a `twitter.com` / `x.com` / `vxtwitter.com` /
`fxtwitter.com` / `fixupx.com` / `fixvx.com` status URL, the function fetches
the tweet body from the public `vxtwitter.com` API and appends it to the
content as a labeled block. Falls back silently to the original content on
any fetch failure or non-200 response.

## 7. Files

| File | Role |
|---|---|
| `app_gemini.py` | Discord client, event handlers, Gemini calls, tweet expansion |
| `ocr.py` | PaddleOCR pipeline management and image → text extraction |
| `requirements.txt` | `discord.py`, `openrouter`, `google-generativeai`, `python-dotenv`, `aiohttp`, `certifi`, `paddleocr`, `paddlepaddle`, `requests` |
| `.env` (not committed) | `DISCORD_TOKEN`, `GEMINI_API_KEY`, `DEBUG_MODE`, `OCR_BACKEND`, `OCR_SPACE_API_KEY`, `OCR_SPACE_LANGUAGE`, `GOOGLE_VISION_API_KEY` |

`openrouter` in `requirements.txt` is only used by `app.py`, not
`app_gemini.py` — kept in the shared requirements file since both bots live
in the same repo.

---

# Linux Deployment Troubleshooting

The code itself has no Windows-only paths (no OS-specific file handling), so
it runs on Linux as-is. The issues below are environment/dependency issues
you're likely to hit on a fresh Linux server or container — not bugs in this
code.

## `ImportError: libGL.so.1: cannot open shared object file`

**Cause:** `paddleocr` depends on `opencv-contrib-python`, which is built
against GUI/X11 libraries. Minimal servers and slim Docker base images
(`python:3.x-slim`, Alpine, etc.) don't have these installed.

**Fix:** install the missing system libraries — no code/requirements change
needed:

```bash
# Debian/Ubuntu
sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0

# RHEL/CentOS/Fedora
sudo dnf install -y mesa-libGL glib2
```

In a Dockerfile:

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
```

## `NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support [...]`

**Cause:** a known regression in `paddlepaddle` 3.3.x's oneDNN/PIR CPU
inference path (tracked upstream at
[PaddlePaddle/Paddle#77340](https://github.com/PaddlePaddle/Paddle/issues/77340)
and
[PaddlePaddle/PaddleOCR#17539](https://github.com/PaddlePaddle/PaddleOCR/issues/17539)).
It reproduces on both Linux and Windows.

**Fix:** already applied in this repo — `ocr.py` constructs every
`PaddleOCR(...)` pipeline with `enable_mkldnn=False`. If you see this error
anyway, confirm you're actually running the code in this repo (not a stale
`__pycache__` or an old venv) and that `paddlepaddle` resolved to `3.3.x` (a
downgrade to `paddlepaddle==3.2.0` is the documented community workaround if
you ever need to disable the `enable_mkldnn=False` workaround for
performance reasons).

## First run hangs / takes minutes to start

**Cause:** `ocr.init_ocr()` runs at startup and downloads PP-OCRv5 model
weights (3 pipelines' worth — `default`, `korean`, `es`) on first use into a
local cache (default `~/.paddlex/`). This requires outbound internet access
and a writable `$HOME`.

**Fix:**
- Make sure the process's user has a writable home directory (common failure
  in containers running as a non-root user with `$HOME=/` or unset).
- Make sure the container/server has outbound HTTPS access on first boot.
- To avoid the download entirely at deploy time, pre-warm the cache by
  running the bot once (or a short script calling `ocr.init_ocr()`) during
  image build, then bake `~/.paddlex/` into the image layer.

## High memory usage / OOM-killed

**Cause:** PaddleOCR pipelines are not free — 3 pipelines loaded
simultaneously (as this bot does, by design, to avoid lazy-load latency on
first reaction) hold multiple models in memory at once. Some PP-OCRv5
language models are known to be memory-heavy on CPU inference (see
[PaddleOCR#17955](https://github.com/PaddlePaddle/PaddleOCR/issues/17955)
for an extreme case with a different language model).

**Fix:** size the server/container with headroom (a few GB free beyond the
Python baseline is a safe starting point) and monitor actual RSS after
`init_ocr()` completes before deciding on a memory limit. If memory is tight,
the `default`/`korean`/`es` split in `ocr.py` could be changed to lazy-load
per-pipeline instead of eager-loading all three at startup — that's a
deliberate current tradeoff (startup latency vs. memory), not a bug.

## `cv2.imdecode` returns `None` / "no text found" for a valid image

**Cause:** `_predict()` in `ocr.py` returns `("", 0.0)` if `cv2.imdecode`
fails to decode the attachment bytes — usually because the attachment's
`content_type` says `image/*` but the bytes are corrupted, truncated, or an
unsupported/exotic format (e.g. some HEIC variants) that this OpenCV build
can't decode.

**Fix:** not Linux-specific, but worth checking first — confirm the image
opens normally in a standard viewer and is a common format (PNG/JPEG/WEBP).

## Discord gateway seems to freeze during OCR

**Cause:** if you ever add code that calls `ocr.extract_image_text()`
directly (bypassing `loop.run_in_executor`), it will block the asyncio event
loop for the duration of CPU inference, stalling the gateway heartbeat and
delaying all other event handling.

**Fix:** always dispatch OCR calls via
`loop.run_in_executor(None, ocr.extract_image_text, image_bytes)`, as
`app_gemini.py` already does — don't call it directly from an `async def`.
