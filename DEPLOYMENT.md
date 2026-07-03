# Deployment Guide

How to get `app_gemini.py` (the Gemini-based translation bot, with OCR
support) running in production. For architecture details see
[APP_GEMINI.md](APP_GEMINI.md).

## 1. Prerequisites

- Python 3.10+ (verified against 3.13; PaddleOCR/paddlepaddle wheels need a
  Python version they publish `manylinux`/`win_amd64` binaries for — check
  before pinning an unusual version).
- A Discord bot application + token (Discord Developer Portal), with the
  **Message Content** privileged intent enabled — `app_gemini.py` sets
  `intents.message_content = True`, which requires this to be turned on in
  the portal or the bot silently receives empty `message.content`.
- A Google Gemini API key.
- If using a cloud OCR backend instead of the default local PaddleOCR: an
  OCR.space API key or a Google Cloud Vision API key (see §4).

## 2. Get the code and install dependencies

```bash
git clone <this-repo-url>
cd transbot
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

`requirements.txt` installs `paddleocr` + `paddlepaddle` (CPU) by default,
which is a sizeable download (models are fetched separately on first run,
see §5). If you plan to only use a cloud OCR backend, you can skip installing
those two packages — `ocr.py` imports them lazily and only requires them
when `OCR_BACKEND=paddleocr` (the default).

## 3. Configure `.env`

Create a `.env` file in the project root (already gitignored — never commit
this file):

```dotenv
DISCORD_TOKEN=your-discord-bot-token
GEMINI_API_KEY=your-gemini-api-key
DEBUG_MODE=false

# Optional — OCR backend selection, defaults to paddleocr if unset
OCR_BACKEND=paddleocr
```

Set `DEBUG_MODE=true` temporarily while verifying a new deployment — it logs
each step of the reaction-handling flow (attachment detection, OCR result
length, Gemini call, etc.) to stdout.

## 4. Choosing an OCR backend

| `OCR_BACKEND` | Setup | Notes |
|---|---|---|
| `paddleocr` (default) | Nothing extra — just `requirements.txt` | Free, runs locally on CPU, ~3 model pipelines loaded at startup (see §5). Best for privacy-sensitive deployments (images never leave the server) or high-volume/low-cost usage. |
| `ocrspace` | Add `OCR_SPACE_API_KEY=...` to `.env` | Cloud API, per-request cost (has a free tier). Faster cold start since no local models to load. |
| `gcv` | Add `GOOGLE_VISION_API_KEY=...` to `.env` | Cloud API (Google Cloud Vision), per-request cost, broadest language auto-detection. |

Only one backend runs at a time. Missing the required API key for
`ocrspace`/`gcv` causes the bot to fail fast at startup with a clear error,
rather than failing silently on the first reaction — check your logs on
first boot.

## 5. First run

```bash
python app_gemini.py
```

Expect a slower-than-usual first startup if using the default `paddleocr`
backend: `ocr.init_ocr()` runs before the bot connects to Discord and
downloads PP-OCRv5 model weights (3 pipelines' worth) into a local cache
(`~/.paddlex/` by default). This needs outbound internet access and a
writable home directory. Subsequent restarts reuse the cache and start much
faster. Cloud backends (`ocrspace`/`gcv`) skip this step entirely — startup
is near-instant.

You should see:
```
Loading OCR models...
OCR models loaded.
==================================================
Logged in as <YourBotName>
Translation bot (Gemini Direct) is online.
==================================================
```

## 6. Keeping it running (Linux)

### Option A: systemd service (recommended for a VPS/bare server)

Create `/etc/systemd/system/transbot.service`:

```ini
[Unit]
Description=Discord translation bot (Gemini + OCR)
After=network-online.target

[Service]
Type=simple
User=transbot
WorkingDirectory=/opt/transbot
EnvironmentFile=/opt/transbot/.env
ExecStart=/opt/transbot/.venv/bin/python app_gemini.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now transbot
sudo journalctl -u transbot -f    # follow logs
```

Running as a dedicated non-root user (`transbot` above) with its own home
directory matters if you're using the `paddleocr` backend — the model cache
needs a writable `$HOME` (see §5 and the Linux troubleshooting notes in
[APP_GEMINI.md](APP_GEMINI.md)).

### Option B: Docker

```dockerfile
FROM python:3.13-slim

# libgl1/libglib2.0-0 are required by opencv (a paddleocr dependency) even
# if you only use the paddleocr backend for OCR decoding — see
# APP_GEMINI.md's Linux troubleshooting section for why.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app_gemini.py ocr.py ./

CMD ["python", "app_gemini.py"]
```

```bash
docker build -t transbot .
docker run -d --name transbot --restart unless-stopped --env-file .env transbot
```

If using the `paddleocr` backend, mount a persistent volume for the model
cache so it isn't re-downloaded on every container restart:

```bash
docker run -d --name transbot --restart unless-stopped \
  --env-file .env \
  -v transbot_paddlex_cache:/root/.paddlex \
  transbot
```

## 7. Updating

```bash
git pull
pip install -r requirements.txt --upgrade
sudo systemctl restart transbot     # or: docker restart transbot
```

## 8. Troubleshooting

See the "Linux Deployment Troubleshooting" section at the bottom of
[APP_GEMINI.md](APP_GEMINI.md) for the specific errors you're most likely to
hit on a fresh Linux server (`libGL.so.1` missing, the paddlepaddle 3.3.x
oneDNN CPU bug, model-download permissions, memory sizing).
