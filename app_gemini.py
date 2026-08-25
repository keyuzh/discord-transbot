"""Discord translation bot backed by Google Gemini, with X/Twitter link expansion."""

from __future__ import annotations

import asyncio
import os
import re

import aiohttp
from google import genai
from google.genai import types

import bot  # imports first so load_dotenv() runs before we read the key

API_KEY = bot.require_env("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL")
MAX_TWEETS = 3

# One client for the whole process; its connection pool is reused across requests.
client = genai.Client(api_key=API_KEY)

# Sampling params are static; only the system instruction varies per request.
_GEN_PARAMS = dict(temperature=0.3, top_p=0.95, top_k=40, max_output_tokens=8192)

# Matches twitter/x and the usual mirror domains, capturing (username, status_id).
_TWEET_RE = re.compile(
    r"https?://(?:www\.)?"
    r"(?:twitter\.com|x\.com|vxtwitter\.com|fxtwitter\.com|fixupx\.com|fixvx\.com)"
    r"/([A-Za-z0-9_]+)/status/([0-9]+)"
)


def _extract(tweet: dict) -> tuple[str, list[str]]:
    """Pull (text, image urls) from a tweet object; videos fall back to a jpg cover."""
    urls = [
        m.get("url") if m.get("type") == "image" else m.get("thumbnail_url")
        for m in tweet.get("media_extended") or []
    ]
    return tweet.get("text", ""), [u for u in urls if u]


async def _fetch_tweet(
    session: aiohttp.ClientSession, username: str, tweet_id: str
) -> tuple[str, list[str]]:
    api_url = f"https://api.vxtwitter.com/{username}/status/{tweet_id}"
    bot.log.debug("Fetching tweet from %s", api_url)
    try:
        async with session.get(api_url) as response:
            if response.status == 200:
                data = await response.json()
                text, urls = _extract(data)
                if isinstance(qrt := data.get("qrt"), dict):  # quoted tweet, own object
                    q_text, q_urls = _extract(qrt)
                    if q_text:
                        text = "\n".join(filter(None, [text, f"[Quoted tweet]: {q_text}"]))
                    urls += q_urls
                return text, urls
            bot.log.warning("Tweet fetch returned HTTP %s", response.status)
    except Exception:
        bot.log.exception("Tweet fetch failed")
    return "", []


async def _fetch_media(
    session: aiohttp.ClientSession, url: str
) -> tuple[bytes, str] | None:
    """Download a tweet image as (bytes, mime); None if unsupported or too large."""
    try:
        async with session.get(url, allow_redirects=False) as response:
            mime = (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if response.status != 200 or mime not in bot.SUPPORTED_IMAGE_TYPES:
                return None
            if int(response.headers.get("Content-Length") or 0) > bot.MAX_IMAGE_BYTES:
                return None
            data = await response.read()
            return (data, mime) if len(data) <= bot.MAX_IMAGE_BYTES else None
    except Exception:
        bot.log.exception("Media fetch failed: %s", url)
        return None


async def _expand_tweets(content: str, limit: int) -> tuple[str, list[tuple[bytes, str]]]:
    """Return the message plus any X/Twitter text, and up to `limit` tweet images."""
    matches = _TWEET_RE.findall(content)[:MAX_TWEETS]
    if not matches:
        return content, []

    timeout = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tweets = await asyncio.gather(*(_fetch_tweet(session, u, t) for u, t in matches))
        urls = [url for _, media in tweets for url in media][:limit]
        media = await asyncio.gather(*(_fetch_media(session, u) for u in urls))

    extracted = "\n\n".join(f"**[Extracted Tweet Text]:**\n{t}" for t, _ in tweets if t)
    text = f"{content}\n\n{extracted}" if extracted else content
    return text, [m for m in media if m]


def _within_budget(images: list[tuple[bytes, str]]) -> list[tuple[bytes, str]]:
    """Cap the combined images by count and total size (attachments come first)."""
    kept: list[tuple[bytes, str]] = []
    total = 0
    for data, mime in images:
        if len(kept) >= bot.MAX_IMAGES or total + len(data) > bot.MAX_TOTAL_IMAGE_BYTES:
            continue
        kept.append((data, mime))
        total += len(data)
    return kept


async def translate(
    content: str, target_language: str, images: list[tuple[bytes, str]]
) -> tuple[str, str]:
    text, tweet_images = await _expand_tweets(content, max(0, bot.MAX_IMAGES - len(images)))
    parts = [
        types.Part.from_bytes(data=data, mime_type=mime)
        for data, mime in _within_budget(images + tweet_images)
    ]
    contents = [text, *parts] if text else parts

    start_time = asyncio.get_running_loop().time()
    delay = 10.0  # initial delay in seconds

    while True:
        try:
            response = await client.aio.models.generate_content(
                model=MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=bot.build_prompt(target_language), **_GEN_PARAMS
                ),
            )
            break
        except google.genai.errors.ServerError as e:
            if getattr(e, "code", None) not in (500, 503):
                raise
            elapsed = asyncio.get_running_loop().time() - start_time
            if elapsed >= 300:  # 5 minutes
                bot.log.error("Gemini API returned HTTP %s after 5 minutes of retries, stopping.", e.code)
                raise
            bot.log.warning("Gemini API returned HTTP %s, retrying in %.1fs (elapsed: %.1fs)...", e.code, delay, elapsed)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 10.0)
        except Exception as e:
            # Check if it's a generic APIError or has status code 500/503
            code = getattr(e, "code", None)
            if code not in (500, 503):
                # Also check response status code if available
                resp = getattr(e, "response", None)
                code = getattr(resp, "status", getattr(resp, "status_code", code))
            if code not in (500, 503):
                raise
            elapsed = asyncio.get_running_loop().time() - start_time
            if elapsed >= 300:
                bot.log.error("Gemini API returned HTTP %s after 5 minutes of retries, stopping.", code)
                raise
            bot.log.warning("Gemini API returned HTTP %s, retrying in %.1fs (elapsed: %.1fs)...", code, delay, elapsed)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 10.0)

    # `.text` is None when the reply was blocked or has no text parts.
    translated = response.text
    if not translated:
        bot.log.warning(
            "Gemini returned no content: %s", getattr(response, "prompt_feedback", None)
        )
        return "", MODEL

    return translated, response.model_version or MODEL


if __name__ == "__main__":
    bot.run(translate, ready_message="Translation bot (Gemini) is online.")
