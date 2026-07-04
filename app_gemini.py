"""Discord translation bot backed by Google Gemini, with X/Twitter link expansion."""

import asyncio
import os
import re

import aiohttp
from google import genai
from google.genai import types

import bot  # imports first so load_dotenv() runs before we read the key

API_KEY = bot.require_env("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
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


async def _fetch_tweet(session: aiohttp.ClientSession, username: str, tweet_id: str) -> str:
    api_url = f"https://api.vxtwitter.com/{username}/status/{tweet_id}"
    bot.log.debug("Fetching tweet from %s", api_url)
    try:
        async with session.get(api_url) as response:
            if response.status == 200:
                return (await response.json()).get("text", "")
            bot.log.warning("Tweet fetch returned HTTP %s", response.status)
    except Exception:
        bot.log.exception("Tweet fetch failed")
    return ""


async def _expand_tweets(content: str) -> str:
    """Append the text of any X/Twitter links so the model can translate them."""
    matches = _TWEET_RE.findall(content)[:MAX_TWEETS]
    if not matches:
        return content

    timeout = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        texts = await asyncio.gather(*(_fetch_tweet(session, u, t) for u, t in matches))

    extracted = "\n\n".join(f"**[Extracted Tweet Text]:**\n{t}" for t in texts if t)
    return f"{content}\n\n{extracted}" if extracted else content


async def translate(
    content: str, target_language: str, images: list[tuple[bytes, str]]
) -> tuple[str, str]:
    text = await _expand_tweets(content) if content else ""
    parts = [types.Part.from_bytes(data=data, mime_type=mime) for data, mime in images]
    contents = [text, *parts] if text else parts

    response = await client.aio.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=bot.build_prompt(target_language), **_GEN_PARAMS
        ),
    )

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
