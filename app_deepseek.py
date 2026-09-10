"""Discord translation bot backed directly by the DeepSeek API."""

from __future__ import annotations

import asyncio
import base64
import os
import re

import aiohttp

import bot  # imports first so load_dotenv() runs before we read the key

API_KEY = bot.require_env("DEEPSEEK_API_KEY")
# The vision model is required for Discord attachments and images from X/Twitter links.
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash-vision-exp")
API_URL = "https://api.deepseek.com/chat/completions"
MAX_TWEETS = 3

# Matches twitter/x and the usual mirror domains, capturing (username, status_id).
_TWEET_RE = re.compile(
    r"https?://(?:www\.)?"
    r"(?:twitter\.com|x\.com|vxtwitter\.com|fxtwitter\.com|fixupx\.com|fixvx\.com)"
    r"/([A-Za-z0-9_]+)/status/([0-9]+)"
)
_PROFILE_RE = re.compile(
    r"https?://(?:www\.)?"
    r"(?:twitter\.com|x\.com|vxtwitter\.com|fxtwitter\.com|fixupx\.com|fixvx\.com)"
    r"/([A-Za-z0-9_]{1,15})/?(?![\w/])"
)
_RESERVED_PATHS = {
    "home", "search", "explore", "i", "intent", "hashtag", "settings",
    "notifications", "messages", "compose", "login", "share",
}


def _extract(tweet: dict) -> tuple[str, list[str]]:
    """Pull (text, image URLs) from a tweet; videos use their cover image."""
    urls = [
        media.get("url") if media.get("type") == "image" else media.get("thumbnail_url")
        for media in tweet.get("media_extended") or []
    ]
    return tweet.get("text", ""), [url for url in urls if url]


async def _fetch_tweet(
    session: aiohttp.ClientSession, username: str, tweet_id: str
) -> tuple[str, list[str]]:
    """Fetch tweet text and media from the public vxtwitter API."""
    api_url = f"https://api.vxtwitter.com/{username}/status/{tweet_id}"
    bot.log.debug("Fetching tweet from %s", api_url)
    try:
        async with session.get(api_url) as response:
            if response.status == 200:
                data = await response.json()
                text, urls = _extract(data)
                if isinstance(quoted_tweet := data.get("qrt"), dict):
                    quoted_text, quoted_urls = _extract(quoted_tweet)
                    if quoted_text:
                        text = "\n".join(
                            filter(None, [text, f"[Quoted tweet]: {quoted_text}"])
                        )
                    urls += quoted_urls
                return text, urls
            bot.log.warning("Tweet fetch returned HTTP %s", response.status)
    except Exception:
        bot.log.exception("Tweet fetch failed")
    return "", []


async def _fetch_profile(session: aiohttp.ClientSession, username: str) -> str:
    """Fetch a profile's display name and bio from the public fxtwitter API."""
    api_url = f"https://api.fxtwitter.com/{username}"
    bot.log.debug("Fetching profile from %s", api_url)
    try:
        async with session.get(api_url) as response:
            if response.status == 200:
                user = (await response.json()).get("user") or {}
                return "\n".join(filter(None, [user.get("name"), user.get("description")]))
            bot.log.warning("Profile fetch returned HTTP %s", response.status)
    except Exception:
        bot.log.exception("Profile fetch failed")
    return ""


async def _fetch_media(
    session: aiohttp.ClientSession, url: str
) -> tuple[bytes, str] | None:
    """Download a supported tweet image, returning None for invalid/oversized media."""
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
    """Append X/Twitter post text and profile bios; return linked images."""
    matches = _TWEET_RE.findall(content)[:MAX_TWEETS]
    usernames = [
        name for name in _PROFILE_RE.findall(content)
        if name.lower() not in _RESERVED_PATHS
    ][:MAX_TWEETS]
    if not matches and not usernames:
        return content, []

    timeout = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tweets, profiles = await asyncio.gather(
            asyncio.gather(*(_fetch_tweet(session, user, post) for user, post in matches)),
            asyncio.gather(*(_fetch_profile(session, user) for user in usernames)),
        )
        urls = [url for _, media in tweets for url in media][:limit]
        media = await asyncio.gather(*(_fetch_media(session, url) for url in urls))

    sections = [f"**[Extracted Tweet Text]:**\n{text}" for text, _ in tweets if text]
    sections += [f"**[Extracted Profile Bio]:**\n{bio}" for bio in profiles if bio]
    extracted = "\n\n".join(sections)
    text = f"{content}\n\n{extracted}" if extracted else content
    return text, [item for item in media if item]


def _within_budget(images: list[tuple[bytes, str]]) -> list[tuple[bytes, str]]:
    """Cap combined image count and size, prioritizing Discord attachments."""
    kept: list[tuple[bytes, str]] = []
    total = 0
    for data, mime in images:
        if len(kept) >= bot.MAX_IMAGES or total + len(data) > bot.MAX_TOTAL_IMAGE_BYTES:
            continue
        kept.append((data, mime))
        total += len(data)
    return kept


async def _create_completion(messages: list[dict]) -> dict:
    """Call DeepSeek's OpenAI-compatible API directly, retrying transient failures."""
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {"model": MODEL, "messages": messages, "temperature": 0.3}
    timeout = aiohttp.ClientTimeout(total=120)
    delay = 2.0

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for attempt in range(4):
            try:
                async with session.post(API_URL, headers=headers, json=payload) as response:
                    if response.status < 400:
                        return await response.json()
                    detail = await response.text()
                    if response.status not in (429, 500, 502, 503, 504) or attempt == 3:
                        raise RuntimeError(
                            f"DeepSeek API returned HTTP {response.status}: {detail[:500]}"
                        )
                    bot.log.warning(
                        "DeepSeek API returned HTTP %s; retrying in %.1fs",
                        response.status,
                        delay,
                    )
            except aiohttp.ClientError as error:
                if attempt == 3:
                    raise RuntimeError(f"DeepSeek API request failed: {error}") from error
                bot.log.warning("DeepSeek API request failed; retrying in %.1fs: %s", delay, error)
            await asyncio.sleep(delay)
            delay *= 2

    raise RuntimeError("DeepSeek API request exhausted retries")


async def translate(
    content: str, target_language: str, images: list[tuple[bytes, str]]
) -> tuple[str, str]:
    content, tweet_images = await _expand_tweets(
        content, max(0, bot.MAX_IMAGES - len(images))
    )
    user_content: list[dict] = []
    if content:
        user_content.append({"type": "text", "text": content})
    for data, mime in _within_budget(images + tweet_images):
        data_uri = f"data:{mime};base64,{base64.b64encode(data).decode()}"
        user_content.append({"type": "image_url", "image_url": {"url": data_uri}})

    response = await _create_completion([
        {"role": "system", "content": bot.build_prompt(target_language)},
        {"role": "user", "content": user_content},
    ])
    choices = response.get("choices") or []
    message = choices[0].get("message") if choices else None
    translated = message.get("content") if message else None
    return (translated or ""), response.get("model") or MODEL


if __name__ == "__main__":
    bot.run(translate, ready_message="Translation bot (DeepSeek direct) is online.")
