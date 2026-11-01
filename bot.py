"""Shared Discord translation-bot core.

A backend supplies a `translate(content, target_language, images) -> (text, model)`
coroutine; this module handles Discord wiring: flag reactions, de-duplication,
embeds and the ping command.
"""

import logging
import os
from typing import Awaitable, Callable

import discord
from cachetools import TTLCache
from dotenv import load_dotenv

load_dotenv()

DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("discord").setLevel(logging.WARNING)
log = logging.getLogger("transbot")

LANGUAGES = {
    "🇺🇸": "English", "🇬🇧": "UK English",
    "🇯🇵": "Japanese",
    "🇨🇳": "Chinese", "🇹🇼": "Traditional Chinese (Taiwan)",
    "🇭🇰": "Cantonese",
    "🇰🇷": "Korean",
    "🇫🇷": "French",
    "🇪🇸": "Spanish", "🇲🇽": "Spanish",
    "🇩🇪": "German",
    "🇷🇺": "Russian",
}

EMBED_DESCRIPTION_LIMIT = 4096  # Discord's hard cap on embed descriptions.
# MIME types both backends (Gemini + OpenAI-compatible) reliably accept.
SUPPORTED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp"}
MAX_IMAGES = 4
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_TOTAL_IMAGE_BYTES = 16 * 1024 * 1024

# Backend contract: async (content, target_language, images=[(bytes, mime)]) -> (text, model).
Translator = Callable[[str, str, list[tuple[bytes, str]]], Awaitable[tuple[str, str]]]


def require_env(name: str) -> str:
    """Return an environment variable's value, or exit with a clear message."""
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"{name} is not set. Add it to your .env file.")
    return value


def build_prompt(target_language: str) -> str:
    """The system instruction shared by every backend."""
    return (
        f"Translate any text in the message and images to {target_language}. "
        "If an image has no text, briefly describe it in that language. "
        "Preserve tone, slang, and formatting. Output ONLY the translation."
    )


async def _fetch_message(client: discord.Client, payload) -> discord.Message:
    """Resolve the message a reaction was added to (cached channel, else fetched)."""
    channel = client.get_channel(payload.channel_id) or await client.fetch_channel(
        payload.channel_id
    )
    return await channel.fetch_message(payload.message_id)


async def _reply_pong(client: discord.Client, payload) -> None:
    try:
        latency = round(client.latency * 1000)
        message = await _fetch_message(client, payload)
        await message.reply(f"🏓 Pong! Latency: **{latency}ms**")
    except Exception:
        log.exception("Ping failed")


async def _collect_images(message: discord.Message) -> list[tuple[bytes, str]]:
    """Read supported image attachments as (bytes, mime), capped in count and size."""
    images: list[tuple[bytes, str]] = []
    total = 0
    for attachment in message.attachments:
        mime = (attachment.content_type or "").split(";")[0].strip().lower()
        if mime not in SUPPORTED_IMAGE_TYPES:
            continue
        if attachment.size > MAX_IMAGE_BYTES or total + attachment.size > MAX_TOTAL_IMAGE_BYTES:
            continue
        images.append((await attachment.read(), mime))
        total += attachment.size
        if len(images) >= MAX_IMAGES:
            break
    return images


async def _reply_translation(
    translate: Translator, message: discord.Message, emoji: str
) -> bool:
    """Translate the message and reply. Returns True if a reply was sent."""
    if message.author.bot:
        return False

    images = await _collect_images(message)
    if not message.content and not images:
        if message.attachments:
            kinds = ", ".join(sorted(t.split("/")[1] for t in SUPPORTED_IMAGE_TYPES))
            await message.reply(
                f"⚠️ No text or supported image found "
                f"({kinds}, ≤{MAX_IMAGE_BYTES // 1024 // 1024} MB each)."
            )
            return True
        return False

    target = LANGUAGES[emoji]
    log.info("Translating message %s to %s", message.id, target)

    async with message.channel.typing():
        translated, model_label = await translate(message.content, target, images)

    if not translated or not translated.strip():
        log.warning("Empty translation for message %s", message.id)
        return False

    if len(translated) > EMBED_DESCRIPTION_LIMIT:
        translated = translated[: EMBED_DESCRIPTION_LIMIT - 1] + "…"

    embed = discord.Embed(title=f"{emoji} Translation", description=translated)
    embed.set_footer(text=f"Model: {model_label}")
    await message.reply(embed=embed)
    return True


def run(translate: Translator, *, ready_message: str) -> None:
    """Start a Discord bot that translates messages via `translate`."""
    token = require_env("DISCORD_TOKEN")

    intents = discord.Intents.default()
    intents.message_content = True
    intents.reactions = True

    client = discord.Client(intents=intents)

    # Dedup (message_id, emoji); bounded in size and time so it can't leak memory.
    seen: TTLCache = TTLCache(maxsize=10_000, ttl=3600)

    @client.event
    async def on_ready():
        log.info("Logged in as %s. %s", client.user, ready_message)

    @client.event
    async def on_raw_reaction_add(payload: discord.RawReactionActionEvent):
        if client.user is None or payload.user_id == client.user.id:
            return

        emoji = str(payload.emoji)

        if emoji == "🏓":
            await _reply_pong(client, payload)
            return

        if emoji not in LANGUAGES:
            return

        key = (payload.message_id, emoji)
        if key in seen:
            return
        seen[key] = True  # reserve now so concurrent reactions don't double-translate

        try:
            message = await _fetch_message(client, payload)
            if not await _reply_translation(translate, message, emoji):
                seen.pop(key, None)  # nothing sent — let the user retry
        except (discord.NotFound, discord.Forbidden):
            # Message deleted or inaccessible: keep the key so we don't keep retrying.
            log.warning("Message %s unavailable (deleted or no access)", payload.message_id)
        except Exception:
            seen.pop(key, None)  # transient failure — let the user retry
            log.exception("Translation failed for message %s", payload.message_id)

    client.run(token, log_handler=None)
