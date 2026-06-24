import os
import discord
from datetime import datetime
from openrouter import OpenRouter
from dotenv import load_dotenv

# =========================
# LOAD ENV
# =========================

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

# =========================
# DEBUG PRINT
# =========================

def debug_log(message):
    if DEBUG_MODE:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [DEBUG] {message}")

# =========================
# DISCORD INTENTS
# =========================

intents = discord.Intents.default()

intents.message_content = True
intents.reactions = True
intents.messages = True
intents.guilds = True

# =========================
# DISCORD CLIENT
# =========================

client = discord.Client(intents=intents)

#using the following codes if using a vpn with port7890
#client = discord.Client(
#    intents=intents,
#    proxy="http://127.0.0.1:7890"
#)

# =========================
# FLAG -> LANGUAGE
# =========================

LANGUAGES = {
    "🇺🇸": "English",
    "🇬🇧": "English",
    "🇯🇵": "Japanese",
    "🇨🇳": "Chinese",
    "🇹🇼": "Chinese",
    "🇰🇷": "Korean",
    "🇫🇷": "French",
    "🇪🇸": "Spanish",
    "🇲🇽": "Spanish",
    "🇩🇪": "German",
    "🇷🇺": "Russian"
}

# =========================
# CACHE
# prevents duplicate translations
# =========================

translated_cache = set()

# =========================
# READY EVENT
# =========================

@client.event
async def on_ready():

    print("=" * 50)
    print(f"Logged in as {client.user}")
    print("Translation bot is online.")
    print("=" * 50)

# =========================
# REACTION EVENT
# =========================

@client.event
async def on_raw_reaction_add(payload):

    # ignore bot reactions
    if payload.user_id == client.user.id:
        return

    emoji = str(payload.emoji)
    debug_log(f"Reaction received: {emoji} from user {payload.user_id}")

    # Handle ping command
    if emoji == "🏓":
        latency = round(client.latency * 1000)
        debug_log(f"Ping requested. Latency: {latency}ms")
        
        channel = client.get_channel(payload.channel_id)
        if channel:
            message = await channel.fetch_message(payload.message_id)
            await message.reply(f"🏓 Pong! Latency: **{latency}ms**")
        return

    # only process flag emojis
    if emoji not in LANGUAGES:
        debug_log(f"Emoji {emoji} not in supported LANGUAGES.")
        return

    cache_key = (payload.message_id, emoji)

    # prevent duplicate translations
    if cache_key in translated_cache:
        debug_log(f"Translation for {cache_key} already in cache. Skipping.")
        return

    translated_cache.add(cache_key)

    try:
        # get channel
        channel = client.get_channel(payload.channel_id)

        if channel is None:
            debug_log(f"Channel {payload.channel_id} not found.")
            return

        # fetch original message
        debug_log(f"Fetching message {payload.message_id}...")
        message = await channel.fetch_message(payload.message_id)

        # ignore bot messages
        if message.author.bot:
            debug_log("Ignoring bot message.")
            return

        # ignore empty messages
        if not message.content:
            debug_log("Ignoring empty message content.")
            return

        target_language = LANGUAGES[emoji]

        print(f"Translating to {target_language}")
        debug_log(f"Original content: {message.content[:50]}...")

        # =========================
        # CALL OPENROUTER API (SDK)
        # =========================

        api_model = "openrouter/free"

        debug_log(f"Calling OpenRouter API for model: {api_model}")
        
        async with OpenRouter(api_key=OPENROUTER_API_KEY) as ai:
            response = await ai.chat.send_async(
                model=api_model,
                messages=[
                    {
                        "role": "system",
                        "content": f"Translate to {target_language}. Preserve tone, slang, and formatting. Output ONLY the translation."
                    },
                    {
                        "role": "user",
                        "content": message.content
                    }
                ],
                temperature=0.3
            )

        translated = response.choices[0].message.content
        actual_model = response.model
        debug_log(f"Translation received from {actual_model}: {translated[:50]}...")

        # =========================
        # EMBED
        # =========================

        embed = discord.Embed(
            title=f"{emoji} Translation",
            description=translated
        )

        embed.set_footer(text=f"Model: {actual_model}")

        # reply to original message
        debug_log(f"Replying to message {payload.message_id} with translation.")
        await message.reply(embed=embed)

    except Exception as e:

        print("Translation error:")
        print(e)
        debug_log(f"Exception details: {type(e).__name__}: {str(e)}")

# =========================
# START BOT
# =========================

if __name__ == '__main__':
    client.run(DISCORD_TOKEN)
