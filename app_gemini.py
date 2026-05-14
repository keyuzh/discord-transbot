import os
import discord
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv

# =========================
# LOAD ENV
# =========================

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

# =========================
# DEBUG PRINT
# =========================

def debug_log(message):
    if DEBUG_MODE:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [DEBUG] {message}")

# =========================
# GEMINI CONFIG
# =========================

genai.configure(api_key=GEMINI_API_KEY)

# Generation config
generation_config = {
  "temperature": 0.3,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-3.1-flash-lite",
  generation_config=generation_config,
)

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
# =========================

translated_cache = set()

# =========================
# READY EVENT
# =========================

@client.event
async def on_ready():
    print("=" * 50)
    print(f"Logged in as {client.user}")
    print("Translation bot (Gemini Direct) is online.")
    print("=" * 50)

# =========================
# REACTION EVENT
# =========================

@client.event
async def on_raw_reaction_add(payload):
    if payload.user_id == client.user.id:
        return

    emoji = str(payload.emoji)
    debug_log(f"Reaction received: {emoji} from user {payload.user_id}")

    if emoji not in LANGUAGES:
        debug_log(f"Emoji {emoji} not in supported LANGUAGES.")
        return

    cache_key = (payload.message_id, emoji)
    if cache_key in translated_cache:
        debug_log(f"Translation for {cache_key} already in cache. Skipping.")
        return

    translated_cache.add(cache_key)

    try:
        channel = client.get_channel(payload.channel_id)
        if channel is None:
            debug_log(f"Channel {payload.channel_id} not found.")
            return

        debug_log(f"Fetching message {payload.message_id}...")
        message = await channel.fetch_message(payload.message_id)

        if message.author.bot or not message.content:
            debug_log("Ignoring bot message or empty content.")
            return

        target_language = LANGUAGES[emoji]
        print(f"Translating to {target_language}")
        debug_log(f"Original content: {message.content[:50]}...")

        # =========================
        # CALL GEMINI API
        # =========================
        
        prompt = (
            f"Translate to {target_language}. "
            "Preserve tone, slang, and formatting. Output ONLY the translation.\n\n"
            f"Content: {message.content}"
        )
        
        debug_log("Calling Gemini API...")
        response = await model.generate_content_async(prompt)
        translated = response.text
        
        debug_log(f"Translation received: {translated[:50]}...")

        # =========================
        # EMBED
        # =========================

        embed = discord.Embed(
            title=f"{emoji} Translation",
            description=translated
        )
        embed.set_footer(text=f"Model: {model.model_name}")

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
