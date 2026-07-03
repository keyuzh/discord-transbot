import os
import asyncio
import discord
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv
import re
import aiohttp
import ocr

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

    # Handle ping command
    if emoji == "🏓":
        latency = round(client.latency * 1000)
        debug_log(f"Ping requested. Latency: {latency}ms")
        
        channel = client.get_channel(payload.channel_id)
        if channel:
            message = await channel.fetch_message(payload.message_id)
            await message.reply(f"🏓 Pong! Latency: **{latency}ms**")
        return

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

        if message.author.bot:
            debug_log("Ignoring bot message.")
            return

        target_language = LANGUAGES[emoji]
        print(f"Translating to {target_language}")
        debug_log(f"Original content: {message.content[:50]}...")

        # PROCESS TWEET LINKS
        text_to_translate = await extract_tweet_text(message.content) if message.content else ""

        # PROCESS IMAGE ATTACHMENT (OCR)
        image_attachment = next(
            (a for a in message.attachments if a.content_type and a.content_type.startswith("image/")),
            None
        )

        if image_attachment:
            debug_log(f"Found image attachment: {image_attachment.filename}")
            image_bytes = await image_attachment.read()

            loop = asyncio.get_running_loop()
            image_text = await loop.run_in_executor(
                None, ocr.extract_image_text, image_bytes, image_attachment.content_type
            )

            if image_text:
                debug_log(f"OCR extracted text: {image_text[:50]}...")
                text_to_translate += f"\n\n**[Extracted Image Text]:**\n{image_text}"
            else:
                debug_log("OCR found no text in image.")

        if not text_to_translate.strip():
            debug_log("Nothing to translate (no content, tweet text, or image text).")
            return

        # =========================
        # CALL GEMINI API
        # =========================
        
        prompt = (
            f"Translate to {target_language}. "
            "Preserve tone, slang, and formatting. Output ONLY the translation.\n\n"
            f"Content: {text_to_translate}" # <--- Updated variable here
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
# X/TWITTER FETCHER
# =========================

async def extract_tweet_text(content: str) -> str:
    """
    Checks if the message contains a Twitter/X or alternative (vxtwitter, fixvx, etc.) URL. 
    If so, fetches the tweet text via the vxtwitter API and appends it to the content.
    """
    # Regex updated to catch x, twitter, vxtwitter, fxtwitter, fixupx, and fixvx
    # Group 1 captures the username, Group 2 captures the status ID
    twitter_pattern = re.compile(
        r'https?://(?:www\.)?(?:twitter\.com|x\.com|vxtwitter\.com|fxtwitter\.com|fixupx\.com|fixvx\.com)/([a-zA-Z0-9_]+)/status/([0-9]+)'
    )
    match = twitter_pattern.search(content)

    if not match:
        return content # No link found, return original content

    username = match.group(1)
    tweet_id = match.group(2)
    
    # Construct the API URL cleanly using the extracted username and ID
    api_url = f"https://api.vxtwitter.com/{username}/status/{tweet_id}"
    
    debug_log(f"Detected X/Twitter link. Fetching from {api_url}")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(api_url) as response:
                if response.status == 200:
                    data = await response.json()
                    tweet_text = data.get('text', '')
                    if tweet_text:
                        debug_log("Successfully fetched tweet content.")
                        # Return the original message plus the extracted tweet text
                        return f"{content}\n\n**[Extracted Tweet Text]:**\n{tweet_text}"
                else:
                    debug_log(f"Failed to fetch tweet. Status code: {response.status}")
    except Exception as e:
        debug_log(f"Error fetching tweet: {e}")

    # Fallback to original content if fetch fails
    return content

# =========================
# START BOT
# =========================

if __name__ == '__main__':
    print("Loading OCR models...")
    ocr.init_ocr()
    print("OCR models loaded.")
    client.run(DISCORD_TOKEN)
