import os
import discord
from openai import OpenAI
from dotenv import load_dotenv

# =========================
# LOAD ENV
# =========================

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# =========================
# DEEPSEEK CLIENT
# =========================

ai = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
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
    "🇯🇵": "Japanese",
    "🇨🇳": "Chinese",
    "🇰🇷": "Korean",
    "🇫🇷": "French",
    "🇪🇸": "Spanish",
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

    # only process flag emojis
    if emoji not in LANGUAGES:
        return

    cache_key = (payload.message_id, emoji)

    # prevent duplicate translations
    if cache_key in translated_cache:
        return

    translated_cache.add(cache_key)

    try:

        # get channel
        channel = client.get_channel(payload.channel_id)

        if channel is None:
            return

        # fetch original message
        message = await channel.fetch_message(payload.message_id)

        # ignore bot messages
        if message.author.bot:
            return

        # ignore empty messages
        if not message.content:
            return

        target_language = LANGUAGES[emoji]

        print(f"Translating to {target_language}")

        # =========================
        # CALL DEEPSEEK API
        # =========================

        response = ai.chat.completions.create(
            model="deepseek-chat",

            messages=[
                {
                    "role": "system",
                    "content":
                    (
                        "You are an expert translator. "
                        "Preserve tone, slang, emojis, formatting, "
                        "and emotional nuance. "
                        "Only output translated text."
                    )
                },

                {
                    "role": "user",
                    "content":
                    (
                        f"Translate this into {target_language}:\n\n"
                        f"{message.content}"
                    )
                }
            ],

            temperature=0.3
        )

        translated = response.choices[0].message.content

        # =========================
        # EMBED
        # =========================

        embed = discord.Embed(
            title=f"{emoji} Translation",
            description=translated
        )

        embed.add_field(
            name="Original",
            value=message.content,
            inline=False
        )

        # reply to original message
        await message.reply(embed=embed)

    except Exception as e:

        print("Translation error:")
        print(e)

# =========================
# START BOT
# =========================

client.run(DISCORD_TOKEN)
