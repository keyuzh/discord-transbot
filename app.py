"""Discord translation bot backed by OpenRouter."""

import base64
import os

from openrouter import OpenRouter

import bot  # imports first so load_dotenv() runs before we read the key

API_KEY = bot.require_env("OPENROUTER_API_KEY")
MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")

# One client for the whole process; its httpx pool is reused across requests.
client = OpenRouter(api_key=API_KEY)


async def translate(
    content: str, target_language: str, images: list[tuple[bytes, str]]
) -> tuple[str, str]:
    user_content = []
    if content:
        user_content.append({"type": "text", "text": content})
    for data, mime in images:
        data_uri = f"data:{mime};base64,{base64.b64encode(data).decode()}"
        user_content.append({"type": "image_url", "image_url": {"url": data_uri}})

    response = await client.chat.send_async(
        model=MODEL,
        messages=[
            {"role": "system", "content": bot.build_prompt(target_language)},
            {"role": "user", "content": user_content},
        ],
        temperature=0.3,
    )
    text = response.choices[0].message.content if response.choices else None
    return (text or ""), response.model


if __name__ == "__main__":
    bot.run(translate, ready_message="Translation bot (OpenRouter) is online.")
