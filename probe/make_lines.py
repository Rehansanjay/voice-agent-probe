"""Regenerate probe/lines/*.wav from the text in probe.call.LINES, with Deepgram TTS.

    python -m probe.make_lines

Only needed if you change what the probe says. Any 8 kHz, 16-bit mono wav
works too, including your own recorded voice.
"""

import asyncio
import os
import wave

import aiohttp
from dotenv import load_dotenv

from probe.call import LINES, LINES_DIR
from probe.detector import RATE


async def speak(text: str) -> bytes:
    url = (
        "https://api.deepgram.com/v1/speak?model=aura-2-thalia-en"
        f"&encoding=linear16&sample_rate={RATE}&container=none"
    )
    headers = {"Authorization": f"Token {os.environ['DEEPGRAM_API_KEY']}"}
    async with aiohttp.ClientSession() as http:
        async with http.post(url, json={"text": text}, headers=headers) as r:
            r.raise_for_status()
            return await r.read()


async def main():
    load_dotenv()
    if not os.environ.get("DEEPGRAM_API_KEY"):
        raise SystemExit("Set DEEPGRAM_API_KEY to regenerate the lines.")
    LINES_DIR.mkdir(exist_ok=True)
    for name, text in LINES.items():
        pcm = await speak(text)
        with wave.open(str(LINES_DIR / f"{name}.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(pcm)
        print(f"{name}.wav  {len(pcm) / 2 / RATE:.2f} s  {text!r}")


if __name__ == "__main__":
    asyncio.run(main())
