"""Rehearse the probe against a fake agent whose timing is known, before calling anyone.

The fake plays EnableX's side of the websocket and behaves like a voice agent:
greets 0.5 s after the stream starts, replies REPLY_DELAY after our line ends
(it notices the end after END_GAP of no audio from us), and when talked over
for BARGE_NOTICE it stops BARGE_STOP later. It streams silence between
utterances, as a real call does. The probe's numbers should land on those,
within the 32 ms detector window and a frame or two of pacing.

    python -m tests.rehearse
"""

import asyncio
import audioop
import base64
import json
import time
from pathlib import Path

import uvicorn
import websockets

import probe.call as pc

GREET_DELAY = 0.5
END_GAP = 0.3
REPLY_DELAY = 1.0
BARGE_NOTICE = 0.4
BARGE_STOP = 0.5

GREETING = "Hello, thanks for calling Acme. How can I help you today?"
ANSWER = "I can help you book appointments and answer questions about our services."
LONG = (
    "Sure. First you tell me what you need, then I check the calendar for open slots, "
    "then I read the options back to you, and finally I confirm the booking and send a text message."
)


async def fake_agent(port: int, clips: dict[str, bytes], log: dict):
    async with websockets.connect(f"ws://127.0.0.1:{port}/ws") as ws:
        await ws.send(json.dumps({"event": "connected"}))
        await ws.send(json.dumps({"event": "start_media", "stream_id": "s1",
                                  "start": {"stream_id": "s1", "voice_id": "v1"}}))
        t0 = time.monotonic()
        last_heard = [0.0]
        heard_since = [None]

        async def listen():
            async for raw in ws:
                if json.loads(raw).get("event") == "media":
                    now = time.monotonic()
                    if heard_since[0] is None or now - last_heard[0] > END_GAP:
                        heard_since[0] = now
                    last_heard[0] = now

        listener = asyncio.create_task(listen())
        silence = audioop.lin2ulaw(b"\x00\x00" * 160, 2)

        async def send(ulaw: bytes):
            await ws.send(json.dumps({"event": "media", "media": {"payload": base64.b64encode(ulaw).decode()}}))

        async def speak(pcm: bytes, interruptible=False) -> str:
            ulaw = audioop.lin2ulaw(pcm, 2)
            start = time.monotonic()
            nxt = start
            for i in range(0, len(ulaw), 160):
                now = time.monotonic()
                if interruptible and heard_since[0] and heard_since[0] > start and now - heard_since[0] >= BARGE_NOTICE:
                    stop_at = heard_since[0] + BARGE_NOTICE + BARGE_STOP
                    while time.monotonic() < stop_at:
                        await send(ulaw[i:i + 160]); i += 160
                        nxt += 0.02; await asyncio.sleep(max(0, nxt - time.monotonic()))
                    log["barge_true"] = round(time.monotonic() - heard_since[0], 2)
                    return "interrupted"
                await send(ulaw[i:i + 160])
                nxt += 0.02
                await asyncio.sleep(max(0, nxt - time.monotonic()))
            return "done"

        clock = [time.monotonic()]

        async def tick():
            clock[0] = max(clock[0] + 0.02, time.monotonic() - 0.2)
            await asyncio.sleep(max(0, clock[0] - time.monotonic()))

        async def silent(seconds: float):
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                await send(silence); await tick()

        async def wait_for_our_line_end():
            seen = last_heard[0]
            while True:
                await send(silence); await tick()
                if last_heard[0] > seen and time.monotonic() - last_heard[0] >= END_GAP:
                    return

        await silent(GREET_DELAY)
        await speak(clips["greeting"])
        await wait_for_our_line_end()
        await silent(REPLY_DELAY)
        await speak(clips["answer"])
        await wait_for_our_line_end()
        await silent(REPLY_DELAY)
        log["long"] = await speak(clips["long"], interruptible=True)
        await silent(3.0)
        listener.cancel()
        await ws.send(json.dumps({"event": "stop_media"}))


async def main():
    cache = Path("results/.tts")
    lines = {k: pc.trim_silence(await pc.tts_pcm(v, cache)) for k, v in pc.LINES.items()}
    clips = {"greeting": await pc.tts_pcm(GREETING, cache), "answer": await pc.tts_pcm(ANSWER, cache),
             "long": await pc.tts_pcm(LONG, cache)}

    async def no_api(*a, **k):
        return 200, {}

    pc.enablex = no_api  # nothing here may reach the real EnableX API
    result = pc.Result(label="rehearsal", to="fake")
    state = {"host": "x", "lines": lines, "result": result, "events": [], "over": asyncio.Event(),
             "t0": time.monotonic(), "call": None}
    port = 8797
    server = uvicorn.Server(uvicorn.Config(pc.build_app(state), host="127.0.0.1", port=port, log_level="warning"))
    serving = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)
    log = {}
    await fake_agent(port, clips, log)
    await asyncio.sleep(2)
    server.should_exit = True
    await serving

    print(pc.markdown(result))
    expect = {
        "greeting_after_s": GREET_DELAY + pc.speech_lead_s(clips["greeting"]),
        "q1_reply_after_s": END_GAP + REPLY_DELAY + pc.speech_lead_s(clips["answer"]),
        "q2_reply_after_s": END_GAP + REPLY_DELAY + pc.speech_lead_s(clips["long"]),
    }
    ok = True
    for k, want in expect.items():
        got = getattr(result, k)
        good = got is not None and abs(got - want) <= 0.15
        ok &= good
        print(f"{'OK ' if good else 'BAD'} {k}: measured {got}, built in {want:.2f}")
    good = result.bargein == "stopped" and result.bargein_stop_after_s is not None and log.get("long") == "interrupted"
    ok &= good
    print(f"{'OK ' if good else 'BAD'} barge-in: measured {result.bargein} after {result.bargein_stop_after_s}, "
          f"fake stopped {log.get('barge_true')} s after first hearing us (built in {BARGE_NOTICE + BARGE_STOP})")
    print("RESULT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    asyncio.run(main())
