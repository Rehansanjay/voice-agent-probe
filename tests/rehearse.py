"""Rehearse the probe against a fake agent whose timing is known, before calling anyone.

The fake plays the telephony provider's side of the websocket and behaves like
a voice agent: greets 0.5 s after the stream starts, replies REPLY_DELAY after
our line ends (it notices the end after END_GAP of no audio from us), and when
talked over for BARGE_NOTICE it stops BARGE_STOP later. It streams silence
between utterances, as a real call does. The probe's numbers should land on
those, within the 32 ms detector window and a frame or two of pacing.

    python -m tests.rehearse                  # both providers
    python -m tests.rehearse --provider twilio

Places no calls and needs no accounts.
"""

import argparse
import asyncio
import audioop
import base64
import json
import time
from pathlib import Path

import uvicorn
import websockets

import probe.call as pc
from probe.providers import PROVIDERS

GREET_DELAY = 0.5
END_GAP = 0.3
REPLY_DELAY = 1.0
BARGE_NOTICE = 0.4
BARGE_STOP = 0.5

CLIPS = Path(__file__).parent / "clips"


# What each provider sends the probe: the agent's side of the wire.
WIRE = {
    "enablex": {
        "hello": [{"event": "connected"},
                  {"event": "start_media", "stream_id": "s1", "start": {"stream_id": "s1", "voice_id": "v1"}}],
        "media": lambda b64: {"event": "media", "media": {"payload": b64}},
        "stop": {"event": "stop_media"},
    },
    "twilio": {
        "hello": [{"event": "connected", "protocol": "Call", "version": "1.0.0"},
                  {"event": "start", "sequenceNumber": "1", "streamSid": "MZ1",
                   "start": {"streamSid": "MZ1", "callSid": "CA1", "tracks": ["inbound"],
                             "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1}}}],
        "media": lambda b64: {"event": "media", "streamSid": "MZ1",
                              "media": {"track": "inbound", "payload": b64}},
        "stop": {"event": "stop", "streamSid": "MZ1"},
    },
}


async def fake_agent(port: int, wire: dict, clips: dict[str, bytes], log: dict):
    async with websockets.connect(f"ws://127.0.0.1:{port}/ws") as ws:
        for msg in wire["hello"]:
            await ws.send(json.dumps(msg))
        last_heard = [0.0]
        heard_since = [None]

        async def listen():
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("event") == "media":
                    log["sent_shape"] = sorted(msg)
                    now = time.monotonic()
                    if heard_since[0] is None or now - last_heard[0] > END_GAP:
                        heard_since[0] = now
                    last_heard[0] = now

        listener = asyncio.create_task(listen())
        silence = audioop.lin2ulaw(b"\x00\x00" * 160, 2)

        async def send(ulaw: bytes):
            await ws.send(json.dumps(wire["media"](base64.b64encode(ulaw).decode())))

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
        await ws.send(json.dumps(wire["stop"]))


async def rehearse(name: str, port: int) -> bool:
    clips = {k: pc.read_wav(CLIPS / f"{k}.wav") for k in ("greeting", "answer", "long")}
    provider = PROVIDERS[name]()

    log = {}

    async def no_dial(*a, **k):
        raise AssertionError("the rehearsal must not reach a real provider")

    async def fake_hangup(call_id):
        # The probe hangs up when its script ends; the fake just notes it.
        log["hung_up"] = call_id

    provider.dial, provider.hangup = no_dial, fake_hangup
    result = pc.Result(label="rehearsal", to="fake", provider=name)
    state = {"host": "x", "lines": pc.load_lines(), "result": result, "events": [], "provider": provider,
             "budget_s": 40.0, "over": asyncio.Event(), "t0": time.monotonic(), "call": None}
    server = uvicorn.Server(uvicorn.Config(pc.build_app(state), host="127.0.0.1", port=port, log_level="warning"))
    serving = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)
    await fake_agent(port, WIRE[name], clips, log)
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
    want_id = {"enablex": "v1", "twilio": "CA1"}[name]
    good = log.get("hung_up") == want_id and result.call_id == want_id
    ok &= good
    print(f"{'OK ' if good else 'BAD'} call id {result.call_id}, hung up {log.get('hung_up')}")
    print(f"     our media messages carried: {log.get('sent_shape')}")
    print(f"{name}:", "PASS" if ok else "FAIL", "\n")
    return ok


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--provider", choices=sorted(PROVIDERS))
    args = p.parse_args()
    names = [args.provider] if args.provider else sorted(PROVIDERS)
    results = [await rehearse(n, 8797 + i) for i, n in enumerate(names)]
    print("RESULT:", "PASS" if all(results) else "FAIL")
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    asyncio.run(main())
