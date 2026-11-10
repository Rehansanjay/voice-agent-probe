"""Rehearse the probe against a fake agent whose timing is known, before calling anyone.

The fake plays the telephony provider's side of the websocket and behaves like
a voice agent: greets 0.5 s after the stream starts, replies REPLY_DELAY after
our line ends (it notices the end after END_GAP of no audio from us), and when
talked over for BARGE_NOTICE it stops BARGE_STOP later. It streams silence
between utterances, as a real call does. The probe's numbers should land on
those, within the 32 ms detector window and a frame or two of pacing.

    python -m tests.rehearse                  # both providers
    python -m tests.rehearse --provider twilio --calls 2

It also runs a second, slower version of the fake agent through the same
runner and checks that probe.compare reports it as slower.

Places no calls and needs no accounts.
"""

import argparse
import asyncio
import audioop
import base64
import json
import tempfile
import time
from pathlib import Path

import websockets

import probe.call as pc
from probe import compare
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


async def fake_agent(port: int, wire: dict, clips: dict[str, bytes], log: dict, reply_delay: float = REPLY_DELAY):
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
        await silent(reply_delay)
        await speak(clips["answer"])
        await wait_for_our_line_end()
        await silent(reply_delay)
        log["long"] = await speak(clips["long"], interruptible=True)
        await silent(3.0)
        listener.cancel()
        await ws.send(json.dumps(wire["stop"]))


def fake_provider(name: str, port: int, clips: dict[str, bytes], log: dict, reply_delay: float):
    """The real provider class, except dialling starts the fake agent instead of a phone call."""
    base = PROVIDERS[name]

    class Fake(base):
        def mount(self, app, state):
            self.state = state  # no webhooks; the fake ends the call itself

        async def dial(self, to, host, label):
            async def agent():
                await fake_agent(port, WIRE[name], clips, log, reply_delay)
                await asyncio.sleep(1)
                self.state["result"].call_seconds = 0.0
                self.state["over"].set()

            self.task = asyncio.create_task(agent())
            # A real provider returns the same id its stream later carries.
            return {"enablex": "v1", "twilio": "CA1"}[name], "dial-out: fake agent"

        async def hangup(self, call_id):
            # The probe hangs up when its script ends; the fake just notes it.
            log.setdefault("hung_up", []).append(call_id)

    return Fake()


def check(name: str, result, clips: dict[str, bytes], log: dict) -> bool:
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
    good = result.bargein == "stopped" and result.bargein_stop_after_s is not None
    ok &= good
    print(f"{'OK ' if good else 'BAD'} barge-in: measured {result.bargein} after {result.bargein_stop_after_s}, "
          f"fake stopped {log.get('barge_true')} s after first hearing us (built in {BARGE_NOTICE + BARGE_STOP})")
    want_id = {"enablex": "v1", "twilio": "CA1"}[name]
    good = result.call_id == want_id and want_id in log.get("hung_up", [])
    ok &= good
    print(f"{'OK ' if good else 'BAD'} call id {result.call_id}, hung up {log.get('hung_up')}")
    print(f"     our media messages carried: {log.get('sent_shape')}")
    return ok


async def rehearse(name: str, port: int, calls: int, out: Path) -> bool:
    clips = {k: pc.read_wav(CLIPS / f"{k}.wav") for k in ("greeting", "answer", "long")}
    log = {}
    provider = fake_provider(name, port, clips, log, REPLY_DELAY)
    results = await pc.run_calls(provider, "x", "fake", name, calls=calls, gap_s=1, budget_s=40.0,
                                 port=port, out_dir=out, with_transcript=False)
    ok = len(results) == calls
    for r in results:
        ok &= check(name, r, clips, log)
    print(f"{name}: {calls} call(s)", "PASS" if ok else "FAIL", end="\n\n")
    return ok


async def rehearse_compare(name: str, port: int, out: Path) -> bool:
    """A second version of the fake that replies 1 s slower must show up as slower."""
    clips = {k: pc.read_wav(CLIPS / f"{k}.wav") for k in ("greeting", "answer", "long")}
    provider = fake_provider(name, port, clips, {}, REPLY_DELAY + 1.0)
    await pc.run_calls(provider, "x", "fake", "slower", calls=1, port=port, out_dir=out,
                       budget_s=40.0, with_transcript=False)
    sums = [compare.summarise(lbl, compare.load(lbl, out)) for lbl in (name, "slower")]
    text = compare.table(sums)
    print(text, end="\n\n")
    rows = {line.split("|")[1].strip(): line for line in text.splitlines() if line.startswith("|")}
    ok = all("slower" in rows[r] for r in ("Reply to a simple question", "Reply to a longer question"))
    ok &= "about the same" in rows["First words after answering"]
    print("compare:", "PASS (saw the 1 s slowdown, and no change in the greeting)" if ok else "FAIL")
    return ok


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--provider", choices=sorted(PROVIDERS))
    p.add_argument("--calls", type=int, default=1, help="calls per provider (default 1)")
    args = p.parse_args()
    names = [args.provider] if args.provider else sorted(PROVIDERS)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        results = [await rehearse(n, 8797 + i, args.calls, out) for i, n in enumerate(names)]
        results.append(await rehearse_compare(names[-1], 8797 + len(names), out))
    print("RESULT:", "PASS" if all(results) else "FAIL")
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    asyncio.run(main())
