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
runner and checks that probe.compare reports it as slower, and rehearses
probe.answer: an agent that phones us and speaks first, and one that waits
for a hello.

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

import aiohttp
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


async def fake_agent(port: int, wire: dict, clips: dict[str, bytes], log: dict, reply_delay: float = REPLY_DELAY,
                     waits_for_hello: bool = False, number: bool = False, end_gap: float = END_GAP):
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
                if last_heard[0] > seen and time.monotonic() - last_heard[0] >= end_gap:
                    return

        if number:  # --script number: greet, then answer whenever we've been quiet for end_gap
            await silent(GREET_DELAY)
            await speak(clips["greeting"])
            for _ in range(4):
                await wait_for_our_line_end()
                await silent(reply_delay)
                await speak(clips["answer"])
            listener.cancel()
            await ws.send(json.dumps(wire["stop"]))
            return

        if waits_for_hello:  # an agent that placed the call and waits for the person to speak
            await wait_for_our_line_end()
            await silent(reply_delay)
        else:
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


async def post(port: int, path: str, form: dict[str, str]) -> tuple[int, str]:
    async with aiohttp.ClientSession() as http:
        async with http.post(f"http://127.0.0.1:{port}{path}", data=form) as r:
            return r.status, await r.text()


def fake_provider(name: str, port: int, clips: dict[str, bytes], log: dict, reply_delay: float, **agent_kw):
    """The real provider class, except dialling starts the fake agent instead of a phone call."""
    base = PROVIDERS[name]

    class Fake(base):
        def mount(self, app, state):
            self.state = state
            super().mount(app, state)  # the real webhooks, so they get exercised too

        async def dial(self, to, host, label):
            async def agent():
                await fake_agent(port, WIRE[name], clips, log, reply_delay, **agent_kw)
                await asyncio.sleep(1)
                if name == "twilio":  # the real status webhook has to end the call
                    await post(port, "/twilio/status", {"CallStatus": "completed", "CallDuration": "12"})
                else:
                    self.state["result"].call_seconds = 0.0
                    self.state["over"].set()

            self.task = asyncio.create_task(agent())
            # A real provider returns the same id its stream later carries.
            return {"enablex": "v1", "twilio": "CA1"}[name], "dial-out: fake agent"

        async def hangup(self, call_id):
            # The probe hangs up when its script ends; the fake just notes it.
            log.setdefault("hung_up", []).append(call_id)

    return Fake()


def check(name: str, result, clips: dict[str, bytes], log: dict, greet_delay: float = GREET_DELAY) -> bool:
    expect = {
        "greeting_after_s": greet_delay + pc.speech_lead_s(clips["greeting"]),
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
    good = good and result.bargein_first_pause_s is not None and result.bargein_first_pause_s <= result.bargein_stop_after_s + 0.3
    ok &= good
    print(f"     first pause after being talked over: {result.bargein_first_pause_s}")
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
        if name == "twilio":
            good = r.call_seconds == 12.0 and r.end_reason == "completed"
            ok &= good
            print(f"{'OK ' if good else 'BAD'} status webhook: {r.call_seconds} s, {r.end_reason}")
    print(f"{name}: {calls} call(s)", "PASS" if ok else "FAIL", end="\n\n")
    return ok


async def rehearse_answer(port: int, out: Path, waits_for_hello: bool) -> bool:
    """The agent phones us: Twilio's real incoming webhook, then the same script."""
    what = "waits for hello" if waits_for_hello else "speaks first"
    clips = {k: pc.read_wav(CLIPS / f"{k}.wav") for k in ("greeting", "answer", "long")}
    log = {}
    provider = fake_provider("twilio", port, clips, log, REPLY_DELAY)
    label = "hello" if waits_for_hello else "first"
    task = asyncio.create_task(pc.answer_calls(provider, "x", label, port=port, out_dir=out, budget_s=40.0,
                                               with_transcript=False, wait_s=30))
    caller = {"From": "+919900000000", "CallSid": "CA1"}
    while True:
        try:
            status, twiml = await post(port, "/twilio/incoming", caller)
            break
        except aiohttp.ClientConnectorError:
            await asyncio.sleep(0.05)
    good = status == 200 and 'Stream url="wss://x/ws"' in twiml
    ok = good
    print(f"{'OK ' if good else 'BAD'} incoming webhook: HTTP {status} {twiml}")
    status, second = await post(port, "/twilio/incoming", caller)
    good = "Reject" in second
    ok &= good
    print(f"{'OK ' if good else 'BAD'} a second caller while busy: {second}")

    await fake_agent(port, WIRE["twilio"], clips, log, waits_for_hello=waits_for_hello)
    results = await task
    ok &= len(results) == 1
    r = results[0]
    ok &= check("twilio", r, clips, log, END_GAP + REPLY_DELAY if waits_for_hello else GREET_DELAY)
    good = r.direction == "inbound" and r.to == caller["From"] and r.said_hello_first == waits_for_hello
    ok &= good
    print(f"{'OK ' if good else 'BAD'} {r.direction} from {r.to}, said hello first: {r.said_hello_first}")
    print(f"answer ({what}):", "PASS" if ok else "FAIL", end="\n\n")
    return ok


async def rehearse_number(name: str, port: int, out: Path, end_gap: float) -> bool:
    """--script number against an agent that ends our turn after end_gap of silence.

    Our pause is 1.0 s, so an agent with a 0.3 s gap must be caught cutting in
    (end_gap + reply delay + clip lead into the pause), and one with 1.5 s must not.
    """
    clips = {k: pc.read_wav(CLIPS / f"{k}.wav") for k in ("greeting", "answer", "long")}
    reply_delay = 0.1
    provider = fake_provider(name, port, clips, {}, reply_delay, number=True, end_gap=end_gap)
    [r] = await pc.run_calls(provider, "x", "fake", f"num{end_gap}", calls=1, port=port, out_dir=out,
                             budget_s=60.0, with_transcript=False, script="number", pause_s=1.0)
    lead = pc.speech_lead_s(clips["answer"])
    want = end_gap + reply_delay + lead
    should_cut = want < 1.0
    ok = len(r.splits) == 2
    print(f"{'OK ' if ok else 'BAD'} both numbers tested: {[s['what'] for s in r.splits]}")
    for s in r.splits:
        cut = s["cut_in_after_s"]
        if should_cut:
            # 0.2 s: the fake's own pacing runs ~0.1 s late (the q1/q2 checks show it
            # too), and the claim this feeds is yes/no: did it start inside the pause.
            good = cut is not None and abs(cut - want) <= 0.2 and cut < 1.0
            print(f"{'OK ' if good else 'BAD'} {s['what']}: cut in {cut} s into the pause, built in {want:.2f}")
        else:
            good = cut is None and s["reply_after_s"] is not None and abs(s["reply_after_s"] - want) <= 0.15
            print(f"{'OK ' if good else 'BAD'} {s['what']}: waited, replied {s['reply_after_s']} s after, built in {want:.2f}")
        ok &= good
    print(f"number script, agent gap {end_gap} s:", "PASS" if ok else "FAIL", end="\n\n")
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
        results.append(await rehearse_number(names[0], 8790, out, end_gap=0.3))
        results.append(await rehearse_number(names[0], 8791, out, end_gap=1.5))
        if "twilio" in names:
            results.append(await rehearse_answer(8797 + len(names) + 1, out, waits_for_hello=False))
            results.append(await rehearse_answer(8797 + len(names) + 2, out, waits_for_hello=True))
    print("RESULT:", "PASS" if all(results) else "FAIL")
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    asyncio.run(main())
