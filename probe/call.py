"""Call a voice agent's phone number and measure how it behaves, the way a caller hears it.

    python -m probe.call --to +14155550100 --label acme-demo

Places an outbound call through Twilio or EnableX, streams the call audio over
a websocket, and runs a short scripted conversation against whatever answers:

1. greeting   how long until the agent says anything after it picks up
2. question   how long it takes to answer a simple question
3. barge-in   while it answers a second question, we talk over it, and
              measure whether and how fast it stops

Every time is measured on this machine from the audio as it arrives, so it
includes the phone network both ways: it is what a caller experiences.

Writes results/<when>_<label>.{json,md,wav}. The wav is stereo: the agent on
the left, our side on the right, lined up in time.

Needs the provider's credentials in the environment or a .env file (see
.env.example), and a running `ngrok http 8765` (or --host). With
DEEPGRAM_API_KEY set, the report also says what the agent said.
"""

import argparse
import asyncio
import audioop
import json
import os
import time
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path

import aiohttp
import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from probe.detector import RATE, SpeechDetector
from probe.providers import PROVIDERS, Media, Provider, Start, Stop

load_dotenv()

CHUNK = 160  # 20 ms of 8 kHz mu-law
LINES_DIR = Path(__file__).parent / "lines"

# What the probe says. The recordings are probe/lines/<name>.wav (8 kHz,
# 16-bit, mono); `python -m probe.make_lines` regenerates them from this text.
LINES = {
    "q1": "Hi, what can you help me with?",
    "q2": "Okay. Can you explain how that works, step by step?",
    "interrupt": "Sorry, can I stop you there for a second?",
    "hello": "Hello?",  # only when the agent called us and is waiting for us to speak
    # --script number: a number read in two chunks with a pause between, the way
    # people read numbers on the phone. The number is made up.
    "num_lead": "Hi, I'd like to check my booking. My mobile number is nine eight four zero one",
    "num_rest": "two three four five six.",
    "pin_lead": "And the pin code is six zero zero",
    "pin_rest": "zero four one.",
}
NUMBER_PAUSE_S = 1.0  # the gap between the two chunks; --pause changes it
QUIET_BEFORE_S = 1.5  # how long the agent must have been quiet before we start a number
HELLO_AFTER_S = 1.5  # how long a person would wait after picking up before saying hello


@dataclass
class Result:
    label: str
    to: str
    provider: str = ""
    direction: str = "outbound"  # inbound: the agent called us
    call_id: str | None = None
    answered: bool = False
    said_hello_first: bool = False
    greeting_after_s: float | None = None
    greeting_len_s: float | None = None
    q1_reply_after_s: float | None = None
    q2_reply_after_s: float | None = None
    bargein: str = "not tested"
    bargein_first_pause_s: float | None = None
    bargein_stop_after_s: float | None = None
    spoke_over_us: int = 0
    longest_turn_s: float | None = None
    call_seconds: float | None = None
    end_reason: str | None = None
    transcript: str | None = None
    script: str = "default"
    number_pause_s: float | None = None
    # --script number: one entry per number read in two chunks
    splits: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def read_wav(path: Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (1, 2, RATE):
            raise SystemExit(f"{path}: needs to be mono, 16-bit, {RATE} Hz")
        return w.readframes(w.getnframes())


def load_lines() -> dict[str, bytes]:
    return {name: trim_silence(read_wav(LINES_DIR / f"{name}.wav")) for name in LINES}


def trim_silence(pcm: bytes, keep_s: float = 0.03, level: float = 0.02) -> bytes:
    """Cut leading and trailing silence, so timing starts and stops with the voice."""
    a = np.frombuffer(pcm, dtype=np.int16)
    loud = np.nonzero(np.abs(a.astype(np.int32)) > level * 32768)[0]
    if not loud.size:
        return pcm
    keep = int(keep_s * RATE)
    return a[max(0, loud[0] - keep) : loud[-1] + keep].tobytes()


def speech_lead_s(pcm: bytes, level: float = 0.02) -> float:
    a = np.frombuffer(pcm, dtype=np.int16)
    loud = np.nonzero(np.abs(a.astype(np.int32)) > level * 32768)[0]
    return float(loud[0]) / RATE if loud.size else 0.0


async def transcribe(pcm: bytes) -> str | None:
    """What the agent said, from Deepgram's prerecorded API, if a key is set. Best effort."""
    if not pcm or not os.environ.get("DEEPGRAM_API_KEY"):
        return None
    # Raw PCM needs its encoding spelled out in the query; a Content-Type alone
    # is not enough for Deepgram, which then returns nothing.
    url = (
        "https://api.deepgram.com/v1/listen?model=nova-3&smart_format=true&language=multi"
        "&encoding=linear16&sample_rate=8000&channels=1"
    )
    headers = {
        "Authorization": f"Token {os.environ['DEEPGRAM_API_KEY']}",
        "Content-Type": "application/octet-stream",
    }
    try:
        async with aiohttp.ClientSession() as http:
            async with http.post(url, data=pcm, headers=headers) as r:
                data = await r.json(content_type=None)
        return data["results"]["channels"][0]["alternatives"][0]["transcript"]
    except Exception as e:
        return f"(transcription failed: {type(e).__name__})"


class Call:
    """One call's websocket session: receives the agent, plays our lines, keeps time."""

    def __init__(self, ws: WebSocket, lines: dict[str, bytes], provider: Provider):
        self.ws = ws
        self.lines = lines
        self.provider = provider
        self.det = SpeechDetector()
        self.call_id: str | None = None
        self.stream_id: str | None = None
        self.seq = 0
        self.t0 = time.monotonic()
        self.inbound: list[tuple[float, bytes]] = []
        self.outbound: list[tuple[float, bytes]] = []
        self.started = asyncio.Event()
        self.closed = asyncio.Event()

    def now(self) -> float:
        return time.monotonic() - self.t0

    async def receive(self):
        try:
            while True:
                ev = self.provider.parse(json.loads(await self.ws.receive_text()))
                if isinstance(ev, Start):
                    self.call_id, self.stream_id = ev.call_id, ev.stream_id
                    self.started.set()
                elif isinstance(ev, Media):
                    t = self.now()
                    pcm = audioop.ulaw2lin(ev.ulaw, 2)
                    self.inbound.append((t, pcm))
                    self.det.feed(pcm, t)
                elif isinstance(ev, Stop):
                    break
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            self.det.close(self.now())
            self.closed.set()

    async def say(self, name: str) -> tuple[float, float]:
        """Play a line in real time. Returns (start, end) in call seconds."""
        pcm = self.lines[name]
        ulaw = audioop.lin2ulaw(pcm, 2)
        start = self.now()
        next_t = time.monotonic()
        for i in range(0, len(ulaw), CHUNK):
            if self.closed.is_set():
                break
            self.seq += 1
            self.outbound.append((self.now(), pcm[i * 2 : (i + CHUNK) * 2]))
            msg = self.provider.media(ulaw[i : i + CHUNK], self.stream_id, self.call_id, self.seq)
            await self.ws.send_text(json.dumps(msg))
            next_t += CHUNK / RATE
            await asyncio.sleep(max(0.0, next_t - time.monotonic()))
        return start, self.now()

    async def wait_onset(self, after: float, timeout: float) -> float | None:
        deadline = self.now() + timeout
        while self.now() < deadline and not self.closed.is_set():
            t = self.det.first_onset_after(after)
            if t is not None:
                return t
            await asyncio.sleep(0.02)
        return self.det.first_onset_after(after)

    async def wait_quiet(self, timeout: float) -> bool:
        """Wait until the agent is not speaking. False on timeout."""
        deadline = self.now() + timeout
        while self.now() < deadline and not self.closed.is_set():
            if not self.det.speaking:
                return True
            await asyncio.sleep(0.02)
        return not self.det.speaking


async def run_script(call: Call, result: Result, budget_s: float, answering: bool = False) -> None:
    """The scripted conversation. Each step is skipped if the call has ended.

    answering: the agent placed this call and we picked up. Many agents then
    wait for the person to say hello, so if it stays quiet we do.
    """
    await asyncio.wait_for(call.started.wait(), 10)
    answered_at = call.now()
    result.answered = True
    result.call_id = result.call_id or call.call_id

    def left() -> float:
        return budget_s - (call.now() - answered_at)

    async def let_it_finish(what: str) -> None:
        # Never start a line over the agent: that measures our manners, not its.
        if not await call.wait_quiet(timeout=min(20.0, max(0.0, left() - 8))):
            result.notes.append(f"agent was still talking after 20 s ({what}); we spoke over it")

    # 1. Greeting: most agents speak first. Audio is timestamped by arrival, so
    # a greeting that starts with the stream can land a few ms before the
    # start event; look from just before it.
    timed_from = answered_at
    onset = await call.wait_onset(answered_at - 0.5, timeout=min(HELLO_AFTER_S if answering else 8.0, left()))
    if onset is None and answering:
        hello_start, timed_from = await call.say("hello")
        result.said_hello_first = True
        result.notes.append("agent waited for us to say hello; first words are timed from the end of our hello")
        onset = await call.wait_onset(hello_start, timeout=min(8.0, left()))
    if onset is None:
        result.notes.append("agent said nothing in the first 8 s; we spoke first")
    else:
        result.greeting_after_s = round(max(0.0, onset - timed_from), 2)
        await let_it_finish("greeting")
        span = call.det.spans[0]
        if span.end is not None:
            result.greeting_len_s = round(span.end - span.start, 2)

    # 2. A simple question.
    if left() > 10 and not call.closed.is_set():
        _, q1_end = await call.say("q1")
        onset = await call.wait_onset(q1_end, timeout=min(8.0, left()))
        if onset is None:
            result.notes.append("no reply to the first question within 8 s")
        else:
            result.q1_reply_after_s = round(onset - q1_end, 2)
            await let_it_finish("first answer")

    # 3. A question that needs a long answer, then talk over the answer.
    if left() > 10 and not call.closed.is_set():
        _, q2_end = await call.say("q2")
        onset = await call.wait_onset(q2_end, timeout=min(8.0, left()))
        if onset is None:
            result.notes.append("no reply to the second question within 8 s")
        else:
            result.q2_reply_after_s = round(onset - q2_end, 2)
            # Let it get 1.5 s into the answer, then interrupt.
            while call.now() < onset + 1.5 and not call.closed.is_set():
                await asyncio.sleep(0.02)
            if not call.det.speaking:
                result.bargein = "not tested (reply shorter than 1.5 s)"
            elif left() < 6:
                result.bargein = "not tested (call time ran out)"
            else:
                t_int = call.now()
                await call.say("interrupt")
                # Give it up to 4 s from the start of our interruption to stop.
                while call.now() < t_int + 4.0 and call.det.speaking and not call.closed.is_set():
                    await asyncio.sleep(0.02)
                pause = call.det.first_pause_after(t_int, t_int + 4.0)
                if pause is not None:
                    result.bargein_first_pause_s = round(max(0.0, pause - t_int), 2)
                end = call.det.speech_end_after(t_int)
                if end is None and call.det.speaking and call.det.last_speech_end < t_int + 3.7:
                    # Quiet by the deadline, just not long enough yet to close the span.
                    end = call.det.last_speech_end
                if end is not None and end <= t_int + 4.0:
                    result.bargein = "stopped"
                    result.bargein_stop_after_s = round(end - t_int, 2)
                else:
                    result.bargein = "kept talking for 4 s+"

    result.longest_turn_s = round(call.det.longest_span(), 2) or None

    # Count replies that started while we were mid-line, outside the barge-in test.
    our_spans = _spans_of(call.outbound)
    result.spoke_over_us = sum(
        1 for s in call.det.spans if any(a < s.start < b - 0.3 for a, b in our_spans)
    )


async def run_number_script(call: Call, result: Result, budget_s: float, pause_s: float = NUMBER_PAUSE_S) -> None:
    """Read a number in two chunks with a pause between, and see whether the agent
    jumps in during the pause (it thinks we're done) or waits for the rest."""
    await asyncio.wait_for(call.started.wait(), 10)
    answered_at = call.now()
    result.answered = True
    result.call_id = result.call_id or call.call_id
    result.script = "number"
    result.number_pause_s = pause_s

    def left() -> float:
        return budget_s - (call.now() - answered_at)

    async def let_it_finish(what: str) -> None:
        if not await call.wait_quiet(timeout=min(20.0, max(0.0, left() - 8))):
            result.notes.append(f"agent was still talking after 20 s ({what}); we spoke over it")

    onset = await call.wait_onset(answered_at - 0.5, timeout=min(8.0, left()))
    if onset is None:
        result.notes.append("agent said nothing in the first 8 s; we spoke first")
    else:
        result.greeting_after_s = round(max(0.0, onset - answered_at), 2)
        await let_it_finish("greeting")

    for what, lead, rest in (("mobile number", "num_lead", "num_rest"), ("pin code", "pin_lead", "pin_rest")):
        if left() < 12 or call.closed.is_set():
            break
        # Agents often pause under a second mid-reply ("let me check... I'm still
        # looking"). Starting then makes their next sentence look like a cut-in,
        # so wait for QUIET_BEFORE_S of real quiet first.
        await call.wait_quiet(timeout=min(20.0, max(0.0, left() - 12)))
        deadline = call.now() + min(10.0, max(0.0, left() - 12))
        while call.now() < deadline and not call.closed.is_set() and (
            call.det.speaking or call.now() - call.det.last_speech_end < QUIET_BEFORE_S
        ):
            await asyncio.sleep(0.02)
        lead_start, lead_end = await call.say(lead)
        # A person pausing mid-number: say nothing, but keep listening.
        while call.now() < lead_end + pause_s and not call.closed.is_set():
            await asyncio.sleep(0.01)
        rest_start, rest_end = await call.say(rest)
        split = {"what": what, "cut_in_after_s": None, "reply_after_s": None, "unclear": None}
        cut = call.det.first_onset_after(lead_end)
        # If the agent spoke at all while we read the first chunk, speech in the
        # pause may just be it finishing its own turn: not a clean test.
        talked_during = any(lead_start <= t0 < lead_end and sp for t0, sp in call.det.windows)
        if talked_during:
            split["unclear"] = "agent was talking while we read the first part"
            result.notes.append(f"{what}: not a clean test, the agent was talking while we read the first part")
        # Speech that starts in the pause, or so early in the second chunk that the
        # agent must have decided before hearing it, counts as cutting in.
        elif cut is not None and cut < rest_start + 0.2:
            split["cut_in_after_s"] = round(cut - lead_end, 2)
            result.notes.append(
                f"{what}: agent started talking {cut - lead_end:.2f} s into our {pause_s:.1f} s pause"
            )
        onset = await call.wait_onset(rest_end, timeout=min(8.0, left()))
        if onset is not None:
            split["reply_after_s"] = round(onset - rest_end, 2)
        result.splits.append(split)
        await let_it_finish(f"reply to the {what}")

    result.longest_turn_s = round(call.det.longest_span(), 2) or None


def _spans_of(chunks: list[tuple[float, bytes]], gap: float = 0.1) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    for t, pcm in chunks:
        end = t + len(pcm) / 2 / RATE
        if spans and t - spans[-1][1] <= gap:
            spans[-1] = (spans[-1][0], end)
        else:
            spans.append((t, end))
    return spans


def write_wav(path: Path, call: Call) -> None:
    total = max([t for t, _ in call.inbound + call.outbound] or [0]) + 1.0
    stereo = np.zeros((int(total * RATE) + RATE, 2), dtype=np.int16)
    for ch, chunks in ((0, call.inbound), (1, call.outbound)):
        for t, pcm in chunks:
            a = np.frombuffer(pcm, dtype=np.int16)
            i = int(t * RATE)
            stereo[i : i + len(a), ch] = a[: max(0, len(stereo) - i)]
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(stereo.tobytes())


def markdown(r: Result) -> str:
    def s(v):
        return "n/a" if v is None else f"{v:.2f} s"

    rows = [
        ("Answered the call", "yes" if r.answered else "no"),
        ("First words after answering", s(r.greeting_after_s)),
        ("Greeting length", s(r.greeting_len_s)),
        ("Reply to a simple question after", s(r.q1_reply_after_s)),
        ("Reply to a longer question after", s(r.q2_reply_after_s)),
        ("When talked over", r.bargein + (f", fully quiet after {r.bargein_stop_after_s:.2f} s" if r.bargein_stop_after_s is not None else "")),
        ("First pause after being talked over", s(r.bargein_first_pause_s)),
        ("Started talking while we were mid-sentence", str(r.spoke_over_us)),
        ("Longest time talking without a break", s(r.longest_turn_s)),
        ("Call length", s(r.call_seconds)),
    ]
    if r.script == "number":
        rows = [
            ("Answered the call", "yes" if r.answered else "no"),
            ("First words after answering", s(r.greeting_after_s)),
            ("Our pause in the middle of each number", s(r.number_pause_s)),
        ]
        for sp in r.splits:
            cut = sp["cut_in_after_s"]
            rows.append((f"{sp['what'].capitalize()}: cut in during our pause",
                         f"unclear: {sp['unclear']}" if sp.get("unclear")
                         else "no, waited" if cut is None else f"YES, {cut:.2f} s into the pause"))
            rows.append((f"{sp['what'].capitalize()}: reply after we finished", s(sp["reply_after_s"])))
        rows += [("Longest time talking without a break", s(r.longest_turn_s)), ("Call length", s(r.call_seconds))]
    way = "call from" if r.direction == "inbound" else "call to"
    out = [f"# {r.label}: {way} {r.to} via {r.provider}", "", "| | |", "|---|---|"]
    out += [f"| {k} | {v} |" for k, v in rows]
    out += ["", "Times are measured from the call audio as it arrived, so they include the phone network both ways: what a caller hears."]
    if r.notes:
        out += ["", "Notes:"] + [f"- {n}" for n in r.notes]
    if r.transcript:
        out += ["", "What the agent said:", "", f"> {r.transcript}"]
    out += ["", f"call id: {r.call_id}"]
    return "\n".join(out) + "\n"


def build_app(state: dict) -> FastAPI:
    app = FastAPI()
    provider: Provider = state["provider"]
    provider.mount(app, state)

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        call = Call(websocket, state["lines"], provider)
        state["call"] = call
        receiver = asyncio.create_task(call.receive())
        try:
            if state.get("script") == "number":
                await run_number_script(call, state["result"], state["budget_s"], state.get("pause_s", NUMBER_PAUSE_S))
            else:
                await run_script(call, state["result"], state["budget_s"], state.get("answering", False))
        except Exception as e:
            state["result"].notes.append(f"script stopped: {type(e).__name__}: {e}")
        finally:
            if call.call_id and not call.closed.is_set():
                await provider.hangup(call.call_id)
            await asyncio.wait([receiver], timeout=5)
            if state.get("answering"):
                # No per-call status webhook on a number's incoming calls, so
                # the stream ending is the end of the call.
                state["result"].call_seconds = round(call.now(), 2)
                state["over"].set()

    return app


async def ngrok_host() -> str:
    try:
        async with aiohttp.ClientSession() as http:
            async with http.get("http://127.0.0.1:4040/api/tunnels", timeout=3) as r:
                tunnels = (await r.json()).get("tunnels", [])
    except Exception:
        return ""
    return next((t["public_url"] for t in tunnels if t.get("public_url", "").startswith("https://")), "")


async def _serve(state: dict, port: int) -> tuple[uvicorn.Server, asyncio.Task]:
    server = uvicorn.Server(uvicorn.Config(build_app(state), host="0.0.0.0", port=port, log_level="warning"))
    serving = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)
    return server, serving


async def _save(state: dict, result: Result, out_dir: Path, label: str, run: str, with_transcript: bool) -> None:
    stem = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_{label}{run}"
    call: Call | None = state["call"]
    if call is not None:
        if with_transcript:
            result.transcript = await transcribe(b"".join(pcm for _, pcm in call.inbound))
        write_wav(stem.with_suffix(".wav"), call)
    stem.with_suffix(".json").write_text(
        json.dumps({**asdict(result), "events": state["events"]}, indent=2), encoding="utf-8"
    )
    md = markdown(result)
    stem.with_suffix(".md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"saved {stem}.{{json,md,wav}}")


async def answer_calls(
    provider: Provider, host: str, label: str, *, calls: int = 1, budget_s: float | None = None,
    port: int = 8765, out_dir: Path = Path("results"), with_transcript: bool = True, wait_s: float = 600.0,
) -> list[Result]:
    """Wait for ``calls`` agents to phone us, run the script on each, save a report for each."""
    def fresh() -> Result:
        return Result(label=label, to="unknown", provider=provider.name, direction="inbound")

    state = {
        "host": host, "lines": load_lines(), "provider": provider, "budget_s": budget_s or provider.budget_s,
        "answering": True, "busy": False,
        "result": fresh(), "events": [], "over": asyncio.Event(), "t0": time.monotonic(), "call": None,
    }
    server, serving = await _serve(state, port)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    try:
        for i in range(calls):
            if i:
                state.update(result=fresh(), events=[], over=asyncio.Event(), t0=time.monotonic(),
                             call=None, busy=False)
            print(f"[{i + 1}/{calls}] waiting for the call, up to {wait_s:.0f} s")
            try:
                await asyncio.wait_for(state["over"].wait(), timeout=wait_s + state["budget_s"])
            except TimeoutError:
                print("no call came in")
                break
            await _save(state, state["result"], out_dir, label, f"_{i + 1}" if calls > 1 else "", with_transcript)
            results.append(state["result"])
    finally:
        server.should_exit = True
        await serving
    return results


async def run_calls(
    provider: Provider, host: str, to: str, label: str, *, calls: int = 1, gap_s: float = 10.0,
    budget_s: float | None = None, port: int = 8765, out_dir: Path = Path("results"),
    with_transcript: bool = True, script: str = "default", pause_s: float = NUMBER_PAUSE_S,
) -> list[Result]:
    """Place ``calls`` calls one after another, saving a report for each."""
    state = {
        "host": host, "lines": load_lines(), "provider": provider, "script": script, "pause_s": pause_s,
        "budget_s": budget_s or provider.budget_s,
        "result": None, "events": [], "over": asyncio.Event(), "t0": time.monotonic(), "call": None,
    }
    server, serving = await _serve(state, port)

    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    try:
        for i in range(calls):
            if i:
                print(f"\nnext call in {gap_s:.0f} s")
                await asyncio.sleep(gap_s)
            result = Result(label=label, to=to, provider=provider.name)
            # Webhooks and the websocket read these at request time, so a fresh
            # set per call keeps one call's late events out of the next.
            state.update(result=result, events=[], over=asyncio.Event(), t0=time.monotonic(), call=None)

            call_id, log = await provider.dial(to, host, label)
            print(f"[{i + 1}/{calls}] {log}")
            if call_id:
                result.call_id = call_id
                try:
                    await asyncio.wait_for(state["over"].wait(), timeout=state["budget_s"] + 50)
                except TimeoutError:
                    result.notes.append("the provider never reported the call ending")
                await asyncio.sleep(1)

            await _save(state, result, out_dir, label, f"_{i + 1}" if calls > 1 else "", with_transcript)
            results.append(result)
    finally:
        server.should_exit = True
        await serving
    return results


async def main():
    p = argparse.ArgumentParser(description="Call a voice agent and measure it.")
    p.add_argument("--to", required=True, help="number to call with country code, e.g. +14155550100")
    p.add_argument("--label", required=True,
                   help="name for this agent version, e.g. v1; compare two with `python -m probe.compare v1 v2`")
    p.add_argument("--calls", type=int, default=1, help="how many calls to make, one after another (default 1)")
    p.add_argument("--gap", type=float, default=10.0, help="seconds between calls (default 10)")
    p.add_argument("--provider", choices=sorted(PROVIDERS), default=os.environ.get("PROBE_PROVIDER", "twilio"),
                   help="who places the call (default: $PROBE_PROVIDER, else twilio)")
    p.add_argument("--budget", type=float, help="seconds the conversation may run (default: per provider)")
    p.add_argument("--host", help="public host forwarding to this machine; default: the ngrok tunnel")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--script", choices=["default", "number"], default="default",
                   help="default: greeting, questions, barge-in. number: read a mobile number and a pin code "
                        "in two chunks each and see if the agent cuts in during the pause")
    p.add_argument("--pause", type=float, default=NUMBER_PAUSE_S,
                   help=f"--script number: seconds of pause in the middle of each number (default {NUMBER_PAUSE_S})")
    args = p.parse_args()
    if args.calls < 1:
        raise SystemExit("--calls must be at least 1")

    provider = PROVIDERS[args.provider]()
    provider.check_env()
    host = (args.host or await ngrok_host()).removeprefix("https://").rstrip("/")
    if not host:
        raise SystemExit("No public host. Start `ngrok http 8765`, or pass --host.")

    await run_calls(provider, host, args.to, args.label, calls=args.calls, gap_s=args.gap,
                    budget_s=args.budget, port=args.port, script=args.script, pause_s=args.pause)
    if args.calls > 1:
        print(f"\nsummary: python -m probe.compare {args.label}")


if __name__ == "__main__":
    asyncio.run(main())
