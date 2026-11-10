"""Call a voice agent's phone number and measure how it behaves, the way a caller hears it.

    python -m probe.call --to 91XXXXXXXXXX --label acme-demo

Places an outbound call through EnableX, streams the call audio over a
websocket, and runs a short scripted conversation against whatever answers:

1. greeting   how long until the agent says anything after it picks up
2. question   how long it takes to answer a simple question
3. barge-in   while it answers a second question, we talk over it, and
              measure whether and how fast it stops

Every time is measured on this machine from the audio as it arrives, so it
includes the phone network both ways: it is what a caller experiences.

Writes results/<when>_<label>.{json,md,wav}. The wav is stereo: the agent on
the left, our side on the right, lined up in time.

Needs ENABLEX_APP_ID, ENABLEX_APP_KEY, ENABLEX_FROM_NUMBER and
DEEPGRAM_API_KEY in the environment or a .env file, and a running
`ngrok http 8765` (or --host).
"""

import argparse
import asyncio
import audioop
import base64
import hashlib
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
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect

from probe.detector import RATE, SpeechDetector

load_dotenv()

API = "https://api.enablex.io/voice/v1/call"
CHUNK = 160  # 20 ms of 8 kHz mu-law
CALL_BUDGET_S = 40.0  # EnableX trial calls are cut at 45 s

LINES = {
    "q1": "Hi, what can you help me with?",
    "q2": "Okay. Can you explain how that works, step by step?",
    "interrupt": "Sorry, can I stop you there for a second?",
}


@dataclass
class Result:
    label: str
    to: str
    voice_id: str | None = None
    answered: bool = False
    greeting_after_s: float | None = None
    greeting_len_s: float | None = None
    q1_reply_after_s: float | None = None
    q2_reply_after_s: float | None = None
    bargein: str = "not tested"
    bargein_stop_after_s: float | None = None
    spoke_over_us: int = 0
    call_seconds: float | None = None
    end_reason: str | None = None
    transcript: str | None = None
    notes: list[str] = field(default_factory=list)


def auth_headers() -> dict:
    token = base64.b64encode(
        f"{os.environ['ENABLEX_APP_ID']}:{os.environ['ENABLEX_APP_KEY']}".encode()
    ).decode()
    return {"Authorization": f"Basic {token}", "Content-Type": "application/json"}


async def enablex(method: str, path: str, body: dict | None = None):
    async with aiohttp.ClientSession() as http:
        async with http.request(method, f"{API}{path}", json=body, headers=auth_headers()) as r:
            try:
                return r.status, await r.json(content_type=None)
            except Exception:
                return r.status, await r.text()


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


async def tts_pcm(text: str, cache: Path) -> bytes:
    """8 kHz 16-bit mono PCM for a line, from Deepgram, cached on disk."""
    # hashlib, not hash(): str hashes change every run, which re-fetched every line.
    path = cache / f"{hashlib.sha1(text.encode()).hexdigest()[:16]}.pcm"
    if path.exists():
        return path.read_bytes()
    url = (
        "https://api.deepgram.com/v1/speak?model=aura-2-thalia-en"
        "&encoding=linear16&sample_rate=8000&container=none"
    )
    headers = {"Authorization": f"Token {os.environ['DEEPGRAM_API_KEY']}"}
    async with aiohttp.ClientSession() as http:
        async with http.post(url, json={"text": text}, headers=headers) as r:
            r.raise_for_status()
            pcm = await r.read()
    cache.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pcm)
    return pcm


async def transcribe(pcm: bytes) -> str | None:
    """What the agent said, from Deepgram's prerecorded API. Best effort."""
    if not pcm:
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

    def __init__(self, ws: WebSocket, lines: dict[str, bytes]):
        self.ws = ws
        self.lines = lines
        self.det = SpeechDetector()
        self.voice_id: str | None = None
        self.stream_id: str | None = None
        self.seq = 0
        self.t0 = time.monotonic()
        self.inbound: list[tuple[float, bytes]] = []
        self.outbound: list[tuple[float, bytes]] = []
        self.started = asyncio.Event()
        self.closed = asyncio.Event()
        self.playing = False

    def now(self) -> float:
        return time.monotonic() - self.t0

    async def receive(self):
        try:
            while True:
                msg = json.loads(await self.ws.receive_text())
                event = msg.get("event")
                if event == "start_media":
                    self.stream_id = msg.get("stream_id")
                    self.voice_id = (msg.get("start") or {}).get("voice_id")
                    self.started.set()
                elif event == "media":
                    payload = (msg.get("media") or {}).get("payload")
                    if payload:
                        t = self.now()
                        pcm = audioop.ulaw2lin(base64.b64decode(payload), 2)
                        self.inbound.append((t, pcm))
                        self.det.feed(pcm, t)
                elif event == "stop_media":
                    break
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            self.det.close(self.now())
            self.closed.set()

    async def say(self, name: str, stop: asyncio.Event | None = None) -> tuple[float, float]:
        """Play a line in real time. Returns (start, end) in call seconds."""
        pcm = self.lines[name]
        ulaw = audioop.lin2ulaw(pcm, 2)
        start = self.now()
        self.playing = True
        next_t = time.monotonic()
        try:
            for i in range(0, len(ulaw), CHUNK):
                if self.closed.is_set() or (stop and stop.is_set()):
                    break
                self.seq += 1
                t = self.now()
                self.outbound.append((t, pcm[i * 2 : (i + CHUNK) * 2]))
                await self.ws.send_text(
                    json.dumps(
                        {
                            "event": "media",
                            "voice_id": self.voice_id,
                            "stream_id": self.stream_id,
                            "media": {
                                "seq": self.seq,
                                "timestamp": int(time.time() * 1000),
                                "format": {"encoding": "ulaw", "sample_rate": 8000, "channels": 1},
                                "payload": base64.b64encode(ulaw[i : i + CHUNK]).decode(),
                            },
                        }
                    )
                )
                next_t += CHUNK / RATE
                await asyncio.sleep(max(0.0, next_t - time.monotonic()))
        finally:
            self.playing = False
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


async def run_script(call: Call, result: Result) -> None:
    """The scripted conversation. Each step is skipped if the call has ended."""
    await asyncio.wait_for(call.started.wait(), 10)
    answered_at = call.now()

    def left() -> float:
        return CALL_BUDGET_S - (call.now() - answered_at)

    # 1. Greeting: most agents speak first.
    onset = await call.wait_onset(answered_at, timeout=min(8.0, left()))
    if onset is None:
        result.notes.append("agent said nothing in the first 8 s; we spoke first")
    else:
        result.greeting_after_s = round(onset - answered_at, 2)
        if not await call.wait_quiet(timeout=min(15.0, left())):
            result.notes.append("greeting still going after 15 s")
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
            await call.wait_quiet(timeout=min(15.0, left()))

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
                end = call.det.speech_end_after(t_int)
                if end is not None and end <= t_int + 4.0:
                    result.bargein = "stopped"
                    result.bargein_stop_after_s = round(end - t_int, 2)
                else:
                    result.bargein = "kept talking for 4 s+"

    # Count replies that started while we were mid-line, outside the barge-in test.
    our_spans = _spans_of(call.outbound)
    result.spoke_over_us = sum(
        1 for s in call.det.spans if any(a < s.start < b - 0.3 for a, b in our_spans)
    )


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
        ("When talked over", r.bargein + (f", after {r.bargein_stop_after_s:.2f} s" if r.bargein_stop_after_s is not None else "")),
        ("Started talking while we were mid-sentence", str(r.spoke_over_us)),
        ("Call length", s(r.call_seconds)),
    ]
    out = [f"# {r.label}: call to {r.to}", "", "| | |", "|---|---|"]
    out += [f"| {k} | {v} |" for k, v in rows]
    out += ["", "Times are measured from the call audio as it arrived, so they include the phone network both ways: what a caller hears."]
    if r.notes:
        out += ["", "Notes:"] + [f"- {n}" for n in r.notes]
    if r.transcript:
        out += ["", "What the agent said:", "", f"> {r.transcript}"]
    out += ["", f"voice_id: {r.voice_id}"]
    return "\n".join(out) + "\n"


def build_app(state: dict) -> FastAPI:
    app = FastAPI()

    @app.post("/event")
    async def event(request: Request):
        body = await request.json()
        st = body.get("state")
        state["events"].append((round(time.monotonic() - state["t0"], 2), st))
        if st == "connected":
            state["result"].answered = True
            status, reply = await enablex(
                "PUT", f"/{body.get('voice_id')}/stream", {"wss_host": f"wss://{state['host']}/ws"}
            )
            if status >= 300 or (isinstance(reply, dict) and reply.get("state") == "stream_failed"):
                state["result"].notes.append(f"stream did not start: HTTP {status} {reply}")
                await enablex("DELETE", f"/{body.get('voice_id')}")
        elif st == "disconnected":
            state["result"].call_seconds = body.get("call_duration")
            state["result"].end_reason = body.get("disconnect_reason")
            state["over"].set()
        return {}

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        call = Call(websocket, state["lines"])
        state["call"] = call
        receiver = asyncio.create_task(call.receive())
        try:
            await run_script(call, state["result"])
        except Exception as e:
            state["result"].notes.append(f"script stopped: {type(e).__name__}: {e}")
        finally:
            if call.voice_id and not call.closed.is_set():
                await enablex("DELETE", f"/{call.voice_id}")
            await asyncio.wait([receiver], timeout=5)

    return app


async def ngrok_host() -> str:
    try:
        async with aiohttp.ClientSession() as http:
            async with http.get("http://127.0.0.1:4040/api/tunnels", timeout=3) as r:
                tunnels = (await r.json()).get("tunnels", [])
    except Exception:
        return ""
    return next((t["public_url"] for t in tunnels if t.get("public_url", "").startswith("https://")), "")


async def main():
    p = argparse.ArgumentParser(description="Call a voice agent and measure it.")
    p.add_argument("--to", required=True, help="number to call, digits only, e.g. 91XXXXXXXXXX")
    p.add_argument("--label", required=True, help="short name for the report, e.g. acme-demo")
    p.add_argument("--host", help="public host forwarding to this machine; default: the ngrok tunnel")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()

    host = (args.host or await ngrok_host()).removeprefix("https://").rstrip("/")
    if not host:
        raise SystemExit("No public host. Start `ngrok http 8765`, or pass --host.")

    cache = Path("results/.tts")
    lines = {k: trim_silence(await tts_pcm(v, cache)) for k, v in LINES.items()}
    result = Result(label=args.label, to=args.to)
    state = {
        "host": host, "lines": lines, "result": result, "events": [],
        "over": asyncio.Event(), "t0": time.monotonic(), "call": None,
    }

    server = uvicorn.Server(uvicorn.Config(build_app(state), host="0.0.0.0", port=args.port, log_level="warning"))
    serving = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)

    status, reply = await enablex(
        "POST", "",
        {"name": "voice-agent-probe", "owner_ref": args.label, "from": os.environ["ENABLEX_FROM_NUMBER"],
         "to": args.to, "event_url": f"https://{host}/event"},
    )
    print(f"dial-out: HTTP {status} {reply}")
    if status < 300 and isinstance(reply, dict):
        result.voice_id = reply.get("voice_id")
        try:
            await asyncio.wait_for(state["over"].wait(), timeout=90)
        except TimeoutError:
            result.notes.append("no disconnected event within 90 s")
        await asyncio.sleep(1)

    server.should_exit = True
    await serving

    call: Call | None = state["call"]
    out = Path("results")
    out.mkdir(exist_ok=True)
    stem = out / f"{time.strftime('%Y%m%d-%H%M%S')}_{args.label}"
    if call is not None:
        result.transcript = await transcribe(b"".join(pcm for _, pcm in call.inbound))
        write_wav(stem.with_suffix(".wav"), call)
    stem.with_suffix(".json").write_text(
        json.dumps({**asdict(result), "events": state["events"]}, indent=2), encoding="utf-8"
    )
    md = markdown(result)
    stem.with_suffix(".md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"saved {stem}.{{json,md,wav}}")


if __name__ == "__main__":
    asyncio.run(main())
