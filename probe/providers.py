"""Telephony providers: how to place the call, and how each one's media websocket talks.

Every provider here streams the call as 8 kHz mu-law in JSON websocket
messages, so the probe only needs to know four things about each: how to dial,
how to hang up, which webhooks it calls, and the shape of its messages.

Adding another provider means one more subclass of Provider.
"""

import base64
import os
import time
from dataclasses import dataclass
from urllib.parse import quote
from xml.sax.saxutils import quoteattr

import aiohttp
from fastapi import FastAPI, Request, Response


@dataclass
class Start:
    call_id: str | None
    stream_id: str | None


@dataclass
class Media:
    ulaw: bytes


@dataclass
class Stop:
    pass


class Provider:
    name = ""
    env: tuple[str, ...] = ()
    budget_s = 50.0  # how long the scripted conversation may run

    def check_env(self) -> None:
        missing = [k for k in self.env if not os.environ.get(k)]
        if missing:
            raise SystemExit(f"{self.name} needs these set (environment or .env): {', '.join(missing)}")

    async def dial(self, to: str, host: str, label: str) -> tuple[str | None, str]:
        """Place the call. Returns (call id or None, a line for the log)."""
        raise NotImplementedError

    async def hangup(self, call_id: str) -> None:
        raise NotImplementedError

    def mount(self, app: FastAPI, state: dict) -> None:
        """Add the provider's webhook routes."""

    def parse(self, msg: dict) -> Start | Media | Stop | None:
        raise NotImplementedError

    def media(self, ulaw: bytes, stream_id: str | None, call_id: str | None, seq: int) -> dict:
        raise NotImplementedError


def _digits(number: str) -> str:
    return "".join(c for c in number if c.isdigit())


class EnableX(Provider):
    """EnableX Voice API. Trial accounts cut every call at 45 s."""

    name = "enablex"
    env = ("ENABLEX_APP_ID", "ENABLEX_APP_KEY", "ENABLEX_FROM_NUMBER")
    budget_s = 40.0
    api = "https://api.enablex.io/voice/v1/call"

    def _headers(self) -> dict:
        token = base64.b64encode(
            f"{os.environ['ENABLEX_APP_ID']}:{os.environ['ENABLEX_APP_KEY']}".encode()
        ).decode()
        return {"Authorization": f"Basic {token}", "Content-Type": "application/json"}

    async def _api(self, method: str, path: str, body: dict | None = None):
        async with aiohttp.ClientSession() as http:
            async with http.request(method, f"{self.api}{path}", json=body, headers=self._headers()) as r:
                try:
                    return r.status, await r.json(content_type=None)
                except Exception:
                    return r.status, await r.text()

    async def dial(self, to, host, label):
        status, reply = await self._api(
            "POST", "",
            {"name": "voice-agent-probe", "owner_ref": label, "from": _digits(os.environ["ENABLEX_FROM_NUMBER"]),
             "to": _digits(to), "event_url": f"https://{host}/event"},
        )
        call_id = reply.get("voice_id") if status < 300 and isinstance(reply, dict) else None
        return call_id, f"dial-out: HTTP {status} {reply}"

    async def hangup(self, call_id):
        await self._api("DELETE", f"/{call_id}")

    def mount(self, app, state):
        @app.post("/event")
        async def event(request: Request):
            body = await request.json()
            st = body.get("state")
            state["events"].append((round(time.monotonic() - state["t0"], 2), st))
            if st == "connected":
                # EnableX starts streaming only when asked, once the call is up.
                vid = body.get("voice_id")
                status, reply = await self._api("PUT", f"/{vid}/stream", {"wss_host": f"wss://{state['host']}/ws"})
                if status >= 300 or (isinstance(reply, dict) and reply.get("state") == "stream_failed"):
                    state["result"].notes.append(f"stream did not start: HTTP {status} {reply}")
                    await self.hangup(vid)
            elif st == "disconnected":
                state["result"].call_seconds = body.get("call_duration")
                state["result"].end_reason = body.get("disconnect_reason")
                state["over"].set()
            return {}

    def parse(self, msg):
        event = msg.get("event")
        if event == "start_media":
            return Start(call_id=(msg.get("start") or {}).get("voice_id"), stream_id=msg.get("stream_id"))
        if event == "media":
            payload = (msg.get("media") or {}).get("payload")
            return Media(base64.b64decode(payload)) if payload else None
        if event == "stop_media":
            return Stop()
        return None

    def media(self, ulaw, stream_id, call_id, seq):
        return {
            "event": "media",
            "voice_id": call_id,
            "stream_id": stream_id,
            "media": {
                "seq": seq,
                "timestamp": int(time.time() * 1000),
                "format": {"encoding": "ulaw", "sample_rate": 8000, "channels": 1},
                "payload": base64.b64encode(ulaw).decode(),
            },
        }


class Twilio(Provider):
    """Twilio Programmable Voice with Media Streams.

    Trial accounts play a notice and wait for a key press before connecting,
    which a voice agent never gives; use an upgraded account.
    """

    name = "twilio"
    env = ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM_NUMBER")

    def _url(self, path: str = "") -> str:
        sid = os.environ["TWILIO_ACCOUNT_SID"]
        return f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Calls{path}.json"

    def _auth(self) -> aiohttp.BasicAuth:
        return aiohttp.BasicAuth(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])

    async def dial(self, to, host, label):
        # The stream starts when the call is answered, as part of the TwiML.
        twiml = f"<Response><Connect><Stream url={quoteattr(f'wss://{host}/ws')} /></Connect></Response>"
        form = [
            ("To", "+" + _digits(to)),
            ("From", "+" + _digits(os.environ["TWILIO_FROM_NUMBER"])),
            ("Twiml", twiml),
            ("StatusCallback", f"https://{host}/twilio/status"),
            ("StatusCallbackEvent", "answered"),
            ("StatusCallbackEvent", "completed"),
        ]
        async with aiohttp.ClientSession(auth=self._auth()) as http:
            async with http.post(self._url(), data=form) as r:
                reply = await r.json(content_type=None)
                status = r.status
        call_id = reply.get("sid") if status < 300 and isinstance(reply, dict) else None
        return call_id, f"dial-out: HTTP {status} {reply.get('status') or reply.get('message') if isinstance(reply, dict) else reply}"

    async def hangup(self, call_id):
        async with aiohttp.ClientSession(auth=self._auth()) as http:
            async with http.post(self._url(f"/{quote(call_id)}"), data={"Status": "completed"}) as r:
                await r.read()

    def mount(self, app, state):
        @app.post("/twilio/status")
        async def status(request: Request):
            form = await request.form()
            st = form.get("CallStatus")
            state["events"].append((round(time.monotonic() - state["t0"], 2), st))
            if st in ("completed", "busy", "no-answer", "failed", "canceled"):
                duration = form.get("CallDuration")
                state["result"].call_seconds = float(duration) if duration else None
                state["result"].end_reason = st
                state["over"].set()
            return Response(status_code=204)

    def parse(self, msg):
        event = msg.get("event")
        if event == "start":
            start = msg.get("start") or {}
            return Start(call_id=start.get("callSid"), stream_id=start.get("streamSid") or msg.get("streamSid"))
        if event == "media":
            payload = (msg.get("media") or {}).get("payload")
            return Media(base64.b64decode(payload)) if payload else None
        if event == "stop":
            return Stop()
        return None

    def media(self, ulaw, stream_id, call_id, seq):
        return {"event": "media", "streamSid": stream_id, "media": {"payload": base64.b64encode(ulaw).decode()}}


PROVIDERS: dict[str, type[Provider]] = {p.name: p for p in (EnableX, Twilio)}
