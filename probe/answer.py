"""Wait for a voice agent to phone us, pick up, and measure it the way probe.call does.

    python -m probe.answer --label acme-demo

For agents you can't dial: many demos are a form where you type a number and
the agent calls you. Type your Twilio number into the form, with this running.

Twilio only for now. In the Twilio console, set the number's "A call comes in"
webhook to https://<your tunnel>/twilio/incoming (HTTP POST).

If the agent stays quiet for 1.5 s after we pick up, we say "Hello?", as a
person would, and its first words are timed from the end of that.

Only submit a demo form for a number that is yours, and only where the
company offers that demo publicly.
"""

import argparse
import asyncio
import os

from probe.call import ngrok_host, answer_calls
from probe.providers import PROVIDERS


async def main():
    p = argparse.ArgumentParser(description="Wait for a voice agent to call, and measure it.")
    p.add_argument("--label", required=True, help="short name for the report, e.g. acme-demo")
    p.add_argument("--calls", type=int, default=1, help="how many calls to wait for, one after another (default 1)")
    p.add_argument("--wait", type=float, default=600.0, help="seconds to wait for each call (default 600)")
    p.add_argument("--provider", choices=sorted(PROVIDERS), default=os.environ.get("PROBE_PROVIDER", "twilio"))
    p.add_argument("--budget", type=float, help="seconds the conversation may run (default: per provider)")
    p.add_argument("--host", help="public host forwarding to this machine; default: the ngrok tunnel")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    if args.calls < 1:
        raise SystemExit("--calls must be at least 1")

    provider = PROVIDERS[args.provider]()
    if not provider.answers_calls:
        raise SystemExit(provider.answer_note or f"{provider.name} can't answer calls here yet.")
    provider.check_env()
    host = (args.host or await ngrok_host()).removeprefix("https://").rstrip("/")
    if not host:
        raise SystemExit("No public host. Start `ngrok http 8765`, or pass --host.")

    print(f"Set the number's incoming-call webhook to https://{host}/twilio/incoming")
    await answer_calls(provider, host, args.label, calls=args.calls, budget_s=args.budget,
                       port=args.port, wait_s=args.wait)
    if args.calls > 1:
        print(f"\nsummary: python -m probe.compare {args.label}")


if __name__ == "__main__":
    asyncio.run(main())
