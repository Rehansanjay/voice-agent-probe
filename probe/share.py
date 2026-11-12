"""Turn a number-test call into one page you can send to the agent's owner.

    python -m probe.share results/20261112-172802_husky-num.json --name "Husky Voice"

Writes results/<stem>_share.html next to the json: a single self-contained file
(audio included) that opens in any browser, so it can go out as an email
attachment. It shows, for each number we read, whether the agent cut in during
our pause or waited, how long it took to reply, and the moment itself to listen
to. Nothing is uploaded anywhere.
"""

import argparse
import base64
import html
import io
import json
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

SPOKEN = {
    "mobile number": ("98401", "23456", "My mobile number is nine eight four zero one … two three four five six"),
    "pin code": ("600", "041", "And the pin code is six zero zero … zero four one"),
}
FIX_URL = "https://github.com/Rehansanjay/numhold"
PROBE_URL = "https://github.com/Rehansanjay/voice-agent-probe"
CLIP_PAD_S = 1.0
CLIP_MAX_S = 16.0


def load_audio(wav_path: Path) -> tuple[np.ndarray, int] | None:
    if not wav_path.exists():
        return None
    with wave.open(str(wav_path)) as w:
        rate, ch = w.getframerate(), w.getnchannels()
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, ch)
    return a, rate


def wav_data_uri(stereo: np.ndarray, rate: int) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(stereo.shape[1])
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(stereo.astype(np.int16).tobytes())
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()


def clips(r: dict, audio) -> list[str | None]:
    """One short clip per number: from just before we start reading it to just
    before the next one (or 16 s), so the owner hears exactly the moment."""
    if audio is None:
        return [None] * len(r["splits"])
    a, rate = audio
    starts = [s.get("at_s") for s in r["splits"]]
    out = []
    for i, t in enumerate(starts):
        if t is None:
            out.append(None)
            continue
        nxt = next((u for u in starts[i + 1:] if u is not None), None)
        end = min(t + CLIP_MAX_S, (nxt - 0.3) if nxt else len(a) / rate)
        seg = a[int(max(0, t - CLIP_PAD_S) * rate): int(end * rate)].astype(np.int32)
        mono = np.clip(seg.sum(axis=1, keepdims=True), -32768, 32767)  # both sides, as the call sounded
        out.append(wav_data_uri(mono, rate))
    return out


def verdict(s: dict, pause: float) -> tuple[str, str, str]:
    """(css class, headline, detail) for one number."""
    first, rest, _ = SPOKEN.get(s["what"], ("", "", ""))
    if s.get("unclear"):
        return ("unclear", "Not a clean test",
                f"The agent was still talking while we started, so we can't say. ({s['unclear']})")
    if s.get("cut_in_after_s") is not None:
        return ("cut", "Cut the caller off",
                f"The agent started talking {s['cut_in_after_s']:.2f} s into our {pause:.1f} s pause, "
                f"after hearing only “{first}”. A real caller would be talked over halfway through the number.")
    reply = s.get("reply_after_s")
    tail = f" It replied {reply:.2f} s after the caller finished." if reply is not None else ""
    return ("ok", "Waited for the whole number", f"It let the {pause:.1f} s pause go by and heard “{first} {rest}”.{tail}")


def summary(r: dict, name: str) -> tuple[str, str]:
    splits = [s for s in r["splits"] if not s.get("unclear")]
    cuts = [s for s in splits if s.get("cut_in_after_s") is not None]
    replies = [s["reply_after_s"] for s in splits if s.get("reply_after_s") is not None]
    if cuts:
        what = " and ".join(s["what"] for s in cuts)
        return ("cut", f"{name} cut the caller off in the middle of the {what}.")
    if not splits:
        return ("unclear", f"The test on {name} didn't run cleanly; worth another call.")
    slow = max(replies) if replies else None
    if slow is not None and slow >= 1.5:
        return ("ok", f"{name} waited for every number, but it waits on every reply too: "
                      f"up to {slow:.1f} s after the caller stops.")
    return ("ok", f"{name} waited for every number and replied quickly.")


def next_step(kind: str) -> str:
    if kind == "cut":
        return ("People read numbers in chunks, and fast turn detection hears the pause as the end of the turn. "
                f"<a href=\"{FIX_URL}\">numhold</a> is a small open-source check in front of the turn detector: "
                "if the transcript shows a number is clearly unfinished, it waits up to 2 s for the rest. "
                "On 341 numbers read by Indian English and Hindi speakers it took Smart Turn from 30% cut off to 8%, "
                "and costs a delay on 6% of normal turn ends. I can help fit it to your stack.")
    return ("Agents usually avoid cutting numbers off by waiting longer after every sentence, which callers feel "
            f"on every reply. <a href=\"{FIX_URL}\">numhold</a> waits only when a number is clearly unfinished, "
            "so the rest of the conversation can use a much shorter wait. I can help measure what that "
            "would save on your agent.")


CSS = """
:root{--bg:#fafaf9;--card:#fff;--ink:#1c1917;--muted:#57534e;--line:#e7e5e4;
--ok:#15803d;--ok-bg:#f0fdf4;--cut:#b91c1c;--cut-bg:#fef2f2;--unclear:#a16207;--unclear-bg:#fefce8;--link:#1d4ed8}
@media (prefers-color-scheme:dark){:root{--bg:#1c1917;--card:#292524;--ink:#f5f5f4;--muted:#a8a29e;--line:#44403c;
--ok:#4ade80;--ok-bg:#14532d40;--cut:#f87171;--cut-bg:#7f1d1d40;--unclear:#facc15;--unclear-bg:#713f1240;--link:#93c5fd}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:16px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:720px;margin:0 auto;padding:32px 16px 48px}
h1{font-size:1.5rem;line-height:1.25;margin:0 0 6px}.sub{color:var(--muted);margin:0 0 24px;font-size:.92rem}
.banner{border-radius:10px;padding:14px 16px;margin:0 0 24px;font-weight:600}
.banner.ok{background:var(--ok-bg);color:var(--ok)}.banner.cut{background:var(--cut-bg);color:var(--cut)}
.banner.unclear{background:var(--unclear-bg);color:var(--unclear)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:0 0 16px}
.card h2{font-size:1.05rem;margin:0 0 4px}.said{color:var(--muted);font-style:italic;margin:0 0 10px}
.tag{display:inline-block;font-size:.8rem;font-weight:700;padding:2px 8px;border-radius:999px;margin:0 0 8px}
.tag.ok{background:var(--ok-bg);color:var(--ok)}.tag.cut{background:var(--cut-bg);color:var(--cut)}
.tag.unclear{background:var(--unclear-bg);color:var(--unclear)}
audio{width:100%;margin-top:10px}.hint{color:var(--muted);font-size:.85rem;margin:6px 0 0}
h3{font-size:1rem;margin:28px 0 8px}blockquote{margin:0;padding:0 0 0 12px;border-left:3px solid var(--line);color:var(--muted)}
a{color:var(--link)}footer{margin-top:32px;color:var(--muted);font-size:.85rem;border-top:1px solid var(--line);padding-top:12px}
"""


def render(r: dict, name: str, audio, stem: str = "") -> str:
    pause = r.get("number_pause_s") or 1.0
    kind, line = summary(r, name)
    try:  # results are named <YYYYmmdd-HHMMSS>_<label>
        when = datetime.strptime(stem[:15], "%Y%m%d-%H%M%S").strftime("%d %b %Y")
    except ValueError:
        when = ""
    cards = []
    per_number = clips(r, audio)
    whole = ""
    if audio is not None and not any(per_number):
        # Older results don't say where each number starts: give the whole call instead.
        a, rate = audio
        mono = np.clip(a.astype(np.int32).sum(axis=1, keepdims=True), -32768, 32767)
        whole = (f'<h3>The whole call</h3><audio controls preload="none" src="{wav_data_uri(mono, rate)}"></audio>'
                 '<p class="hint">Agent and caller mixed as the call sounded.</p>')
    for s, clip in zip(r["splits"], per_number):
        cls, head, detail = verdict(s, pause)
        said = SPOKEN.get(s["what"], ("", "", ""))[2]
        player = (f'<audio controls preload="none" src="{clip}"></audio>'
                  '<p class="hint">Agent and caller mixed as the call sounded. Our pause comes after the first chunk.</p>'
                  if clip else "")
        cards.append(f'<section class="card"><span class="tag {cls}">{html.escape(head)}</span>'
                     f'<h2>The {html.escape(s["what"])}</h2><p class="said">“{html.escape(said)}”</p>'
                     f'<p>{html.escape(detail)}</p>{player}</section>')
    transcript = r.get("transcript")
    heard = (f"<h3>What the agent said</h3><blockquote>{html.escape(transcript)}</blockquote>" if transcript else "")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(name)}: number test</title><style>{CSS}</style></head><body><main>
<h1>Does {html.escape(name)} let callers finish a number?</h1>
<p class="sub">A real phone call{(' on ' + when) if when else ''}. We read a mobile number and a PIN code the way people do,
in two chunks with a {pause:.1f} s pause in the middle, and listened for whether the agent jumped in.
Times are measured from the call audio, phone network included.</p>
<div class="banner {kind}">{html.escape(line)}</div>
{''.join(cards)}
{whole}
<h3>Why it matters</h3><p>{next_step(kind)}</p>
{heard}
<footer>Measured with <a href="{PROBE_URL}">voice-agent-probe</a> by Rehan Sanjay.
Happy to re-run it after any change, or with longer pauses.</footer>
</main></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result", type=Path, help="a results/*.json from a --script number call")
    ap.add_argument("--name", help="the agent's name as the owner calls it (default: the label)")
    args = ap.parse_args()
    r = json.loads(args.result.read_text(encoding="utf-8"))
    if r.get("script") != "number" or not r.get("splits"):
        raise SystemExit("this call didn't run the number test (use --script number)")
    out = args.result.with_name(args.result.stem + "_share.html")
    out.write_text(render(r, args.name or r["label"], load_audio(args.result.with_suffix(".wav")),
                                args.result.stem), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
