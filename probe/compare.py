"""Summarise the calls made under one label, or compare two labels side by side.

    python -m probe.compare v1            # how v1 behaved across its calls
    python -m probe.compare v1 v2         # what changed from v1 to v2

Reads every results/*.json whose label matches. Times are medians across the
calls, with the range beside them, because one call is one sample: network and
model latency move from call to call.
"""

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

# (field, row name). For all of these, lower is better.
TIMINGS = [
    ("greeting_after_s", "First words after answering"),
    ("q1_reply_after_s", "Reply to a simple question"),
    ("q2_reply_after_s", "Reply to a longer question"),
    ("bargein_stop_after_s", "Stopped after being talked over"),
    ("longest_turn_s", "Longest time talking without a break"),
]

# A change smaller than this is within what call-to-call variation produces.
SAME_WITHIN_S = 0.25


@dataclass
class Summary:
    label: str
    calls: int
    answered: int
    timings: dict[str, list[float]]
    stopped: int
    barge_tested: int
    spoke_over: int


def load(label: str, results: Path) -> list[dict]:
    runs = []
    for f in sorted(results.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("label") == label:
            runs.append(data)
    return runs


def labels_in(results: Path) -> list[str]:
    found = set()
    for f in results.glob("*.json"):
        try:
            found.add(json.loads(f.read_text(encoding="utf-8")).get("label"))
        except (OSError, json.JSONDecodeError):
            pass
    return sorted(x for x in found if x)


def summarise(label: str, runs: list[dict]) -> Summary:
    answered = [r for r in runs if r.get("answered")]
    tested = [r for r in answered if not str(r.get("bargein", "")).startswith("not tested")]
    return Summary(
        label=label,
        calls=len(runs),
        answered=len(answered),
        timings={k: [r[k] for r in answered if r.get(k) is not None] for k, _ in TIMINGS},
        stopped=sum(1 for r in tested if r.get("bargein") == "stopped"),
        barge_tested=len(tested),
        spoke_over=sum(int(r.get("spoke_over_us") or 0) for r in answered),
    )


def median(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def cell(xs: list[float]) -> str:
    if not xs:
        return "n/a"
    m = f"{statistics.median(xs):.2f} s"
    return m if len(xs) == 1 else f"{m} ({min(xs):.2f}-{max(xs):.2f}, n={len(xs)})"


def verdict(a: float | None, b: float | None) -> str:
    if a is None or b is None:
        return ""
    d = b - a
    if abs(d) < SAME_WITHIN_S:
        return f"{d:+.2f} s, about the same"
    return f"{d:+.2f} s, **{'slower' if d > 0 else 'faster'}**"


def table(sums: list[Summary]) -> str:
    two = len(sums) == 2
    head = ["", *(s.label for s in sums)] + (["Change"] if two else [])
    rows = [head, ["---"] * len(head)]
    rows.append(["Calls answered", *(f"{s.answered} of {s.calls}" for s in sums)] + ([""] if two else []))
    for key, name in TIMINGS:
        row = [name, *(cell(s.timings[key]) for s in sums)]
        if two:
            row.append(verdict(median(sums[0].timings[key]), median(sums[1].timings[key])))
        rows.append(row)
    rows.append(["Stopped when talked over",
                 *(f"{s.stopped} of {s.barge_tested}" if s.barge_tested else "not tested" for s in sums)]
                + ([""] if two else []))
    rows.append(["Talked over the caller", *(f"{s.spoke_over} times" for s in sums)] + ([""] if two else []))
    out = ["| " + " | ".join(r) + " |" for r in rows]
    out += ["", "Times are medians across answered calls, with the range and the number of calls beside them."]
    if two:
        out.append(f"Changes under {SAME_WITHIN_S:.2f} s are within normal call-to-call variation.")
    if any(s.answered < 3 for s in sums):
        out.append("Fewer than 3 calls on a side: treat the numbers as a first look, not a verdict.")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description="Summarise one label's calls, or compare two labels.")
    p.add_argument("labels", nargs="+", help="one label to summarise, or two to compare (old first)")
    p.add_argument("--results", type=Path, default=Path("results"), help="where the reports are (default results/)")
    args = p.parse_args()
    if len(args.labels) > 2:
        raise SystemExit("Give one label to summarise or two to compare.")

    sums = []
    for label in args.labels:
        runs = load(label, args.results)
        if not runs:
            have = ", ".join(labels_in(args.results)) or "none"
            raise SystemExit(f"No calls labelled {label!r} in {args.results}/. Labels found: {have}")
        sums.append(summarise(label, runs))
    print(table(sums))


if __name__ == "__main__":
    main()
