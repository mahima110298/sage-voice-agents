#!/usr/bin/env python3
"""Live progress of experiment runs: one bar per (split, agent, config, pass) file,
plus total spend. Read-only. Usage:
    python3 scripts/progress.py [--split dev_v2|test] [--watch 20]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

STUDY = Path(__file__).resolve().parent.parent
RES = STUDY / "results" / "rp"
TURNS = {"dev_v2": 270, "test": 254}
WIDTH = 30


def expected_turns(split: str, name: str) -> int:
    subset = STUDY / "data" / "dev_v2" / "_tuning_subset.json"
    if split == "dev_v2" and name in {"pass11", "pass12", "pass13", "pass30"} and subset.exists():
        raw = json.loads(subset.read_text())
        ids = set(raw if isinstance(raw, list) else (raw.get("case_ids") or raw.get("cases") or []))
        n = 0
        for d in (STUDY / "data" / "dev_v2").iterdir():
            m = d / "metadata.json"
            if m.exists():
                c = json.loads(m.read_text())
                if c["id"] in ids:
                    n += len(c["steps"])
        return n or TURNS[split]
    return TURNS[split]


def rows_done(path: Path) -> int:
    seen = set()
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if not r.get("sut_error") and r.get("verdict") != "ERROR":
            seen.add((r.get("case_id"), r.get("turn")))
    return len(seen)


def spend() -> float:
    total = 0.0
    for log in (STUDY / "results").glob("rp*/cost_log.jsonl"):
        for line in log.read_text().splitlines():
            try:
                total += json.loads(line).get("known_cost_usd") or 0
            except ValueError:
                pass
    return total


def running() -> set[tuple]:
    try:
        out = subprocess.run(["pgrep", "-fl", "rp.runner"], capture_output=True, text=True).stdout
    except OSError:
        return set()
    live = set()
    for line in out.splitlines():
        parts = line.split()
        get = lambda f: parts[parts.index(f) + 1] if f in parts else None
        live.add((get("--split"), get("--agent"), get("--config"), f"pass{get('--pass')}"))
    return live


def bar(done: int, total: int) -> str:
    frac = min(1.0, done / total) if total else 0.0
    fill = int(round(frac * WIDTH))
    return "█" * fill + "░" * (WIDTH - fill) + f" {100 * frac:5.1f}%  {done}/{total}"


STAGES = [
    ("Build + review all code", "done"),
    ("Dev baselines + original SAGE", "done"),
    ("Judge v3: build + review (5 fixes)", "done"),
    ("Judge v3: fresh re-score of 40-case dev subset", "done"),
    ("New-SAGE dev runs (40 cases x 2 agents, pass30)", "done"),
    ("Pick 80 test cases, freeze SAGE, analysis plan", "done"),
    ("Test runs (80 cases, 2 passes, 52 runs)", "now"),
]


def rescore_bars() -> list[str]:
    out = []
    subset_file = STUDY / "data" / "dev_v2" / "_tuning_subset.json"
    subset = None
    if subset_file.exists():
        raw = json.loads(subset_file.read_text())
        subset = set(raw if isinstance(raw, list) else raw.get("case_ids", []))
    for root in ("rp_v3final2",):
        base = STUDY / "results" / root / "dev_v2"
        if not base.exists():
            continue
        done = total = 0
        for f in (RES / "dev_v2").glob("*/*/pass*.jsonl"):
            if ".judge2" in f.name:
                continue
            src = {(json.loads(l).get("case_id"), json.loads(l).get("turn")) for l in f.read_text().splitlines() if l.strip()}
            if subset is not None:
                src = {k for k in src if k[0] in subset}
            total += len(src)
            g = base / f.relative_to(RES / "dev_v2")
            if g.exists():
                got = set()
                for l in g.read_text().splitlines():
                    try:
                        r = json.loads(l)
                    except ValueError:
                        continue
                    if r.get("verdict") != "ERROR":
                        got.add((r.get("case_id"), r.get("turn")))
                done += len(got & src)
        out.append(f"   judge re-score {root:<8} {bar(done, total)}")
    return out


def render(split_filter: str | None) -> str:
    live = running()
    lines = [f"Right Pause runs — {time.strftime('%H:%M:%S')}   spend so far: ${spend():.2f}   active runners: {len(live)}", ""]
    icon = {"done": "✓", "now": "▶", "next": "·"}
    lines.append("== pipeline")
    lines += [f" {icon[st]} {name}" for name, st in STAGES]
    # lines += rescore_bars()  # dev re-score finished (old-SAGE files skipped on purpose)
    lines.append("")
    grand_done = grand_total = 0
    for split in ("dev_v2", "test"):
        if split_filter and split != split_filter:
            continue
        base = RES / split
        if not base.exists():
            continue
        lines.append(f"== {split}")
        for f in sorted(base.glob("*/*/pass*.jsonl")):
            if ".judge2" in f.name:
                continue
            agent, config, name = f.parent.parent.name, f.parent.name, f.stem
            total = expected_turns(split, name)
            done = rows_done(f)
            grand_done += min(done, total)
            grand_total += total
            mark = "▶" if (split, agent, config, name) in live else ("✓" if done >= total else " ")
            lines.append(f" {mark} {agent:<30} {config:<24} {name:<7} {bar(done, total)}")
        lines.append("")
    if grand_total:
        lines.append(f"OVERALL  {bar(grand_done, grand_total)}")
    lines.append("▶ running   ✓ complete")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(TURNS), default=None)
    ap.add_argument("--watch", type=int, default=0, help="refresh every N seconds")
    a = ap.parse_args()
    while True:
        text = render(a.split)
        if a.watch:
            os.system("clear")
        print(text)
        sys.stdout.flush()
        if not a.watch:
            break
        time.sleep(a.watch)


if __name__ == "__main__":
    main()
