#!/usr/bin/env python3
"""Whole-project progress: pipeline, dev runs, test runs (grouped), what's left, spend.
Read-only. Usage: python3 scripts/overview.py [--watch 30]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

STUDY = Path(__file__).resolve().parent.parent
RES = STUDY / "results" / "rp"
W = 24


def bar(done, total):
    f = min(1.0, done / total) if total else 0.0
    n = int(round(f * W))
    return "█" * n + "░" * (W - n) + f" {100 * f:5.1f}%"


def turns_done(path: Path) -> int:
    if not path.exists():
        return 0
    seen = {}
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        seen[(r.get("case_id"), r.get("turn"))] = not (r.get("sut_error") or r.get("verdict") == "ERROR")
    return sum(seen.values())


def subset_turns(split_dir: str, ids_file: Path) -> int:
    raw = json.loads(ids_file.read_text())
    ids = set(raw if isinstance(raw, list) else raw["case_ids"])
    n = 0
    for d in (STUDY / "data" / split_dir).iterdir():
        m = d / "metadata.json"
        if m.exists():
            c = json.loads(m.read_text())
            if c["id"] in ids:
                n += len(c["steps"])
    return n


def running() -> set:
    out = subprocess.run(["pgrep", "-fl", "rp.runner"], capture_output=True, text=True).stdout
    live = set()
    for line in out.splitlines():
        p = line.split()
        g = lambda f: p[p.index(f) + 1] if f in p else None
        live.add((g("--split"), g("--agent"), g("--config"), g("--pass")))
    return live


def spend():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import run_test_grid as rtg  # same provider bucketing as the budget guard
    return rtg.spend_by_provider([rtg.RESULTS_ROOT / "cost_log.jsonl", *rtg.sibling_cost_logs()])


_or_cache = {"t": 0, "v": None}


def openrouter_left():
    if time.time() - _or_cache["t"] < 300:
        return _or_cache["v"]
    key = None
    env = STUDY / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"')
    try:
        req = urllib.request.Request("https://openrouter.ai/api/v1/credits", headers={"Authorization": f"Bearer {key}"})
        d = json.load(urllib.request.urlopen(req, timeout=10))["data"]
        _or_cache["v"] = d["total_credits"] - d["total_usage"]
    except Exception:
        pass
    _or_cache["t"] = time.time()
    return _or_cache["v"]


# ------------------------------------------------------------------ definitions
PIPELINE = [
    ("Build + review all code (24 tasks)", "done"),
    ("Dev runs: baselines, SAGE, tuning", "done"),
    ("Judge v3 built, validated, dev re-scored", "done"),
    ("Pick 80 test cases, freeze SAGE, pre-register", "done"),
    ("Test: cheap open models (Qwen, Voxtral, open-cascade)", "now"),
    ("Test: closed models (Gemini 3.8, GPT mini, Gemini 3.1)", "decide"),
    ("Second-judge re-score (Llama 3.3 70B)", "next"),
    ("Analysis tables + figures (report.py)", "next"),
    ("Human validation (2 raters)", "you"),
    ("Thesis writing (chapters from current results)", "now"),
    ("Final review + submit", "next"),
]
ICON = {"done": "✓", "now": "▶", "next": "·", "decide": "?", "you": "👤"}

DEV = [  # (label, pass names, per-file expected turns: 'full' | 'subset')
    ("Dev baselines + original SAGE (pass 1)", ["pass1"], "full"),
    ("New-SAGE redesign, withdrawn (pass 2, partial)", ["pass2"], "full"),
    ("SAGE tuning rounds (pass 11-13, subset)", ["pass11", "pass12", "pass13"], "subset"),
    ("SAGE full-dev confirmation (pass 20)", ["pass20"], "full"),
    ("New SAGE = policy + enforcement (pass 30, subset)", ["pass30"], "subset"),
]

FULL = ["B0", "B2", "B3", "B4", "SAGE"]
LIGHT = ["B0", "B2", "SAGE"]
ABL = ["SAGE-no_gate_types", "SAGE-no_verifier", "SAGE-no_credential_check", "SAGE-no_policy"]
TEST = [
    ("Qwen 3.8 Omni Flash - full grid", "qwen3.8-omni-flash", FULL),
    ("Qwen 3.8 Omni Flash - ablations", "qwen3.8-omni-flash", ABL),
    ("Voxtral Small 24B", "voxtral-small-24b", LIGHT),
    ("Open cascade (Whisper + gpt-oss)", "open-cascade", LIGHT),
    ("Gemini 3.8 Live - full grid", "gemini-3.8-live", FULL),
    ("GPT Realtime 2.1 mini", "gpt-realtime-2.1-mini", LIGHT),
    ("Gemini 3.1 Flash Live", "gemini-3.1-flash-live-preview", LIGHT),
]


def render():
    live = running()
    full_dev = 270
    sub_dev = subset_turns("dev_v2", STUDY / "data" / "dev_v2" / "_tuning_subset.json")
    test_turns = subset_turns("test_cases_v3", STUDY / "data" / "test_subset_v3.json")
    L = [f"Right Pause — whole project   {time.strftime('%Y-%m-%d %H:%M:%S')}   runners active: {len(live)}", ""]

    L.append("== PIPELINE")
    L += [f" {ICON[s]} {name}" for name, s in PIPELINE]
    L.append("")

    L.append("== DEV RUNS (dev_v2)")
    for label, passes, kind in DEV:
        files = [f for p in passes for f in (RES / "dev_v2").glob(f"*/*/{p}.jsonl")]
        exp = full_dev if kind == "full" else sub_dev
        d = sum(min(turns_done(f), exp) for f in files)
        t = exp * len(files)
        L.append(f" {'✓' if d >= 0.97 * t or 'withdrawn' in label else '▶'} {label:<50} {bar(d, t)}  {len(files)} files")
    L.append("")

    L.append(f"== TEST RUNS (80 cases, {test_turns} turns x 2 passes per setup)")
    gd = gt = 0
    for label, agent, cfgs in TEST:
        d = t = nrun = ndone = 0
        act = False
        for c in cfgs:
            for p in (1, 2):
                f = RES / "test" / agent / c / f"pass{p}.jsonl"
                x = min(turns_done(f), test_turns)
                d += x; t += test_turns; nrun += 1; ndone += x >= test_turns
                act |= ("test", agent, c, str(p)) in live
        gd += d; gt += t
        mark = "✓" if d >= t else ("▶" if act else ("·" if d == 0 or d < 0.1 * t else "…"))
        L.append(f" {mark} {label:<36} {bar(d, t)}  {ndone:>2}/{nrun} runs")
    L.append(f"   {'ALL 52 TEST RUNS':<36} {bar(gd, gt)}  {gd}/{gt} turns")
    L.append("")

    sp = spend()
    orl = openrouter_left()
    L.append("== SPEND  " + "  ".join(f"{k} ${v:.2f}" for k, v in sorted(sp.items())) + f"   total ${sum(sp.values()):.2f}"
             + (f"   | OpenRouter left ${orl:.2f}" if orl is not None else ""))
    L.append("✓ done  ▶ running  … partial/paused  · not started  ? your decision  👤 you")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=int, default=0)
    a = ap.parse_args()
    while True:
        text = render()
        if a.watch:
            os.system("clear")
        print(text, flush=True)
        if not a.watch:
            break
        time.sleep(a.watch)


if __name__ == "__main__":
    main()
