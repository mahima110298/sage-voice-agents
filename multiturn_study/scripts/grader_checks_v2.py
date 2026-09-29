#!/usr/bin/env python3
"""Grader-robustness checks for judge v2 (thesis Section 6.6).

  stability  the same 80 baseline turns judged twice with v2: verdict and gate agreement
  open       all 270 baseline turns, pinned v2 judge vs open-weights judge under v2

Verdicts are compared both as the judge returned them and after trap_match
decides SEVERE, since the thesis reports the latter.

Run:  python3 grader_checks_v2.py > ../results/v2/grader_checks.txt
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_multiturn as R  # noqa: E402
from trap_match import final_verdict  # noqa: E402

V2 = R.STUDY_DIR / "results" / "v2"


def load(name):
    p = V2 / name
    return {(d["case_id"], d["turn"]): d for d in map(json.loads, open(p))} if p.exists() else {}


def main():
    cases = {c["id"]: c for c in R.load_cases()}

    def traps(k):
        return next(s for s in cases[k[0]]["steps"] if s["turn"] == k[1]).get("trap_tool_calls") or []

    def fin(d, k):
        return final_verdict(d["verdict"], d["sut_tool_calls"], traps(k))

    base = load("baseline.jsonl")
    for label, other in (("STABILITY (v2 judged twice)", load("stability_rejudge.jsonl")),
                         ("OPEN-WEIGHTS JUDGE vs pinned judge, v2 prompt", load("openjudge_baseline.jsonl"))):
        keys = sorted(set(base) & set(other))
        print("=" * 70)
        print(label, f"n={len(keys)}")
        print("=" * 70)
        if not keys:
            print("  not available yet\n")
            continue
        raw = sum(base[k]["verdict"] == other[k]["verdict"] for k in keys)
        f = [(fin(base[k], k), fin(other[k], k)) for k in keys]
        agree = sum(a == b for a, b in f)
        sev = sum((a == "SEVERE") == (b == "SEVERE") for a, b in f)
        pf = sum((a == "PASS") == (b == "PASS") for a, b in f)
        gate = sum(base[k]["detected_gate"] == other[k]["detected_gate"] for k in keys)
        n = len(keys)
        print(f"  raw verdict agreement          {raw}/{n} = {100*raw/n:.1f}%")
        print(f"  reported verdict agreement     {agree}/{n} = {100*agree/n:.1f}%")
        print(f"  pass vs not-pass agreement     {pf}/{n} = {100*pf/n:.1f}%")
        print(f"  severe vs not-severe agreement {sev}/{n} = {100*sev/n:.1f}%")
        print(f"  detected-gate agreement        {gate}/{n} = {100*gate/n:.1f}%")
        print("  disagreement cells (pinned, other):", dict(Counter((a, b) for a, b in f if a != b)))
        by_gate = Counter()
        tot_gate = Counter()
        for k, (a, b) in zip(keys, f):
            g = base[k]["gate_expected"]
            tot_gate[g] += 1
            by_gate[g] += a != b
        print("  disagreement by required gate:",
              {g: f"{by_gate[g]}/{tot_gate[g]} = {100*by_gate[g]/tot_gate[g]:.1f}%" for g in tot_gate})
        by_turn = Counter()
        tot_turn = Counter()
        for k, (a, b) in zip(keys, f):
            tot_turn[k[1]] += 1
            by_turn[k[1]] += a != b
        print("  disagreement by turn:", {t: f"{by_turn[t]}/{tot_turn[t]}" for t in sorted(tot_turn)})
        if "open" in label.lower():
            ts_p = sum(a == "PASS" for a, _ in f) / n
            ts_o = sum(b == "PASS" for _, b in f) / n
            print(f"  task_success pinned {100*ts_p:.1f}%  open {100*ts_o:.1f}%")
        print()


if __name__ == "__main__":
    main()
