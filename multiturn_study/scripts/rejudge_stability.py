#!/usr/bin/env python3
"""Decompose evaluation noise: is verdict instability the AGENT or the JUDGE?

Two independent runs of the same benchmark disagree on ~16% of turns. That
number mixes two sources: the SUT is stochastic (Gemini Live may act
differently on identical audio) and the judge is stochastic (an LLM scoring
identical text may return a different verdict).

This script isolates the judge. It re-scores ALREADY-STORED agent outputs --
same sut_text, same sut_tool_calls, same prompt, same pinned model -- so the
agent contributes nothing. Any disagreement is judge-only noise.

    judge-only flip rate        = measured here
    total flip rate (2 runs)    = 16.1% (agent + judge)
    => agent-attributable noise = total - judge-only (approximately)

Run:  python3 rejudge_stability.py [--n 80]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from env_utils import load_env_file  # noqa: E402
load_env_file()

from run_multiturn import load_cases, run_judge_multiturn, RESULTS_PATH  # noqa: E402

OUT = HERE.parent / "results" / "judge_stability.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80, help="how many stored turns to re-judge")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(RESULTS_PATH) if l.strip()]
    cases = {c["id"]: c for c in load_cases()}

    # group stored rows by case so we can rebuild the sut_turns list the judge needs
    by_case = defaultdict(dict)
    for r in rows:
        by_case[r["case_id"]][r["turn"]] = r

    candidates = [r for r in rows if r["case_id"] in cases and not r.get("sut_error")]
    random.Random(args.seed).shuffle(candidates)
    picked = candidates[: args.n]
    print(f"re-judging {len(picked)} stored turns with the identical judge and inputs\n")

    done = 0
    agree = 0
    by_turn = defaultdict(lambda: [0, 0])
    with open(OUT, "w") as f:
        for r in picked:
            case = cases[r["case_id"]]
            stored = by_case[r["case_id"]]
            # rebuild the per-turn SUT outputs exactly as first scored
            sut_turns = []
            ok = True
            for step in case["steps"]:
                s = stored.get(step["turn"])
                if s is None:
                    ok = False
                    break
                sut_turns.append({"turn": step["turn"], "text": s["sut_text"],
                                  "tool_calls": s["sut_tool_calls"], "_error": None})
            if not ok:
                continue
            step = next(s for s in case["steps"] if s["turn"] == r["turn"])

            v2 = run_judge_multiturn(case, step, sut_turns)
            if v2.get("verdict") == "ERROR":
                continue
            done += 1
            same = (v2["verdict"] == r["verdict"])
            agree += same
            by_turn[r["turn"]][0] += 1
            if not same:
                by_turn[r["turn"]][1] += 1
            f.write(json.dumps({"case_id": r["case_id"], "turn": r["turn"],
                                "verdict_run1": r["verdict"], "verdict_rejudge": v2["verdict"],
                                "agree": same}) + "\n")
            f.flush()
            mark = "OK " if same else "FLIP"
            print(f"  [{done:3d}] {mark} {r['case_id']:22s} t{r['turn']}  "
                  f"{r['verdict']:6s} -> {v2['verdict']}")

    if not done:
        print("nothing re-judged")
        return
    print(f"\n{'='*60}")
    print(f"JUDGE-ONLY stability (identical agent outputs, identical judge)")
    print(f"  n = {done}")
    print(f"  agreement  = {agree}/{done} = {agree/done:.1%}")
    print(f"  FLIP RATE  = {1-agree/done:.1%}")
    print(f"\n  by turn depth:")
    for t in sorted(by_turn):
        n, fl = by_turn[t]
        print(f"    turn {t}: {fl}/{n} = {fl/n:.0%}")
    print(f"\n  (total 2-run flip rate incl. agent stochasticity was 16.1%)")


if __name__ == "__main__":
    main()
