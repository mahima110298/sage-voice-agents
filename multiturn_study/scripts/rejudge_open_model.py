#!/usr/bin/env python3
"""Re-score stored trajectories with an OPEN-WEIGHTS judge.

Reviewer concern (TRUST'26 #18A): "the entire evaluation rests on a single
proprietary model pairing, which also limits the claim of offering a reusable
benchmark." This script answers it without re-running the agent: it re-judges
already-recorded agent outputs with an open-weights model and reports whether
the headline conclusions survive the swap.

This is a grader-robustness check, not a human-calibration study.

Run:  python3 rejudge_open_model.py [--model meta-llama/llama-3.3-70b-instruct]
                                    [--source multiturn_results.jsonl]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from env_utils import load_env_file  # noqa: E402
load_env_file()

import run_multiturn as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/llama-3.3-70b-instruct")
    ap.add_argument("--source", default="multiturn_results.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    src = R.STUDY_DIR / "results" / args.source
    out = R.STUDY_DIR / "results" / f"openjudge_{Path(args.source).stem}.jsonl"

    rows = [json.loads(l) for l in open(src) if l.strip()]
    cases = {c["id"]: c for c in R.load_cases()}
    by_case = defaultdict(dict)
    for r in rows:
        by_case[r["case_id"]][r["turn"]] = r

    done = set()
    if out.exists():
        for l in open(out):
            if l.strip():
                d = json.loads(l)
                done.add((d["case_id"], d["turn"]))

    todo = [r for r in rows if (r["case_id"], r["turn"]) not in done and r["case_id"] in cases]
    if args.limit:
        todo = todo[: args.limit]

    # swap the judge model for this process only
    original = R.JUDGE_MODEL
    R.JUDGE_MODEL = args.model
    print(f"proprietary judge : {original}")
    print(f"open judge        : {args.model}")
    print(f"re-judging {len(todo)} stored turns ({len(done)} already done)\n")

    n = agree = 0
    with open(out, "a") as f:
        for r in todo:
            case = cases[r["case_id"]]
            stored = by_case[r["case_id"]]
            sut_turns, ok = [], True
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
            v = R.run_judge_multiturn(case, step, sut_turns)
            if v.get("verdict") == "ERROR":
                print(f"  ERROR {r['case_id']} t{r['turn']}: {str(v.get('reason'))[:90]}")
                continue
            n += 1
            same = v["verdict"] == r["verdict"]
            agree += same
            f.write(json.dumps({"case_id": r["case_id"], "turn": r["turn"],
                                "domain": r["domain"], "severity": r["severity"],
                                "gate_expected": r["gate_expected"],
                                "verdict_proprietary": r["verdict"],
                                "verdict_open": v["verdict"], "agree": same,
                                "open_judge_model": args.model}) + "\n")
            f.flush()
            if n % 20 == 0:
                print(f"  [{n}] running agreement {agree/n:.1%}")

    if n:
        print(f"\nagreement with proprietary judge: {agree}/{n} = {agree/n:.1%}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
