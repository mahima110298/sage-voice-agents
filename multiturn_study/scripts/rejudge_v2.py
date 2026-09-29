#!/usr/bin/env python3
"""Re-score stored Study 2 trajectories with judge v2 (see judge_v2.py).

The agent is not re-run: every arm's stored sut_text and sut_tool_calls are
re-judged, so only the scoring rules change. Also re-runs the two grader checks
under v2, so they describe the judge the thesis now reports:

  arms       all 270 turns of baseline, clarify-budget and guardrail
  stability  80 baseline turns re-judged a second time with v2 (judge-only noise)
  open       all 270 baseline turns with the open-weights judge under v2

Output goes to results/v2/. Resumable: finished turns are skipped.

Run:  python3 rejudge_v2.py [--only arms|stability|open] [--workers 8]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from env_utils import load_env_file  # noqa: E402
load_env_file()

import run_multiturn as R  # noqa: E402
from judge_v2 import run_judge_v2  # noqa: E402

RES = R.STUDY_DIR / "results"
OUT = RES / "v2"
ARMS = {
    "baseline": "multiturn_results.jsonl",
    "clarify_budget": "multiturn_results_clarifybudget.jsonl",
    "guardrail": "multiturn_results_guardrail.jsonl",
}
OPEN_MODEL = "meta-llama/llama-3.3-70b-instruct"


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def sut_turns_for(case, stored):
    return [{"turn": s["turn"], "text": stored[s["turn"]]["sut_text"],
             "tool_calls": stored[s["turn"]]["sut_tool_calls"], "_error": None}
            for s in case["steps"]]


def rejudge(rows, out_path, model, cases, workers, label, context=None):
    by_case = defaultdict(dict)
    for r in context or rows:
        by_case[r["case_id"]][r["turn"]] = r
    done = set()
    if out_path.exists():
        done = {(d["case_id"], d["turn"]) for d in load(out_path) if d["verdict"] != "ERROR"}
    todo = [r for r in rows if (r["case_id"], r["turn"]) not in done]
    print(f"[{label}] {len(todo)} to judge ({len(done)} already done) with {model}", flush=True)

    lock = threading.Lock()
    counter = {"n": 0, "err": 0}

    def work(r):
        case = cases[r["case_id"]]
        step = next(s for s in case["steps"] if s["turn"] == r["turn"])
        v = run_judge_v2(case, step, sut_turns_for(case, by_case[r["case_id"]]), model=model)
        row = dict(r)
        row["verdict_v1"], row["detected_gate_v1"] = r["verdict"], r["detected_gate"]
        row.update({k: v.get(k) for k in ("verdict", "reason", "detected_gate", "judge_model",
                                          "judge_provider", "judge_prompt_version")})
        with lock:
            counter["n"] += 1
            counter["err"] += v.get("verdict") == "ERROR"
            if v.get("verdict") != "ERROR":
                with open(out_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
            if counter["n"] % 25 == 0:
                print(f"  [{label}] {counter['n']}/{len(todo)}  errors={counter['err']}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, todo))
    print(f"[{label}] finished, errors={counter['err']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["arms", "stability", "open"], default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    cases = {c["id"]: c for c in R.load_cases()}

    if args.only in (None, "arms"):
        for arm, f in ARMS.items():
            rejudge(load(RES / f), OUT / f"{arm}.jsonl", R.JUDGE_MODEL, cases, args.workers, arm)

    if args.only in (None, "stability"):
        # Same 80 turns as the v1 stability check, re-judged against the v2 baseline.
        v1_ids = [(d["case_id"], d["turn"]) for d in load(RES / "judge_stability.jsonl")]
        base = {(r["case_id"], r["turn"]): r for r in load(RES / ARMS["baseline"])}
        rows = [base[k] for k in v1_ids if k in base]
        rejudge(rows, OUT / "stability_rejudge.jsonl", R.JUDGE_MODEL, cases, args.workers, "stability",
                context=list(base.values()))

    if args.only in (None, "open"):
        rejudge(load(RES / ARMS["baseline"]), OUT / "openjudge_baseline.jsonl", OPEN_MODEL,
                cases, args.workers, "open")


if __name__ == "__main__":
    main()
