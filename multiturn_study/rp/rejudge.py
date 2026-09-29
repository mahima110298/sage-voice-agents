"""Re-score stored agent outputs with judge v3 (never re-runs the agent).

  cd multiturn_study
  python3 -m rp.rejudge --judge v3 [--samples N] --split dev_v2 \\
      [--agent A --config C --pass K] --dst-root results/rp_v3 [--workers 8] [--max-usd 12]

Reads every primary pass file `{src-root}/{split}/{agent}/{config}/pass{K}.jsonl`
(default src-root results/rp; last row per (case_id, turn) wins, rp.results_io)
READ-ONLY, and writes the same layout under `--dst-root`:
  {split}/{agent}/{config}/pass{K}.jsonl    the source rows with the judge fields
      replaced by v3's (verdict, verdict_judge, detected_gate, reason, judge_model,
      judge_prompt_version, severe_matcher, judge_path, checklist) plus
      judge_samples, source_verdict, source_detected_gate, source_run_id,
      rejudge_run_id, rejudge_git_sha. `run_id` stays the source (agent) run id.
  {split}/{agent}/{config}/manifest[.pass{K}].json   judge version/model/samples,
      source file and source run_ids, one entry per invocation
  cost_log.jsonl    one record per case (same schema as the runner's, kind
      "rejudge", a single "judge" component)

Every non-ERROR source row is re-scored from its stored sut_text /
sut_tool_calls (executed calls; `blocked` never reached the backend and is not
shown). A source row with sut_error or verdict ERROR is copied through with
verdict ERROR and judge_path "source_error" (it has no agent output to score).

Resumable: a (case_id, turn) already in the destination with a non-ERROR
verdict (or a source_error pass-through) is skipped; judge ERRORs are re-judged
and appended (last row wins). Refuses to resume into a folder whose manifest
names a different judge version/model/samples, source file or judge_rules_hash
(sha256 of the checklist prompt, question texts and judge_v3.py source), or
whose first invocation ran at a different git SHA (override:
--allow-code-change, recorded). --dst-root must be a direct child of
multiturn_study/results/ named rp* and never results/rp (or inside/containing it).

Every v3 row carries `judge_rules_hash` (judge_v3.rules_hash). `--remap`
re-applies the current judge code to the stored checklist answers of rows
whose hash differs -- no LLM calls (a turn whose new route needs an answer that
was never asked becomes a judge ERROR and is counted).

Cost cap: the global $400 cap counts this destination's cost log plus every
sibling `results/rp*/cost_log.jsonl` (rp.runner.sibling_cost_logs), so runner
and re-score spend are one budget. `--max-usd` additionally stops this
invocation once its own spend reaches the limit (exit code 3).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import threading
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import hashlib

from rp import judge_v3, results_io, runner
from rp.cases import SPLITS, load_split
from rp.runner import (COST_CAP_USD, EXIT_COST_CAP, RESULTS_DIR, STUDY, CostGuard, check_allowed,
                       load_prices, manifest_path, price_usage, sibling_cost_logs)
from run_multiturn import JUDGE_MODEL
from env_utils import load_env_file

PASS_RE = re.compile(r"^pass(\d+)\.jsonl$")
JUDGE_FIELDS = ("verdict", "verdict_judge", "detected_gate", "reason", "judge_model", "judge_prompt_version",
                "severe_matcher", "judge_path", "checklist", "judge_rules_hash")
MANIFEST_KEYS = ("split", "agent", "config", "pass", "judge_model", "judge_prompt_version", "judge_samples",
                 "source_file", "judge_rules_hash")
# Destination roots must be direct children of this folder named rp* (so the
# runner's cost cap sees their cost logs), never results/rp itself.
RESULTS_BASE = STUDY / "results"


def judge_rules_hash() -> str:
    """judge_v3.rules_hash(): resuming under different judge rules is refused."""
    return judge_v3.rules_hash()


class ReplayClient:
    """Returns stored checklist answers in order (for --remap): no LLM call."""

    def __init__(self, answers: list[dict]):
        self.answers = list(answers)

    def complete(self, model, messages, temperature):
        if not self.answers:
            raise RuntimeError("remap: no stored checklist answer for this turn under the new rules")
        return json.dumps(self.answers.pop(0)), {"prompt_tokens": 0, "completion_tokens": 0}


def _stored_answers(row: dict) -> list[dict]:
    c = row.get("checklist") or {}
    if c.get("_raw"):
        return list(c["_raw"])
    if not c:
        return []
    return [c.get("_llm") or {k: v for k, v in c.items() if not k.startswith("_")}]


def remap_file(out: Path, src_rows: list[dict], selected: set, cases: dict, extra: dict) -> Counter:
    """Re-apply the current judge code to one destination pass file's stored
    checklist answers (no LLM calls); appends one row per re-mapped turn."""
    current = judge_v3.rules_hash()
    by_case: dict[str, dict] = {}
    for r in src_rows:
        by_case.setdefault(r["case_id"], {})[r["turn"]] = r
    stats: Counter = Counter()
    rows = []
    for r in results_io.load_rows(out):
        key = (r.get("case_id"), r.get("turn"))
        if key not in selected or r.get("judge_rules_hash") == current:
            continue
        if r.get("judge_path") == "source_error":   # nothing to judge: restamp only
            rows.append({**r, "judge_rules_hash": current, **extra})
            stats["restamped"] += 1
            continue
        if r.get("judge_path") not in ("code", "checklist", "code+checklist"):
            continue
        case = cases[r["case_id"]]
        rs = by_case.get(r["case_id"], {})
        sut = [{"turn": s["turn"], "text": (rs.get(s["turn"]) or {}).get("sut_text") or "",
                "tool_calls": (rs.get(s["turn"]) or {}).get("sut_tool_calls") or [], "_error": None}
               for s in case["steps"]]
        samples = int(r.get("judge_samples") or 1)
        j = judge_v3.judge_case(case, sut, client=ReplayClient(_stored_answers(r)), model=r.get("judge_model"),
                                samples=samples, only_turns={r["turn"]})[r["turn"] - 1]
        stats["remapped"] += 1
        stats["changed"] += j["verdict"] != r.get("verdict")
        stats["errors"] += j["verdict"] == "ERROR"
        rows.append({**r, **{k: j.get(k) for k in JUDGE_FIELDS}, "prefix_verdict": r.get("verdict"),
                     "remapped_from_stored_checklist": True, **extra})
    if rows:
        results_io.append_rows(out, rows)
    return stats


def check_dst_root(dst_root: Path, src_root: Path) -> None:
    base = Path(RESULTS_BASE).resolve()
    d, src, main = Path(dst_root).resolve(), Path(src_root).resolve(), (base / "rp").resolve()
    for other in {src, main}:
        if d == other or other in d.parents or d in other.parents:
            raise SystemExit(f"refusing --dst-root {dst_root}: it is, is inside, or contains {other} "
                             "(source results are read-only)")
    if d.parent != base or not d.name.startswith("rp"):
        raise SystemExit(f"refusing --dst-root {dst_root}: must be a direct child of {base} named rp* "
                         "(so the $400 cap counts its cost log)")
DEFAULT_CASE_USD = 0.05  # in-flight estimate for the cap before any case is logged


def _is_source_error(r: dict) -> bool:
    return bool(r.get("sut_error")) or r.get("verdict") == "ERROR"


def _done(r: dict) -> bool:
    return r.get("judge_path") == "source_error" or (
        r.get("verdict") != "ERROR" and r.get("verdict_judge") != "ERROR")


def source_files(src_root: Path, split: str, agent=None, config=None, pass_=None):
    base = Path(src_root) / split
    if not base.exists():
        raise SystemExit(f"no source results for split {split!r}: {base}")
    for f in sorted(base.glob("*/*/pass*.jsonl")):
        m = PASS_RE.match(f.name)
        if not m:
            continue
        a, c, k = f.parent.parent.name, f.parent.name, int(m.group(1))
        if (agent and a != agent) or (config and c != config) or (pass_ is not None and k != pass_):
            continue
        yield a, c, k, f


def _rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(STUDY))
    except ValueError:
        return str(p)


def write_manifest(folder: Path, cur: dict, inv: dict, allow_code_change: bool = False,
                   allow_rules_change: bool = False) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    path = manifest_path(folder, cur["pass"])
    old = json.loads(path.read_text()) if path.exists() else None
    if old:
        diff = [k for k in MANIFEST_KEYS if old.get(k) != cur.get(k)]
        if allow_rules_change and "judge_rules_hash" in diff:
            diff.remove("judge_rules_hash")
            inv = {**inv, "previous_judge_rules_hash": old.get("judge_rules_hash")}
        if diff:
            raise SystemExit(f"refusing to resume into {folder}: manifest differs on {diff} "
                             f"({ {k: (old.get(k), cur.get(k)) for k in diff} }); use another --dst-root")
        first_sha = (old.get("invocations") or [{}])[0].get("git_sha")
        if first_sha != inv["git_sha"] and not allow_code_change:
            raise SystemExit(f"refusing to resume into {folder}: git_sha {inv['git_sha']} differs from the pass "
                             f"file's first invocation ({first_sha}); use another --dst-root, or "
                             "--allow-code-change (recorded in the manifest)")
    man = {**cur, "created": old["created"] if old else inv["created"],
           "invocations": (old or {}).get("invocations", []) + [inv]}
    path.write_text(json.dumps(man, indent=2) + "\n")


def parse_args(argv):
    p = argparse.ArgumentParser(prog="python3 -m rp.rejudge", description=__doc__.split("\n\n")[0])
    p.add_argument("--judge", required=True, choices=("v3",))
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--split", required=True, choices=sorted(SPLITS))
    p.add_argument("--agent")
    p.add_argument("--config")
    p.add_argument("--pass", dest="pass_", type=int)
    p.add_argument("--src-root", default=str(RESULTS_DIR))
    p.add_argument("--dst-root", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-usd", type=float, default=None, help="stop this invocation at this own spend")
    p.add_argument("--allow-code-change", action="store_true",
                   help="resume a destination pass file although HEAD differs from its first invocation")
    p.add_argument("--case-ids-file", default=None,
                   help="JSON list of case ids or {\"case_ids\": [...]} (same format as the runner's): "
                        "re-score only these cases")
    p.add_argument("--redo-keys", default=None,
                   help="JSON list of [agent, config, pass, case_id, turn]: re-judge these turns even if already "
                        "done (appended; last row wins)")
    p.add_argument("--allow-rules-change", action="store_true",
                   help="resume although judge_rules_hash changed (the manifest takes the new hash; the old one "
                        "is recorded in the invocation). Use with --redo-keys for turns the change affects")
    p.add_argument("--remap", action="store_true",
                   help="no LLM calls: re-apply the current judge code to the stored checklist answers of rows "
                        "already in --dst-root whose judge_rules_hash differs (appended; last row wins)")
    p.add_argument("--keys-file", default=None,
                   help="JSON list of [agent, config, pass, case_id, turn]: re-score only these turns")
    a = p.parse_args(argv)
    if a.samples < 1:
        raise SystemExit("--samples must be >= 1")
    return a


def main(argv=None, *, client=None) -> int:
    args = parse_args(argv)
    check_allowed(args.split)
    src_root, dst_root = Path(args.src_root), Path(args.dst_root)
    check_dst_root(dst_root, src_root)
    if args.remap:
        args.allow_code_change = args.allow_rules_change = True  # remap exists to apply changed code
    if client is None and not args.remap:
        load_env_file()
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise SystemExit("refusing to start: OPENROUTER_API_KEY is not set")
        client = judge_v3.default_client()
    redo = {tuple(k) for k in json.loads(Path(args.redo_keys).read_text())} if args.redo_keys else set()
    keys = None
    if args.keys_file:
        keys = {tuple(k) for k in json.loads(Path(args.keys_file).read_text())}
    cases = {c["id"]: c for c in load_split(args.split)}
    only_cases = None
    if args.case_ids_file:
        ids = runner.read_case_ids(Path(args.case_ids_file))
        unknown = [i for i in ids if i not in cases]
        if unknown:
            raise SystemExit(f"case ids not in split {args.split}: {unknown}")
        only_cases = set(ids)
    prices = load_prices()
    cost_log = dst_root / "cost_log.jsonl"
    own = cost_log.resolve()
    extra = sorted({p.resolve() for p in sibling_cost_logs(dst_root)} | {(Path(RESULTS_BASE) / "rp" / "cost_log.jsonl").resolve()}
                   - {own})  # resolved: never count one log twice
    guard = CostGuard(cost_log=cost_log, extra_logs=extra, max_case=DEFAULT_CASE_USD)
    if not guard.can_start():
        raise SystemExit(f"refusing to start: cumulative spend ${guard.cumulative:.2f} would exceed the "
                         f"${COST_CAP_USD:.0f} cap -- ask the student before continuing")
    run_id, git_sha = uuid.uuid4().hex, runner._git("rev-parse", "HEAD").strip() or None
    rules_hash = judge_rules_hash()
    created = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    lock = threading.Lock()
    state = {"spend": 0.0, "stopped": False, "judged": 0, "remap": Counter()}

    def over_budget() -> bool:
        return args.max_usd is not None and state["spend"] >= args.max_usd

    for agent, config, pass_, src in source_files(src_root, args.split, args.agent, args.config, args.pass_):
        src_rows = results_io.load_rows(src)
        selected = {(r.get("case_id"), r.get("turn")) for r in src_rows
                    if (keys is None or (agent, config, pass_, r.get("case_id"), r.get("turn")) in keys)
                    and (only_cases is None or r.get("case_id") in only_cases)}
        if not selected:
            continue
        folder = dst_root / args.split / agent / config
        out = folder / f"pass{pass_}.jsonl"
        if args.remap and not out.exists():
            continue
        write_manifest(folder, {
            "kind": "rejudge", "split": args.split, "agent": agent, "config": config, "pass": pass_,
            "judge_model": JUDGE_MODEL, "judge_prompt_version": args.judge, "judge_samples": args.samples,
            "source_file": _rel(src), "source_run_ids": sorted({r.get("run_id") for r in src_rows if r.get("run_id")}),
            "judge_rules_hash": rules_hash,
        }, {"run_id": run_id, "created": created, "git_sha": git_sha, "allow_code_change": bool(args.allow_code_change),
            "argv": sys.argv[1:] if argv is None else argv}, allow_code_change=args.allow_code_change,
            allow_rules_change=args.allow_rules_change)
        if args.remap:
            st = remap_file(out, src_rows, selected, cases, {"rejudge_run_id": run_id, "rejudge_git_sha": git_sha})
            state["remap"].update(st)
            print(f"{args.split}/{agent}/{config}/pass{pass_}: remapped {st['remapped']} turn(s), "
                  f"{st['changed']} verdict(s) changed, {st['errors']} error(s)", flush=True)
            continue
        done = {(r.get("case_id"), r.get("turn")) for r in results_io.load_rows(out) if _done(r)
                and (agent, config, pass_, r.get("case_id"), r.get("turn")) not in redo}
        by_case: dict[str, dict] = {}
        for r in src_rows:
            by_case.setdefault(r["case_id"], {})[r["turn"]] = r
        todo = {cid: rs for cid, rs in by_case.items()
                if any((cid, t) in selected and (cid, t) not in done for t in rs)}

        def one(cid, rs, out=out, agent=agent, config=config, pass_=pass_):
            with lock:
                if state["stopped"] or over_budget() or not guard.can_start():
                    state["stopped"] = True
                    return
            case = cases.get(cid)
            if case is None:
                print(f"warning: {cid} not in split {args.split}; skipped", file=sys.stderr)
                return
            sut = []
            for s in case["steps"]:
                r = rs.get(s["turn"])
                sut.append({"turn": s["turn"], "text": (r or {}).get("sut_text") or "",
                            "tool_calls": (r or {}).get("sut_tool_calls") or [],
                            "_error": "no source row" if r is None else (r.get("sut_error") or None)})
            write = {t for t in rs if (cid, t) in selected and (cid, t) not in done}
            want = {t for t in write if not _is_source_error(rs[t])}
            usage: dict = {}
            judged = judge_v3.judge_case(case, sut, client=client, model=JUDGE_MODEL, samples=args.samples,
                                         usage_out=usage, only_turns=want) if want else [None] * len(case["steps"])
            rows = []
            for s, j in zip(case["steps"], judged):
                r = rs.get(s["turn"])
                if s["turn"] not in write:
                    continue
                extra = {"judge_samples": args.samples, "source_verdict": r.get("verdict"),
                         "source_detected_gate": r.get("detected_gate"), "source_run_id": r.get("run_id"),
                         "rejudge_run_id": run_id, "rejudge_git_sha": git_sha}
                if _is_source_error(r):
                    new = {"verdict": "ERROR", "verdict_judge": None, "detected_gate": "unknown",
                           "reason": f"source row not scoreable: {r.get('sut_error') or 'judge ERROR'}",
                           "judge_model": None, "judge_prompt_version": args.judge,
                           "severe_matcher": r.get("severe_matcher"), "judge_path": "source_error", "checklist": None,
                           "judge_rules_hash": rules_hash}
                else:
                    new = {k: j.get(k) for k in JUDGE_FIELDS}
                rows.append({**r, **new, **extra})
            comp = {"role": "judge", "model": JUDGE_MODEL, "usage": usage}
            comp.update(price_usage(usage, JUDGE_MODEL, prices))
            rec = {"ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "kind": "rejudge",
                   "split": args.split, "agent": agent, "config": config, "pass": pass_, "run_id": run_id,
                   "git_sha": git_sha, "case_id": cid, "judge_prompt_version": args.judge,
                   "judge_samples": args.samples, "source_run_ids": sorted({r.get("run_id") for r in rs.values()
                                                                             if r.get("run_id")}),
                   "components": [comp], "cost_usd": comp["cost_usd"], "known_cost_usd": comp["known_cost_usd"]}
            with lock:
                results_io.append_rows(out, rows)
                guard.record(comp["known_cost_usd"], rec)
                state["spend"] += comp["known_cost_usd"]
                state["judged"] += len(want)

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            list(ex.map(lambda kv: one(*kv), todo.items()))
        print(f"{args.split}/{agent}/{config}/pass{pass_}: {len(todo)} case(s) re-scored | run "
              f"${state['spend']:.4f} | cumulative ${guard.cumulative:.4f} / ${COST_CAP_USD:.0f}", flush=True)
        if state["stopped"]:
            break

    if args.remap:
        print(f"remap done: {dict(state['remap'])} (no LLM calls)", flush=True)
        return 0
    print(f"done: {state['judged']} turn(s) judged; run ${state['spend']:.4f}; cumulative "
          f"${guard.cumulative:.4f}", flush=True)
    if state["stopped"]:
        print(f"STOPPED on budget (run ${state['spend']:.2f}, max-usd {args.max_usd}, cumulative "
              f"${guard.cumulative:.2f} / ${COST_CAP_USD:.0f}); re-run to resume", file=sys.stderr)
        return EXIT_COST_CAP
    return 0


if __name__ == "__main__":
    sys.exit(main())
