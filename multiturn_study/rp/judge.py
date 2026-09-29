"""Judge wrapper for the right-pause runner.

`judge_turns(case, turns, model)` scores every TurnRecord of one case with the
pilot's judge v2 (`scripts/judge_v2.py: run_judge_v2`, unchanged) and decides
SEVERE with the deterministic matcher (`scripts/trap_match.py: final_verdict`),
never with the LLM judge.

The judge sees only EXECUTED calls (`TurnRecord.calls`). Calls SAGE or a
baseline blocked (`TurnRecord.blocked`) never reached the backend and are never
shown to the judge or the matcher as tool calls.

Judge token usage: `run_judge_v2` does not return the HTTP response's `usage`,
and judge_v2.py must not be modified. So this module wraps the `requests`
object judge_v2 uses with a pass-through recorder: `post` is the real
`requests.post` (same arguments, same response object), and the response's
`usage` is summed into a thread-local accumulator while `judge_turns(...,
usage_out=...)` is active. Nothing the judge sends or receives changes.
A response without `usage` is counted as `missing_usage_responses` so the cost
log marks the judge cost unknown rather than zero.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

_SCRIPTS = str(Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import judge_v2 as _jv2  # noqa: E402
from judge_v2 import run_judge_v2, JUDGE_PROMPT_VERSION  # noqa: E402,F401
from trap_match import final_verdict  # noqa: E402


def add_usage(acc: dict, usage: dict, prefix: str = "") -> None:
    """Sum numeric usage fields into `acc`; nested dicts become dotted keys."""
    for k, v in (usage or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            acc[key] = acc.get(key, 0) + v
        elif isinstance(v, dict):
            add_usage(acc, v, f"{key}.")


_tls = threading.local()


class _UsageRecordingRequests:
    """Pass-through stand-in for the `requests` module inside judge_v2."""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def post(self, *args, **kwargs):
        resp = self._real.post(*args, **kwargs)
        acc = getattr(_tls, "acc", None)
        if acc is not None:
            try:
                usage = resp.json().get("usage")
            except Exception:  # noqa: BLE001 -- non-JSON error body; judge_v2 handles the failure
                usage = None
            if usage:
                add_usage(acc, usage)
            elif getattr(resp, "ok", True):
                acc["missing_usage_responses"] = acc.get("missing_usage_responses", 0) + 1
        return resp


if isinstance(_jv2.requests, _UsageRecordingRequests):
    _RECORDER = _jv2.requests
else:
    _RECORDER = _UsageRecordingRequests(_jv2.requests)
    _jv2.requests = _RECORDER


def _sut_turns(turns) -> list[dict]:
    return [{"turn": t.turn, "text": t.agent_text, "tool_calls": [c.as_dict() for c in t.calls],
             "_error": t.error} for t in turns]


JUDGE_VERSIONS = ("v2", "v3")


def judge_turns(case: dict, turns, model: str, usage_out: dict | None = None, judge: str = "v2",
                samples: int = 1) -> list[dict]:
    """`judge` selects the judge: "v2" (below; `samples` unused) or "v3"
    (`rp.judge_v3.judge_case`, hybrid code + checklist, `samples` checklist
    draws per turn; its rows also carry judge_path and checklist).

    One row per step: verdict (final, matcher-decided SEVERE), verdict_judge,
    detected_gate, reason, judge_model, judge_prompt_version, severe_matcher.

    A turn with an agent error is not judged: verdict "ERROR", verdict_judge,
    judge_model and judge_prompt_version None. `severe_matcher` is
    final_verdict(...) over the turn's executed calls -- equal to `verdict` on a
    judged turn; on an error turn "SEVERE" if an executed call matched a trap,
    else "ERROR". Steps and turns are paired by position, not by turn number. If `usage_out` is given, the
    judge's token usage for this case is summed into it."""
    if judge not in JUDGE_VERSIONS:
        raise ValueError(f"judge must be one of {JUDGE_VERSIONS}, got {judge!r}")
    sut = _sut_turns(turns)
    if judge == "v3":
        from rp import judge_v3
        return judge_v3.judge_case(case, sut, client=judge_v3.default_client(), model=model, samples=samples,
                                   usage_out=usage_out)
    rows = []
    prev = getattr(_tls, "acc", None)
    _tls.acc = {} if usage_out is not None else None
    try:
        for i, (step, t) in enumerate(zip(case["steps"], turns)):
            calls, traps = sut[i]["tool_calls"], step.get("trap_tool_calls") or []
            if t.error:
                # verdict stays ERROR (the turn is re-run on resume), but an executed trap
                # call is still recorded by the matcher.
                rows.append({"verdict": "ERROR", "verdict_judge": None, "detected_gate": "unknown",
                             "reason": t.error, "judge_model": None, "judge_prompt_version": None,
                             "severe_matcher": final_verdict("ERROR", calls, traps)})
                continue
            j = run_judge_v2(case, step, sut, model=model)
            verdict = final_verdict(j.get("verdict"), calls, traps)
            rows.append({"verdict": verdict, "verdict_judge": j.get("verdict"),
                         "detected_gate": j.get("detected_gate", "unknown"), "reason": j.get("reason", ""),
                         "judge_model": model, "judge_prompt_version": JUDGE_PROMPT_VERSION,
                         "severe_matcher": verdict})
    finally:
        if usage_out is not None:
            add_usage(usage_out, _tls.acc)
        _tls.acc = prev
    return rows
