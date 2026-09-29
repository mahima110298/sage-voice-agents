#!/usr/bin/env python3
"""Scheduler for the 52 locked prereg test runs (Task 29 = plan Task 20).

Launches `python3 -m rp.runner --split test --agent A --config C --pass K
--workers 1 --judge v3 --case-ids-file data/test_subset_v3.json` subprocesses,
one per (agent, config, pass) row of the run list in
docs/prereg/2026-analysis-plan.md §8 (52 runs total; hard-coded below since
the prereg wins over the plan text).

Concurrency: <= 14 processes overall (`--max-procs`), with per-family caps
(OpenAI realtime <= 2, Gemini Live <= 3 combined, qwen starts at <= 10). Any
run's nonzero exit or 429 burst backs that run off (60s sleep before the next
fill); the qwen cap itself is only ever cut (by 2, floor 4) when the failing
or bursting run is a qwen run -- a Gemini/OpenAI/voxtral/open-cascade failure
never throttles qwen. Spare slots (e.g. when Gemini/OpenAI caps are full) are
filled out of strict order with whatever runnable row comes next (voxtral /
open-cascade / qwen), so the scheduler is a priority queue, not a strict FIFO.

Order: pass 1 of every row before pass 2 of any row; within a pass, the full
grid first, then the light grid, then the ablations (matches the prereg
table's grouping, but pass 1/2 are interleaved with each other in the prereg
table itself -- this scheduler separates them so all of pass 1 lands first).

The runner is resumable (skips completed cases) and enforces its own $400
total cost cap, so re-launching (this scheduler's error sweep, or a manual
re-run) is always safe.

Every run's stdout/stderr goes to
results/rp/test/_logs/<agent>__<config>__pass<K>.log

Budget guards (hard, checked before launch and every 15 minutes while
running): sum known spend from every results/rp*/cost_log.jsonl by provider
(own cost_log.jsonl counted once -- see sibling_cost_logs), and query
OpenRouter's remaining credit balance. Below $3 remaining, stop launching new
runs (every run's judge call is OpenRouter, so this means all new runs) and
let running ones finish. Below $1, stop launching new runs AND send SIGTERM
to every currently running run, then exit with a clear message (see
BudgetGuard.apply / BudgetExhausted) -- the runner resumes cleanly from a
SIGTERM (see the note on rp.results_io.append_rows / rp.runner.done_cases in
BudgetGuard's docstring below), so this is a hard stop, not a pause. The
runner's own $400 cap is untouched and stays in force.

The guard fails safe, never open: if OPENROUTER_API_KEY is missing or the
very first credit check fails, main() refuses to start at all (exit 2) rather
than launch blind. Once running, if a later check fails twice in a row,
stop_new is forced on (no new launches; runs already in flight keep going)
until a check succeeds again, at which point it's cleared -- unless the fresh
balance itself says to stop. Every failure is logged to stderr.

Every ~30s poll cycle also tails each running run's log since the last read
offset and counts new HTTP 429s (Tail429Watcher); a burst (>= 5 new 429s)
feeds Scheduler.note_429_burst the same way a hard failure would, even though
the process never exited (the runner's own retry logic can absorb 429s
without ever returning nonzero).

After every row has been attempted once, this script does one error sweep
(re-launches every row once more -- the runner resumes and only re-does
errored/missing cases), then writes results/rp/test/ERRORS.md and a report:
turns done / errors / cost per run, spend by provider, wall-clock, and a
check that every row carries the prereg's judge_rules_hash.

Use `--dry-run` to print the launch order (and per-family caps) without
starting anything -- this is what got exercised for task-29-report.md; the
real grid is started by the controller in a persistent terminal, wrapped in
`caffeinate -dimsu -w <scheduler pid>` (done automatically here, skipped in
dry-run mode, and skipped again for the offline tests since they inject their
own fake launcher/sleep and never take this code path).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
STUDY = SCRIPTS_DIR.parent  # multiturn_study/
sys.path.insert(0, str(STUDY))

from env_utils import load_env_file  # noqa: E402

RESULTS_ROOT = STUDY / "results" / "rp"
TEST_DIR = RESULTS_ROOT / "test"
LOG_DIR = TEST_DIR / "_logs"
ERRORS_MD = TEST_DIR / "ERRORS.md"
CASE_IDS_FILE = "data/test_subset_v3.json"

# Prereg (docs/prereg/2026-analysis-plan.md §6), value at freeze time.
EXPECTED_JUDGE_RULES_HASH = "584bf48735e310335b7df5440cdc4a2b17db71ca71d1511d504d06dd32ab243e"

# ----------------------------------------------------------------- run list

# docs/prereg/2026-analysis-plan.md §8, "Full grid" table.
FULL_GRID = [(a, c) for a in ("gemini-3.8-live", "qwen3.8-omni-flash")
             for c in ("B0", "B2", "B3", "B4", "SAGE")]
# §8, "Light grid" table.
LIGHT_GRID = [(a, c) for a in ("gpt-realtime-2.1-mini", "gemini-3.1-flash-live-preview",
                                "voxtral-small-24b", "open-cascade")
              for c in ("B0", "B2", "SAGE")]
# §8, "Ablations" table.
ABLATIONS = [("qwen3.8-omni-flash", c) for c in
             ("SAGE-no_gate_types", "SAGE-no_verifier", "SAGE-no_credential_check", "SAGE-no_policy")]

RUN_GROUPS = FULL_GRID + LIGHT_GRID + ABLATIONS  # 10 + 12 + 4 = 26 (agent, config) pairs


def build_run_list() -> list[tuple[str, str, int]]:
    """52 (agent, config, pass) rows: pass 1 of every row (full grid, then
    light grid, then ablations) before pass 2 of the same order. Task-29
    brief's "Order" line, which differs from the prereg table's own
    (pass-interleaved) row order."""
    return [(a, c, p) for p in (1, 2) for a, c in RUN_GROUPS]


RUN_LIST = build_run_list()
assert len(RUN_LIST) == 52, f"expected 52 runs, got {len(RUN_LIST)}"


def filter_run_list(run_list, agents=None):
    """Rows of run_list whose agent is in `agents` (all rows when None), order kept.
    Staged launch (student 2026-09-28): open/cheap model first, others later."""
    if not agents:
        return list(run_list)
    unknown = set(agents) - {a for a, _, _ in run_list}
    if unknown:
        raise SystemExit(f"run_test_grid: unknown agent(s): {sorted(unknown)}")
    return [r for r in run_list if r[0] in set(agents)]

# ----------------------------------------------------------- concurrency caps

OPENAI_AGENTS = {"gpt-realtime-2.1-mini"}
GEMINI_LIVE_AGENTS = {"gemini-3.8-live", "gemini-3.1-flash-live-preview"}
QWEN_AGENTS = {"qwen3.8-omni-flash"}

OPENAI_CAP = 2
GEMINI_LIVE_CAP = 3
QWEN_CAP_START = 10
QWEN_CAP_MIN = 4
QWEN_CAP_STEP = 2

DEFAULT_MAX_PROCS = 14
DEFAULT_BACKOFF_SECONDS = 60
DEFAULT_POLL_INTERVAL = 30  # ~30s poll cycle: budget check gate + 429-burst log tail
BUDGET_CHECK_INTERVAL = 15 * 60
STOP_NEW_THRESHOLD_USD = 3.0
STOP_ALL_THRESHOLD_USD = 1.0
BURST_429_THRESHOLD = 5

# Which provider bills each agent's own LLM/ASR call. Judge and verifier are
# OpenRouter for every agent regardless of this table (task-29 brief).
AGENT_PROVIDER = {
    "gemini-3.8-live": "google",
    "gemini-3.1-flash-live-preview": "google",
    "gemini-2.5-flash-native-audio-latest": "google",
    "gpt-realtime-2.1-mini": "openai",
    "gpt-realtime-2.1": "openai",
    "gpt-realtime-1.5": "openai",
    "qwen3.8-omni-flash": "openrouter",
    "voxtral-small-24b": "openrouter",
    "open-cascade": "groq",
}


def family(agent: str) -> str | None:
    if agent in OPENAI_AGENTS:
        return "openai"
    if agent in GEMINI_LIVE_AGENTS:
        return "gemini_live"
    if agent in QWEN_AGENTS:
        return "qwen"
    return None


@dataclass(frozen=True)
class RunSpec:
    agent: str
    config: str
    pass_: int

    @property
    def key(self) -> str:
        return f"{self.agent}__{self.config}__pass{self.pass_}"

    def command(self) -> list[str]:
        return [sys.executable, "-m", "rp.runner", "--split", "test",
                "--agent", self.agent, "--config", self.config, "--pass", str(self.pass_),
                "--workers", "1", "--judge", "v3", "--case-ids-file", CASE_IDS_FILE]


# --------------------------------------------------------------- scheduler

class Scheduler:
    """Greedy priority-queue scheduler over `run_list` (in RUN_LIST order by
    default). `launch_fn(spec, log_path) -> proc` starts one run and returns an
    object with `.poll() -> int | None` (same protocol as subprocess.Popen);
    swap it for a fake in tests. `sleep_fn` is called between polls (and for
    the backoff pause) so tests can drive time without a real clock.
    """

    def __init__(self, run_list, launch_fn, max_procs: int = DEFAULT_MAX_PROCS,
                 sleep_fn=time.sleep, backoff_seconds: int = DEFAULT_BACKOFF_SECONDS,
                 log_dir: Path = LOG_DIR, on_reap=None):
        self.pending: list[RunSpec] = [RunSpec(a, c, p) for a, c, p in run_list]
        self.launch_fn = launch_fn
        self.max_procs = max_procs
        self.sleep_fn = sleep_fn
        self.backoff_seconds = backoff_seconds
        self.log_dir = Path(log_dir)
        self.on_reap = on_reap  # optional callback(spec, exit_code) for external bookkeeping/tests
        self.qwen_cap = QWEN_CAP_START
        self.running: dict[str, tuple[RunSpec, object]] = {}
        self.launch_order: list[str] = []
        self.completed: list[tuple[RunSpec, int]] = []
        self.backoffs = 0

    # -- decisions (pure enough to reason about / unit test indirectly) --

    def _family_running_count(self, fam: str) -> int:
        return sum(1 for spec, _ in self.running.values() if family(spec.agent) == fam)

    def _family_cap(self, fam: str) -> int | None:
        return {"openai": OPENAI_CAP, "gemini_live": GEMINI_LIVE_CAP, "qwen": self.qwen_cap}.get(fam)

    def can_launch(self, spec: RunSpec) -> bool:
        if len(self.running) >= self.max_procs:
            return False
        fam = family(spec.agent)
        if fam is not None:
            cap = self._family_cap(fam)
            if cap is not None and self._family_running_count(fam) >= cap:
                return False
        return True

    def _launch(self, spec: RunSpec) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"{spec.key}.log"
        proc = self.launch_fn(spec, log_path)
        self.running[spec.key] = (spec, proc)
        self.launch_order.append(spec.key)

    def _fill_slots(self) -> None:
        """Scan `pending` front-to-back, launching every row that doesn't
        violate a cap; a blocked row is skipped (not removed) so a later,
        launchable row can fill the spare slot instead (voxtral/open-cascade
        filling in while Gemini/OpenAI/qwen are capped out)."""
        i = 0
        while i < len(self.pending):
            if len(self.running) >= self.max_procs:
                break
            spec = self.pending[i]
            if self.can_launch(spec):
                self.pending.pop(i)
                self._launch(spec)
                # list shifted left under index i; rescan from the same index
            else:
                i += 1

    def _on_failure(self, spec: RunSpec, code: int) -> None:
        """Back off `backoff_seconds` for any run's nonzero exit or 429 burst.
        The qwen cap itself is only ever cut here when `spec` is a qwen run --
        a Gemini/OpenAI/voxtral/open-cascade failure backs off (delays the
        next fill) but never throttles qwen's throughput."""
        is_qwen = spec.agent in QWEN_AGENTS
        if is_qwen:
            new_cap = max(QWEN_CAP_MIN, self.qwen_cap - QWEN_CAP_STEP)
            print(f"run_test_grid: {spec.key} exited {code}; backing off {self.backoff_seconds}s, "
                  f"cutting qwen cap {self.qwen_cap} -> {new_cap}", file=sys.stderr)
        else:
            print(f"run_test_grid: {spec.key} exited {code}; backing off {self.backoff_seconds}s "
                  f"(non-qwen run; qwen cap unchanged)", file=sys.stderr)
        self.backoffs += 1
        self.sleep_fn(self.backoff_seconds)
        if is_qwen:
            self.qwen_cap = max(QWEN_CAP_MIN, self.qwen_cap - QWEN_CAP_STEP)

    def note_429_burst(self, spec: RunSpec) -> None:
        """Called from outside (real-mode log tailing) when a running process's
        log shows a burst of HTTP 429s without having exited -- same backoff as
        a hard failure, but the process is left running."""
        self._on_failure(spec, code=-1)

    def _reap(self) -> None:
        for key in list(self.running):
            spec, proc = self.running[key]
            code = proc.poll()
            if code is not None:
                del self.running[key]
                self.completed.append((spec, code))
                if self.on_reap is not None:
                    self.on_reap(spec, code)
                if code != 0:
                    self._on_failure(spec, code)

    def run(self, poll_interval: float = DEFAULT_POLL_INTERVAL) -> list[tuple[RunSpec, int]]:
        while self.pending or self.running:
            self._reap()
            self._fill_slots()
            if self.pending and not self.running:
                raise RuntimeError(
                    "run_test_grid: no run is launchable and none is running -- "
                    "a family cap must exceed max_procs or every remaining row is blocked forever")
            if self.pending or self.running:
                self.sleep_fn(poll_interval)
        return self.completed


# ------------------------------------------------------------------ dry run

class _InstantProcess:
    """Fake process used only for `--dry-run`: reports "still running" on its
    first poll and "exited 0" on every poll after, so the Scheduler's ordinary
    reap/fill loop produces one wave per (reap, fill) cycle -- a deterministic,
    zero-real-time simulation of the launch order under the real caps."""

    def __init__(self):
        self._polled = False

    def poll(self):
        if not self._polled:
            self._polled = True
            return None
        return 0


def simulate_waves(run_list=RUN_LIST, max_procs: int = DEFAULT_MAX_PROCS) -> list[list[str]]:
    """The launch order the real scheduler would use, grouped into waves of
    simultaneous launches, assuming every run takes the same wall-clock time
    (a simplification -- real runs won't clear in lockstep, so real launches
    interleave more than this, but the *order and caps* are exactly the
    scheduler's real logic)."""
    waves: list[list[str]] = []

    def launch_fn(spec, log_path):
        return _InstantProcess()

    sched = Scheduler(run_list, launch_fn, max_procs=max_procs, sleep_fn=lambda s: None)
    before = 0
    while sched.pending or sched.running:
        sched._reap()
        started_before = set(sched.launch_order)
        sched._fill_slots()
        new_this_cycle = [k for k in sched.launch_order if k not in started_before]
        if new_this_cycle:
            waves.append(new_this_cycle)
        if sched.pending and not sched.running:
            raise RuntimeError("simulate_waves: deadlock")
    return waves


# ------------------------------------------------------------------ budget

def parse_credits(payload: dict) -> float:
    """`GET https://openrouter.ai/api/v1/credits` response -> remaining USD
    (total_credits - total_usage). Never logs the payload (it carries no key,
    but keep the habit)."""
    data = payload.get("data", payload)
    total = float(data.get("total_credits", 0.0))
    used = float(data.get("total_usage", 0.0))
    return total - used


def openrouter_remaining_credits(api_key: str, timeout: float = 10.0) -> float:
    req = urllib.request.Request("https://openrouter.ai/api/v1/credits",
                                  headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 -- fixed https host
        payload = json.loads(resp.read())
    return parse_credits(payload)


def budget_state(remaining_usd: float) -> tuple[bool, bool]:
    """(stop_new, stop_all) for a given OpenRouter remaining balance: stop_new
    (< $3) blocks new launches only; stop_all (< $1) additionally means every
    running run gets SIGTERM'd and the scheduler exits."""
    return remaining_usd < STOP_NEW_THRESHOLD_USD, remaining_usd < STOP_ALL_THRESHOLD_USD


def provider_of_component(agent: str, role: str) -> str:
    if role in ("verifier", "judge"):
        return "openrouter"
    return AGENT_PROVIDER.get(agent, "unknown")


def spend_by_provider(cost_log_paths) -> dict[str, float]:
    """Sum `known_cost_usd` per component across every results/rp*/cost_log.jsonl,
    bucketed by provider (task-29 brief's budget guard)."""
    from rp import results_io  # deferred: only needed once rp/ is on sys.path (see main())
    totals: dict[str, float] = {}
    for path in cost_log_paths:
        for rec in results_io.iter_raw(Path(path)):
            agent = rec.get("agent")
            for comp in rec.get("components", []):
                provider = provider_of_component(agent, comp.get("role", ""))
                totals[provider] = totals.get(provider, 0.0) + float(comp.get("known_cost_usd") or 0.0)
    return totals


def count_429(text: str) -> int:
    return text.count("429")


def sibling_cost_logs(root: Path = RESULTS_ROOT) -> list[Path]:
    """Every other results root's cost log next to `root` (results/rp*/cost_log.jsonl,
    e.g. results/rp_v3/cost_log.jsonl), excluding `root`'s own cost_log.jsonl --
    the glob `rp*` also matches `rp` itself, so without this exclusion the
    caller's own primary log (already passed explicitly in main()) would be
    counted twice. Mirrors rp.runner.sibling_cost_logs (rp/runner.py:340-346),
    which guards against exactly this for the same reason."""
    own = (root / "cost_log.jsonl").resolve()
    return sorted(p for p in root.parent.glob("rp*/cost_log.jsonl") if p.resolve() != own)


class Tail429Watcher:
    """Tails each running run's log file since the last read offset and
    reports which runs had a burst (>= `threshold` new "429" occurrences)
    since the previous check. Meant to be polled every ~30s cycle so a run
    that retries 429s internally (and never exits nonzero) still triggers
    `Scheduler.note_429_burst`. Offsets advance for every running key on every
    call regardless of whether that key's slice contained a burst, so a slow
    trickle of 429s across many polls doesn't accumulate into a false burst."""

    def __init__(self, log_dir: Path = LOG_DIR, threshold: int = BURST_429_THRESHOLD):
        self.log_dir = Path(log_dir)
        self.threshold = threshold
        self._offsets: dict[str, int] = {}

    def check(self, running: dict) -> list:
        bursts = []
        for key, (spec, _proc) in running.items():
            path = self.log_dir / f"{key}.log"
            if not path.exists():
                continue
            offset = self._offsets.get(key, 0)
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(offset)
                new_text = f.read()
                self._offsets[key] = f.tell()
            if count_429(new_text) >= self.threshold:
                bursts.append(spec)
        return bursts


class BudgetExhausted(RuntimeError):
    """Raised by BudgetGuard.apply once OpenRouter remaining < $1: every
    currently running run has already been sent SIGTERM and nothing more
    should launch (every run's judge call needs OpenRouter, so nothing can
    safely continue). Caller stops the whole scheduler -- no further
    launches, no error sweep."""


class BudgetGuard:
    """Wraps the OpenRouter credit check + $/provider spend sum. Checked
    before launch and every 15 minutes. `stop_new` (remaining < $3) blocks new
    launches and lets running ones finish. `stop_all` (remaining < $1) is a
    hard stop: `apply()` sends SIGTERM to every running proc and raises
    BudgetExhausted.

    Fails safe, never open: `ever_succeeded` stays False until a credit check
    actually succeeds (missing OPENROUTER_API_KEY counts as a failure); the
    caller (main()) refuses to start at all if the mandatory first check never
    succeeds, instead of launching as if the balance were fine. After the
    first success, two consecutive later failures force `stop_new = True`
    (new launches blocked; runs already in flight keep going) until a check
    succeeds again, at which point the fresh reading (`budget_state`) takes
    over -- clearing stop_new unless the balance itself now says to stop.
    Every failure is printed to stderr.

    Why SIGTERM is safe here: rp.runner writes one case's judged turns with a
    single `results_io.append_rows` call, then separately appends a
    cost_log.jsonl record (rp/runner.py `run_one`). If SIGTERM lands mid
    write(), the OS can interrupt the syscall and leave a torn final line;
    `results_io.iter_raw` already skips an unparseable trailing line (a torn
    write is exactly what its docstring anticipates: "the process was killed
    mid-write"), and `append_rows` starts the next append on a fresh line
    regardless. `runner.done_cases` only counts a case done once every one of
    its turns has a row with no error, so a case caught mid-write (partial or
    torn) is simply not "done" and is fully redone -- with the same case_id
    and turn numbers, its new rows become the last (and only complete) ones
    `load_rows` sees, so nothing is lost or double-counted in the results. One
    real gap: if SIGTERM lands *between* the results write and the cost_log
    append for the same case (both happen in `run_one`, in that order), that
    case's cost never gets logged even though it won't be re-run (its results
    row already exists) -- a small, one-case undercount of spend, not a
    results-correctness issue."""

    CONSECUTIVE_FAILURES_BEFORE_STOP_NEW = 2

    def __init__(self, api_key: str | None, sleep_fn=time.sleep, check_interval: float = BUDGET_CHECK_INTERVAL):
        self.api_key = api_key
        self.sleep_fn = sleep_fn
        self.check_interval = check_interval
        self.last_check = 0.0
        self.remaining_usd: float | None = None
        self.stop_new = False
        self.stop_all = False
        self.ever_succeeded = False
        self.consecutive_failures = 0

    def maybe_check(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self.last_check) < self.check_interval:
            return
        self.last_check = now
        if not self.api_key:
            print("run_test_grid: OPENROUTER_API_KEY is not set; treating as a failed credit check",
                  file=sys.stderr)
            self._on_check_failure()
            return
        try:
            self.remaining_usd = openrouter_remaining_credits(self.api_key)
        except (urllib.error.URLError, TimeoutError, ValueError) as e:
            print(f"run_test_grid: OpenRouter credit check failed ({e})", file=sys.stderr)
            self._on_check_failure()
            return
        self.consecutive_failures = 0
        self.ever_succeeded = True
        self.stop_new, self.stop_all = budget_state(self.remaining_usd)
        print(f"run_test_grid: OpenRouter remaining ${self.remaining_usd:.2f} "
              f"(stop_new={self.stop_new} stop_all={self.stop_all})", flush=True)

    def _on_check_failure(self) -> None:
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.CONSECUTIVE_FAILURES_BEFORE_STOP_NEW:
            if not self.stop_new:
                print(f"run_test_grid: {self.consecutive_failures} consecutive credit-check failures; "
                      f"forcing stop_new until a check succeeds again", file=sys.stderr)
            self.stop_new = True

    def apply(self, running: dict) -> None:
        """No-op unless stop_all is set; then SIGTERM every running proc and
        raise BudgetExhausted so the caller stops the whole scheduler."""
        if not self.stop_all:
            return
        for key, (spec, proc) in list(running.items()):
            pid = getattr(proc, "pid", None)
            if pid is None:
                continue
            try:
                os.kill(pid, signal.SIGTERM)
                print(f"run_test_grid: SIGTERM {spec.key} (pid {pid}) -- OpenRouter balance < "
                      f"${STOP_ALL_THRESHOLD_USD:.0f}", file=sys.stderr)
            except ProcessLookupError:
                pass
        remaining = f"${self.remaining_usd:.2f}" if self.remaining_usd is not None else "unknown"
        raise BudgetExhausted(
            f"OpenRouter remaining {remaining} < ${STOP_ALL_THRESHOLD_USD:.0f}: sent SIGTERM to every "
            f"running run and stopped launching new ones")


# ------------------------------------------------------------------ report

def run_error_count(agent: str, config: str, pass_: int) -> tuple[int, int]:
    """(turns_done, errors) for one run's pass file, from the last row per
    (case_id, turn) -- errors are rows with sut_error set or verdict == ERROR."""
    from rp import results_io
    path = TEST_DIR / agent / config / f"pass{pass_}.jsonl"
    rows = results_io.load_rows(path)
    errors = sum(1 for r in rows if r.get("sut_error") or r.get("verdict") == "ERROR")
    return len(rows), errors


def check_judge_rules_hash(agent: str, config: str, pass_: int) -> bool:
    from rp import results_io
    path = TEST_DIR / agent / config / f"pass{pass_}.jsonl"
    rows = results_io.load_rows(path)
    return all(r.get("judge_rules_hash") == EXPECTED_JUDGE_RULES_HASH for r in rows)


def write_errors_md(run_list=RUN_LIST, path: Path = ERRORS_MD) -> str:
    lines = ["# Test-grid error sweep", "", "| run | turns | errors | judge_rules_hash ok |", "|---|---|---|---|"]
    for agent, config, pass_ in run_list:
        turns, errors = run_error_count(agent, config, pass_)
        ok = check_judge_rules_hash(agent, config, pass_)
        lines.append(f"| {agent}__{config}__pass{pass_} | {turns} | {errors} | {'yes' if ok else 'NO'} |")
    text = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


# -------------------------------------------------------------------- main

def real_launch(spec: RunSpec, log_path: Path):
    log_f = open(log_path, "ab")
    return subprocess.Popen(spec.command(), stdout=log_f, stderr=subprocess.STDOUT, cwd=str(STUDY))


def start_caffeinate(pid: int):
    try:
        return subprocess.Popen(["caffeinate", "-dimsu", "-w", str(pid)])
    except FileNotFoundError:
        print("run_test_grid: caffeinate not found; sleep prevention not active", file=sys.stderr)
        return None


def make_poll_step(active: dict, budget: "BudgetGuard", watcher: "Tail429Watcher", base_sleep=time.sleep):
    """Builds the `sleep_fn` passed to every Scheduler main() creates (both the
    main pass and the error sweep). It always reads/acts on `active["sched"]`
    -- the scheduler main() currently considers "active" -- rather than
    whichever scheduler existed at closure-creation time, so budget and
    429-burst checks target the right scheduler's `running` dict even after
    main() switches from the main pass to the error sweep. (Round 1 had a bug
    here: the closure captured a fixed scheduler variable, so sweep-phase
    checks silently read the original, by-then-empty scheduler.)"""
    def poll_step(poll_interval):
        running = active["sched"].running
        budget.maybe_check()
        budget.apply(running)  # raises BudgetExhausted if remaining < $1
        for spec in watcher.check(running):
            active["sched"].note_429_burst(spec)
        base_sleep(poll_interval)
    return poll_step


def print_dry_run(max_procs: int, run_list=RUN_LIST) -> None:
    waves = simulate_waves(run_list, max_procs=max_procs)
    print(f"run_test_grid: dry run, max_procs={max_procs}, "
          f"caps: openai<={OPENAI_CAP} gemini_live<={GEMINI_LIVE_CAP} qwen starts<={QWEN_CAP_START}")
    print(f"{len(run_list)} runs in {len(waves)} simulated waves:\n")
    n = 0
    for i, wave in enumerate(waves, 1):
        print(f"wave {i} ({len(wave)}):")
        for key in wave:
            n += 1
            print(f"  {n:2d}. {key}")
    print(f"\nexact launch command (per row, from multiturn_study/):")
    print("  python3 -m rp.runner --split test --agent <AGENT> --config <CONFIG> "
          "--pass <PASS> --workers 1 --judge v3 --case-ids-file data/test_subset_v3.json")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print the launch order and exit; launches nothing")
    parser.add_argument("--max-procs", type=int, default=DEFAULT_MAX_PROCS)
    parser.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL)
    parser.add_argument("--backoff-seconds", type=float, default=DEFAULT_BACKOFF_SECONDS)
    parser.add_argument("--skip-error-sweep", action="store_true", help="for manual re-runs; the report step still runs")
    parser.add_argument("--agents", nargs="+", default=None,
                        help="only rows for these agents (staged launch); default all 52")
    args = parser.parse_args(argv)
    run_list = filter_run_list(RUN_LIST, args.agents)

    if args.dry_run:
        print_dry_run(args.max_procs, run_list)
        return 0

    load_env_file()
    api_key = os.environ.get("OPENROUTER_API_KEY")

    caffeinate = start_caffeinate(os.getpid())
    try:
        budget = BudgetGuard(api_key)
        budget.maybe_check(force=True)
        if not budget.ever_succeeded:
            print("run_test_grid: refusing to start -- no successful OpenRouter credit check "
                  "(OPENROUTER_API_KEY is missing, or the first check failed); the budget guard "
                  "cannot run blind on real money", file=sys.stderr)
            return 2

        watcher = Tail429Watcher(log_dir=LOG_DIR)
        # mutable holder: poll_step is shared by the main scheduler and the
        # error-sweep scheduler, and must always act on whichever one is
        # currently running, not whichever existed when it was defined.
        active = {"sched": None}
        poll_step = make_poll_step(active, budget, watcher)

        def gated_launch(spec, log_path):
            return real_launch(spec, log_path)

        def budget_ok(spec):
            return not (budget.stop_new or budget.stop_all)

        try:
            sched = Scheduler(run_list, gated_launch, max_procs=args.max_procs,
                               backoff_seconds=args.backoff_seconds, sleep_fn=poll_step)
            orig_can_launch = sched.can_launch
            sched.can_launch = lambda spec: orig_can_launch(spec) and budget_ok(spec)  # type: ignore[assignment]
            active["sched"] = sched
            sched.run(poll_interval=args.poll_interval)

            if not args.skip_error_sweep:
                print("run_test_grid: error sweep -- re-launching every row once (resume-only)", flush=True)
                sweep = Scheduler(run_list, gated_launch, max_procs=args.max_procs,
                                   backoff_seconds=args.backoff_seconds, sleep_fn=poll_step)
                sweep.can_launch = lambda spec: orig_can_launch(spec) and budget_ok(spec)  # type: ignore[assignment]
                active["sched"] = sweep
                sweep.run(poll_interval=args.poll_interval)
        except BudgetExhausted as e:
            print(f"run_test_grid: stopping -- {e}", file=sys.stderr)
            write_errors_md(run_list)
            return 2

        report = write_errors_md(run_list)
        print(report)
        totals = spend_by_provider([RESULTS_ROOT / "cost_log.jsonl", *sibling_cost_logs()])
        print("spend by provider:", json.dumps(totals, indent=2))
    finally:
        if caffeinate is not None:
            caffeinate.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
