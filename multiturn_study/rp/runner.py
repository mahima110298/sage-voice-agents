"""Resumable experiment runner: one agent x one config x one pass over a split.

  cd multiturn_study
  python3 -m rp.runner --split dev_v2 --agent gemini-3.1-flash-live-preview --config B0 --pass 1 \\
      [--limit N] [--case-id ID] [--workers 4]

Writes (all under results/rp/):
  {split}/{agent}/{config}/pass{K}.jsonl   one row per judged turn (ROW_FIELDS)
  {split}/{agent}/{config}/manifest.json   provenance of every invocation into pass1.jsonl
  {split}/{agent}/{config}/manifest.pass{K}.json   the same for pass K > 1
  cost_log.jsonl                           one record per case, global across all runs

Refuses to start (SystemExit, nothing run) when:
  - the split is locked (test) and the git tag `prereg` does not exist;
  - any turn of the selected cases lacks a cached ASR transcript (run
    `python3 -m rp.asr --split <split>` first; the runner never transcribes);
  - the config needs a verifier (SAGE, SAGE-no_* except SAGE-no_verifier, B4)
    and none is available;
  - OPENROUTER_API_KEY is not set (the judge and verifier need it);
  - a model the run bills is not in pricing.json (override: --allow-unpriced);
  - the output folder's manifest was written under different experiment
    settings (tool overlay, policy, judge, verifier, ASR, models, lock);
  - cumulative spend in cost_log.jsonl plus one case would exceed the $400 cap;
  - resuming a pass file whose first invocation ran at a different git SHA
    (override: --allow-code-change, recorded in the manifest invocation).

Resume: a case is done when every one of its turns has a row in the output
file and none of those rows has sut_error or a judge ERROR (last row per
(case_id, turn) wins; see rp.results_io). Every other case is re-run in full
and its new rows are APPENDED; old rows are kept as an audit trail.

Provenance: every row and every cost record carries `run_id` (uuid4 per
invocation, also in manifest.invocations[i]) and `git_sha`.

Cost cap: before each case starts, the runner re-reads cumulative known spend
from cost_log.jsonl under an exclusive fcntl lock (`cost_log.jsonl.lock`; cost
records are appended under the same lock, so concurrent runner processes see
each other's spend) and checks that spend + (cases in flight + 1) x max case
cost stays within $400. max case cost starts from the largest logged case for
the same agent, else a conservative default (DEFAULT_CASE_COST_USD). Otherwise
no new case starts and the run exits with code 3. The cap also counts every
sibling results root's cost log (results/rp*/cost_log.jsonl, e.g. the judge-v3
re-scores written by `rp.rejudge`), read without their locks (append-only).

Concurrency note: cases run concurrently (asyncio.Semaphore(workers)); the
judge runs in a thread pool. The verifier call inside the interceptor is
synchronous and runs on the event loop, so with workers > 1 a verifier call in
one case pauses the audio streaming of the others (can inflate latency_ms for
SAGE/B4 runs). Use --workers 1 when latency is the measured outcome.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import fcntl
import functools
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import uuid
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

STUDY = Path(__file__).resolve().parent.parent
REPO = STUDY.parent
RESULTS_DIR = STUDY / "results" / "rp"
PRICING_PATH = Path(__file__).resolve().parent / "pricing.json"
COST_CAP_USD = 400.0
EXIT_COST_CAP = 3
# Conservative per-case cost assumed before any case of that agent has been logged,
# so the very first cases are also guarded by the in-flight estimate.
DEFAULT_CASE_COST_USD = {"gpt-realtime-2.1": 0.60, "gemini-3.8-live": 0.60, "cascade": 0.10}
DEFAULT_SPEECH_CASE_COST_USD = 0.30

_SCRIPTS = str(STUDY / "scripts")
if _SCRIPTS not in sys.path:  # harness_defs, run_multiturn, judge_v2 live in scripts/
    sys.path.insert(0, _SCRIPTS)

from rp import asr, results_io  # noqa: E402
from rp.agents import AGENTS  # noqa: E402
from rp.cases import LOCKED, SPLITS, load_split  # noqa: E402
from rp.interceptors import CONFIGS, build_config  # noqa: E402
from rp.judge import JUDGE_VERSIONS, add_usage, judge_turns  # noqa: E402
from rp.prompts import BASE_PROMPT, POLICY_MD  # noqa: E402
from rp.tools import OVERLAY  # noqa: E402
from rp.types import TurnRecord  # noqa: E402
from rp.verifier import VERIFIER_MODEL, OpenRouterClient, Verifier  # noqa: E402
from run_multiturn import JUDGE_MODEL  # noqa: E402  (anthropic/claude-haiku-4.5, pinned)
from env_utils import load_env_file  # noqa: E402

ROW_FIELDS = (
    # case identity
    "case_id", "domain", "turn", "num_steps", "severity", "gate_expected", "policy_rule_id",
    # agent output
    "sut_text", "sut_tool_calls", "blocked", "user_asr", "latency_ms", "guard_ms", "verifier_calls",
    "usage", "sut_error", "safety_valve", "response_status",
    # judge output
    "verdict", "verdict_judge", "detected_gate", "reason", "judge_model", "judge_prompt_version",
    "severe_matcher", "judge_path", "checklist", "judge_rules_hash",
    # run identity / provenance
    "split", "agent", "config", "pass", "run_id", "git_sha",
)

# Manifest fields that define the experimental condition: resuming into a file
# written under different values would mix conditions, so the runner refuses.
MANIFEST_KEYS = ("split", "agent", "config", "pass", "agent_model", "tool_overlay_sha", "policy_sha",
                 "lock_sha", "judge_model", "judge_prompt_version", "judge_samples", "verifier_model", "asr_model",
                 "cascade_llm")


# ------------------------------------------------------------------ git / split gate

def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                              timeout=30).stdout
    except Exception:  # noqa: BLE001 -- no git: provenance fields stay empty
        return ""


def _has_tag(tag: str) -> bool:
    return bool(_git("tag", "--list", tag).strip())


def check_allowed(split: str) -> None:
    if split in LOCKED and not _has_tag("prereg"):
        raise SystemExit(f"refusing --split {split}: the git tag 'prereg' does not exist "
                         "(the locked test set is run only after pre-registration)")


# ------------------------------------------------------------------ preflight checks

def needs_verifier(config: str) -> bool:
    return config == "B4" or (config.startswith("SAGE") and config != "SAGE-no_verifier")


def make_interceptor(config: str, verifier_obj):
    """A FRESH interceptor for one case (B3 and SAGE keep per-conversation state)."""
    if needs_verifier(config) and verifier_obj is None:
        raise SystemExit(f"refusing to run config {config} without a verifier "
                         f"(it needs Verifier({VERIFIER_MODEL!r}))")
    return build_config(config, verifier_obj)


def check_asr_cache(cases: list[dict], split: str) -> None:
    missing = [c["_folder"] / f"input_turn{s['turn']}.wav" for c in cases for s in c["steps"]
               if not asr.has_cache(c["_folder"] / f"input_turn{s['turn']}.wav")]
    if missing:
        shown = "\n  ".join(str(p) for p in missing[:10])
        raise SystemExit(f"refusing to start: {len(missing)} turn(s) have no cached ASR transcript, e.g.\n  "
                         f"{shown}\nrun first:  python3 -m rp.asr --split {split}")


# agents whose own "ears" are Groq Whisper (rp.agents.cascade asr="groq-whisper")
WHISPER_AGENTS = ("open-cascade",)


def check_whisper_cache(cases: list[dict], split: str) -> None:
    missing = [c["_folder"] / f"input_turn{s['turn']}.wav" for c in cases for s in c["steps"]
               if not asr.has_whisper_cache(c["_folder"] / f"input_turn{s['turn']}.wav")]
    if missing:
        shown = "\n  ".join(str(p) for p in missing[:10])
        raise SystemExit(f"refusing to start: {len(missing)} turn(s) have no cached Whisper transcript, e.g.\n  "
                         f"{shown}\nrun first:  python3 -m rp.asr --whisper --split {split}")


def done_cases(path: Path, steps_by_case: dict) -> set[str]:
    """Cases whose every turn has a row (last row per (case_id, turn) wins) with no
    sut_error and no judge ERROR. `steps_by_case`: case_id -> number of turns
    (turns 1..n) or an iterable of turn numbers."""
    by_case: dict[str, dict] = {}
    for r in results_io.load_rows(path):
        by_case.setdefault(r.get("case_id"), {})[r.get("turn")] = r
    done = set()
    for cid, steps in steps_by_case.items():
        want = set(range(1, steps + 1)) if isinstance(steps, int) else set(steps)
        got = by_case.get(cid, {})
        if want and want <= set(got) and all(
                got[t].get("sut_error") is None and "ERROR" not in (got[t].get("verdict"), got[t].get("verdict_judge"))
                for t in want):
            done.add(cid)
    return done


# ------------------------------------------------------------------ pricing

_RATES = {
    "text_in": ("input_per_mtok",),
    "text_in_cached": ("cached_input_per_mtok", "input_per_mtok"),
    "audio_in": ("audio_input_per_mtok",),
    "audio_in_cached": ("cached_audio_input_per_mtok", "audio_input_per_mtok"),
    "text_out": ("output_per_mtok",),
    "audio_out": ("audio_output_per_mtok",),
}
_UNSPLIT = {"unsplit_in": ("input_per_mtok", "audio_input_per_mtok"),
            "unsplit_out": ("output_per_mtok", "audio_output_per_mtok")}


def load_prices(path: Path = PRICING_PATH) -> dict:
    return {k: v for k, v in json.loads(Path(path).read_text()).items() if not k.startswith("_")}


def _buckets(u: dict):
    """Map a flattened usage dict (any of: OpenAI Realtime, Gemini Live, OpenRouter
    chat) onto priced buckets. Returns (buckets, unknown_fields, notes, incomplete)."""
    b = dict.fromkeys(list(_RATES) + list(_UNSPLIT), 0)
    used, notes, unknown = set(), [], []

    def take(k):
        used.add(k)
        return u.get(k, 0)

    def detail_group(prefix, known):
        keys = [k for k in u if k.startswith(prefix)]
        for k in keys:
            if k[len(prefix):] not in known:
                used.add(k)
                if u[k]:
                    unknown.append(k)
        return bool(keys)

    incomplete = False
    for k in [k for k in u if k.startswith("missing_usage_")]:
        used.add(k)
        if u[k]:
            incomplete = True
            notes.append(f"{k}={u[k]}: usage not reported, cost unknown")

    if "input_tokens" in u or "output_tokens" in u:                     # OpenAI Realtime
        inp, out = take("input_tokens"), take("output_tokens")
        take("total_tokens")
        d = "input_token_details."
        if detail_group(d, {"text_tokens", "audio_tokens", "cached_tokens", "cached_tokens_details.text_tokens",
                            "cached_tokens_details.audio_tokens"}):
            text, audio = take(d + "text_tokens"), take(d + "audio_tokens")
            cached = take(d + "cached_tokens")
            ct, ca = take(d + "cached_tokens_details.text_tokens"), take(d + "cached_tokens_details.audio_tokens")
            if cached and not (ct or ca):
                notes.append("cached input without modality split: billed at uncached rate")
            b["text_in"] += text - ct
            b["text_in_cached"] += ct
            b["audio_in"] += audio - ca
            b["audio_in_cached"] += ca
            b["unsplit_in"] += max(0, inp - text - audio)
        else:
            b["unsplit_in"] += inp
        d = "output_token_details."
        if detail_group(d, {"text_tokens", "audio_tokens"}):
            text, audio = take(d + "text_tokens"), take(d + "audio_tokens")
            b["text_out"] += text
            b["audio_out"] += audio
            b["unsplit_out"] += max(0, out - text - audio)
        else:
            b["unsplit_out"] += out
    elif any(k in u for k in ("prompt_token_count", "response_token_count", "candidates_token_count")):  # Gemini
        take("total_token_count")
        for count_key, detail, tb, ab, rb in (
                ("prompt_token_count", "prompt_tokens_details.", "text_in", "audio_in", "unsplit_in"),
                ("response_token_count", "response_tokens_details.", "text_out", "audio_out", "unsplit_out"),
                ("candidates_token_count", "candidates_tokens_details.", "text_out", "audio_out", "unsplit_out")):
            n = take(count_key)
            detail_group(detail, {"TEXT", "AUDIO"})
            t, a = take(detail + "TEXT"), take(detail + "AUDIO")
            other = sum(u[k] for k in unknown if k.startswith(detail))
            b[tb] += t
            b[ab] += a
            b[rb] += max(0, n - t - a - other)
        b["text_out"] += take("thoughts_token_count")          # billed as output ("including thinking")
        b["text_in"] += take("tool_use_prompt_token_count")
        for k in [k for k in u if k.startswith(("tool_use_prompt_tokens_details.", "cache_tokens_details."))]:
            used.add(k)
        if take("cached_content_token_count"):
            notes.append("cached_content_token_count billed at uncached rate")
    elif "prompt_tokens" in u or "completion_tokens" in u:               # OpenRouter chat
        take("total_tokens")
        cached = take("prompt_tokens_details.cached_tokens")
        take("prompt_tokens_details.cache_write_tokens")
        take("completion_tokens_details.reasoning_tokens")               # subset of completion_tokens
        # audio-in chat (Task 16b): audio units are a subset of prompt/completion
        # tokens, billed at the audio rate (OpenRouter pricing.audio; Voxtral 1e-4
        # USD per unit); they are moved out of the text count, never double-counted.
        audio_in = take("prompt_tokens_details.audio_tokens")
        audio_out = take("completion_tokens_details.audio_tokens")
        detail_group("prompt_tokens_details.", {"cached_tokens", "cache_write_tokens", "audio_tokens"})
        detail_group("completion_tokens_details.", {"reasoning_tokens", "audio_tokens"})
        b["text_in"] += max(0, take("prompt_tokens") - cached - audio_in)
        b["text_in_cached"] += cached
        b["audio_in"] += audio_in
        b["text_out"] += max(0, take("completion_tokens") - audio_out)
        b["audio_out"] += audio_out
    for k in u:
        if k not in used and k != "cost" and not k.startswith("cost_details.") and u[k]:
            unknown.append(k)
    if b["unsplit_in"] or b["unsplit_out"]:
        notes.append("tokens without modality split priced at the higher of text/audio rate (upper bound)")
    return b, sorted(set(unknown)), notes, incomplete


def price_usage(usage: dict, model: str, prices: dict) -> dict:
    """Price one usage dict. cost_usd is None (never a silent zero) when the model
    is unpriced, a needed rate is null, a usage field is unknown, or usage was not
    reported; known_cost_usd is what could be priced."""
    u: dict = {}
    add_usage(u, usage or {})
    b, unknown, notes, incomplete = _buckets(u)
    entry = prices.get(model)
    known, unpriced = 0.0, []
    for bucket, n in b.items():
        if not n:
            continue
        if entry is None:
            unpriced.append(bucket)
            continue
        if bucket in _UNSPLIT:
            rates = [entry.get(k) for k in _UNSPLIT[bucket] if entry.get(k) is not None]
            rate = max(rates) if rates else None
        else:
            rate = next((entry.get(k) for k in _RATES[bucket] if entry.get(k) is not None), None)
        if rate is None:
            unpriced.append(bucket)
            continue
        known += n * rate / 1_000_000
    if entry is None and u:
        notes.append(f"model {model!r} not in pricing.json")
    ok = not (unknown or unpriced or incomplete)
    return {"cost_usd": known if ok else None, "known_cost_usd": known, "unknown_fields": unknown,
            "unpriced": unpriced, "notes": notes, "reported_cost_usd": u.get("cost")}


def cost_usd(usage: dict, model: str, prices: dict) -> float | None:
    return price_usage(usage, model, prices)["cost_usd"]


def sibling_cost_logs(root: Path) -> list[Path]:
    """Every other results root's cost log next to `root` (results/rp*/cost_log.jsonl,
    e.g. the judge-v3 re-score logs results/rp_v3/ and results/rp_v3s3/), so the
    $400 cap counts all spend, not only the runner's own log."""
    root = Path(root)
    own = (root / "cost_log.jsonl").resolve()
    return sorted(p for p in root.parent.glob("rp*/cost_log.jsonl") if p.resolve() != own)


def cumulative_spend(cost_log: Path) -> float:
    total = 0.0
    for r in results_io.iter_raw(cost_log):
        v = r.get("known_cost_usd", r.get("cost_usd"))
        total += float(v or 0.0)
    return total


@contextmanager
def _locked(cost_log: Path):
    """Exclusive fcntl lock on a sibling `<cost_log>.lock` file, shared by every
    runner process that writes the same cost log."""
    cost_log.parent.mkdir(parents=True, exist_ok=True)
    with open(cost_log.with_name(cost_log.name + ".lock"), "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def seed_max_case(cost_log: Path, agent: str) -> float:
    """Largest known per-case cost already logged for `agent`, else a conservative default."""
    seen = [float(r.get("known_cost_usd") or 0.0) for r in results_io.iter_raw(cost_log) if r.get("agent") == agent]
    if seen:
        return max(seen)
    return DEFAULT_CASE_COST_USD.get(agent, DEFAULT_SPEECH_CASE_COST_USD)


class CostGuard:
    """Cost cap. With `cost_log`, cumulative spend is re-read from the shared log
    (under its lock) before every case start and every record is appended under
    the same lock, so concurrent runner processes see each other's spend.
    Without `cost_log`, it tracks `cumulative` in memory only."""

    def __init__(self, cumulative: float = 0.0, cap: float = COST_CAP_USD, cost_log: Path | None = None,
                 max_case: float = 0.0, extra_logs=()):
        self.cumulative, self.cap, self.cost_log = cumulative, cap, cost_log
        self.extra_logs = [Path(p) for p in extra_logs]
        self.max_case, self.in_flight, self.stopped = max_case, 0, False
        if cost_log is not None:
            with _locked(cost_log):
                self.cumulative = self._spend()

    def _spend(self) -> float:
        """Own log (read under its lock by the caller) plus every extra log
        (append-only JSONL; a torn last line is skipped by iter_raw)."""
        return cumulative_spend(self.cost_log) + sum(cumulative_spend(p) for p in self.extra_logs)

    def _fits(self) -> bool:
        return (self.cumulative < self.cap
                and self.cumulative + (self.in_flight + 1) * self.max_case <= self.cap)

    def can_start(self) -> bool:
        if self.cost_log is None:
            return self._fits()
        with _locked(self.cost_log):
            self.cumulative = self._spend()
            return self._fits()

    def record(self, cost: float, rec: dict | None = None) -> None:
        """Add one case's known cost; with a log, append `rec` (stamped with the
        cumulative total) under the lock."""
        self.max_case = max(self.max_case, cost)
        if self.cost_log is None:
            self.cumulative += cost
            return
        with _locked(self.cost_log):
            self.cumulative = self._spend() + cost
            if rec is not None:
                results_io.append_rows(self.cost_log, [{**rec, "cumulative_known_usd": self.cumulative}])

    def over_cap(self) -> bool:
        return self.cumulative > self.cap


class UsageClient:
    """Wraps the verifier's OpenRouter client and sums every completion's usage,
    so verifier spend is attributed to the case that caused it."""

    def __init__(self, inner):
        self.inner, self.usage, self.calls = inner, {}, 0

    def complete(self, model, messages):
        r = self.inner.complete(model, messages)
        self.calls += 1
        if r.get("usage"):
            add_usage(self.usage, r["usage"])
        else:
            self.usage["missing_usage_responses"] = self.usage.get("missing_usage_responses", 0) + 1
        return r


# ------------------------------------------------------------------ manifest

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _versions() -> dict:
    out = {}
    for p in ("google-genai", "openai", "websockets", "requests"):
        try:
            out[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            out[p] = None
    return out


def build_manifest(args, agent_model: str | None) -> dict:
    from rp.agents.cascade import CASCADE_LLM
    lock = SPLITS["test"][0] / "LOCK.json"
    dirty_src = _git("status", "--porcelain", "--", "multiturn_study/rp", "multiturn_study/scripts",
                     "multiturn_study/tests").splitlines()
    dirty_data = _git("status", "--porcelain", "--untracked-files=no", "--", "multiturn_study/data").splitlines()
    return {
        "git_sha": _git("rev-parse", "HEAD").strip() or None,
        "dirty": bool(_git("status", "--porcelain", "--untracked-files=no").strip()),
        "dirty_study_files": dirty_src + dirty_data,
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "split": args.split, "agent": args.agent, "config": args.config, "pass": args.pass_,
        "agent_model": agent_model,
        "tool_overlay_sha": _sha256(json.dumps(OVERLAY, sort_keys=True).encode()),
        "policy_sha": _sha256(POLICY_MD.read_bytes()),
        "policy_path": str(POLICY_MD.relative_to(STUDY)),
        "lock_sha": _sha256(lock.read_bytes()) if args.split == "test" else None,
        "judge_model": JUDGE_MODEL, "judge_prompt_version": args.judge,
        "judge_samples": args.judge_samples if args.judge == "v3" else None,
        "verifier_model": VERIFIER_MODEL if needs_verifier(args.config) else None,
        "asr_model": asr.ASR_MODEL, "cascade_llm": CASCADE_LLM,
        "python": platform.python_version(), "packages": _versions(),
        "workers": args.workers, "argv": sys.argv[1:] if args.argv is None else args.argv,
        "run_id": args.run_id, "allow_code_change": bool(args.allow_code_change),
    }


def manifest_path(folder: Path, pass_: int) -> Path:
    """One manifest per pass file: pass 1 keeps the original name `manifest.json`
    (so existing results stay valid), pass K > 1 uses `manifest.pass{K}.json`."""
    return folder / ("manifest.json" if pass_ == 1 else f"manifest.pass{pass_}.json")


def write_manifest(folder: Path, args, agent_model: str | None = None) -> dict:
    """Write/extend manifest.json. Top-level fields describe the latest invocation
    (`created` keeps the first); `invocations` lists every invocation. Refuses if
    an experiment-defining field differs from the existing manifest."""
    folder.mkdir(parents=True, exist_ok=True)
    path = manifest_path(folder, args.pass_)
    cur = build_manifest(args, agent_model)
    old = json.loads(path.read_text()) if path.exists() else None
    if old:
        diff = [k for k in MANIFEST_KEYS if old.get(k) != cur.get(k)]
        if diff:
            raise SystemExit(f"refusing to resume into {folder}: manifest differs on {diff} "
                             f"({ {k: (old.get(k), cur.get(k)) for k in diff} }); use a new --pass")
        first_sha = (old.get("invocations") or [{}])[0].get("git_sha", old.get("git_sha"))
        if first_sha != cur["git_sha"] and not args.allow_code_change:
            raise SystemExit(f"refusing to resume into {folder}: git_sha {cur['git_sha']} differs from the pass "
                             f"file's first invocation ({first_sha}); the code changed mid-pass. Use a new "
                             "--pass, or --allow-code-change to resume anyway (recorded in the manifest)")
    inv = {k: cur[k] for k in ("run_id", "created", "git_sha", "dirty", "dirty_study_files", "allow_code_change",
                               "python", "packages", "workers", "argv")}
    man = {**cur, "created": old["created"] if old else cur["created"],
           "invocations": (old or {}).get("invocations", []) + [inv]}
    path.write_text(json.dumps(man, indent=2) + "\n")
    return man


# ------------------------------------------------------------------ one case

def _align(case: dict, turns: list) -> list:
    by_turn = {t.turn: t for t in turns}
    return [by_turn.get(s["turn"]) or TurnRecord(turn=s["turn"], error="runner_error: no record for turn")
            for s in case["steps"]]


def build_rows(case: dict, turns: list, judged: list[dict], run: dict) -> list[dict]:
    rows = []
    for step, t, j in zip(case["steps"], turns, judged):
        rows.append({
            "case_id": case["id"], "domain": case["domain"], "turn": step["turn"],
            "num_steps": case.get("num_steps", len(case["steps"])), "severity": step.get("severity"),
            "gate_expected": step.get("gate_expected"), "policy_rule_id": step.get("policy_rule_id"),
            "sut_text": t.agent_text, "sut_tool_calls": [c.as_dict() for c in t.calls],
            "blocked": [dict(b) for b in t.blocked], "user_asr": t.user_text, "latency_ms": t.latency_ms,
            "guard_ms": t.guard_ms, "verifier_calls": t.verifier_calls, "usage": dict(t.usage),
            "sut_error": t.error, "safety_valve": t.usage.get("safety_valve"),
            "response_status": t.usage.get("response_status"),
            "verdict": j.get("verdict"), "verdict_judge": j.get("verdict_judge"),
            "detected_gate": j.get("detected_gate"), "reason": j.get("reason"),
            "judge_model": j.get("judge_model"), "judge_prompt_version": j.get("judge_prompt_version"),
            "severe_matcher": j.get("severe_matcher"), "judge_path": j.get("judge_path"),
            "checklist": j.get("checklist"), "judge_rules_hash": j.get("judge_rules_hash"),
            **run,
        })
    return rows


def agent_usage(turns: list) -> dict:
    acc: dict = {}
    for t in turns:
        numeric = {k: v for k, v in (t.usage or {}).items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        if numeric:
            add_usage(acc, numeric)
        elif t.error is None:
            acc["missing_usage_turns"] = acc.get("missing_usage_turns", 0) + 1
    return acc


class Ctx:
    def __init__(self, args, agent_factory, judge_fn, verifier_client_factory, prices, out, cost_log,
                 agent_model, guard, git_sha=None):
        self.args, self.agent_factory, self.judge_fn = args, agent_factory, judge_fn
        self.verifier_client_factory, self.prices, self.out, self.cost_log = verifier_client_factory, prices, out, cost_log
        self.agent_model, self.guard = agent_model, guard
        self.run = {"split": args.split, "agent": args.agent, "config": args.config, "pass": args.pass_,
                    "run_id": args.run_id, "git_sha": git_sha}
        self.run_cost, self.finished, self.total = 0.0, 0, 0
        self.valves, self.errors, self.null_cost_cases = Counter(), 0, 0


async def run_one(case: dict, ctx: Ctx) -> None:
    adapter = ctx.agent_factory()
    vclient = ctx.verifier_client_factory() if needs_verifier(ctx.args.config) else None
    verifier = Verifier(VERIFIER_MODEL, client=vclient) if vclient is not None else None
    interceptor = make_interceptor(ctx.args.config, verifier)
    try:
        turns = await adapter.run(case, interceptor, BASE_PROMPT)
    except Exception as e:  # noqa: BLE001 -- adapters record errors themselves; this is a last resort
        turns = [TurnRecord(turn=s["turn"], error=f"runner_error: {str(e)[:200] or type(e).__name__}")
                 for s in case["steps"]]
    turns = _align(case, turns)
    judge_usage: dict = {}
    try:
        judged = await asyncio.to_thread(ctx.judge_fn, case, turns, JUDGE_MODEL, judge_usage)
    except Exception as e:  # noqa: BLE001 -- recorded as judge ERROR, so the case is re-run on resume
        judged = [{"verdict": "ERROR", "verdict_judge": "ERROR", "detected_gate": "unknown",
                   "reason": f"judge_exception: {str(e)[:200]}", "judge_model": JUDGE_MODEL,
                   "judge_prompt_version": ctx.args.judge, "severe_matcher": None} for _ in turns]
    rows = build_rows(case, turns, judged, ctx.run)
    results_io.append_rows(ctx.out, rows)

    comps = [{"role": "agent", "model": ctx.agent_model, "usage": agent_usage(turns)}]
    if vclient is not None:
        comps.append({"role": "verifier", "model": VERIFIER_MODEL, "usage": vclient.usage, "calls": vclient.calls})
    comps.append({"role": "judge", "model": JUDGE_MODEL, "usage": judge_usage})
    for c in comps:
        c.update(price_usage(c["usage"], c["model"], ctx.prices))
    known = sum(c["known_cost_usd"] for c in comps)
    complete = all(c["cost_usd"] is not None for c in comps)
    ctx.run_cost += known
    rec = {"ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **ctx.run,
           "case_id": case["id"], "components": comps, "cost_usd": known if complete else None,
           "known_cost_usd": known}
    ctx.guard.record(known, rec)

    ctx.finished += 1
    errs = sum(1 for t in turns if t.error)
    ctx.errors += bool(errs)
    for t in turns:
        if t.usage.get("safety_valve"):
            ctx.valves[t.usage["safety_valve"]] += 1
    verdicts = Counter(r["verdict"] for r in rows)
    flag = ""
    if not complete:
        ctx.null_cost_cases += 1
        flag = " (INCOMPLETE: see cost_log)"
        for c in comps:
            if c["cost_usd"] is None:
                print(f"warning: {case['id']} {c['role']} cost unknown: unknown_fields={c['unknown_fields']} "
                      f"unpriced={c['unpriced']} notes={c['notes']}", file=sys.stderr)
    print(f"[{ctx.finished}/{ctx.total}] {case['id']}: {dict(verdicts)} errors={errs} | case ${known:.4f}{flag}"
          f" | run ${ctx.run_cost:.4f} | cumulative ${ctx.guard.cumulative:.4f} / ${ctx.guard.cap:.0f}",
          flush=True)


async def run_all(cases: list[dict], ctx: Ctx, workers: int) -> None:
    sem = asyncio.Semaphore(workers)

    async def one(case):
        async with sem:
            if ctx.guard.stopped or not ctx.guard.can_start():
                ctx.guard.stopped = True
                return
            ctx.guard.in_flight += 1
            try:
                await run_one(case, ctx)
            finally:
                ctx.guard.in_flight -= 1
            if ctx.guard.over_cap():
                ctx.guard.stopped = True

    await asyncio.gather(*(one(c) for c in cases))


# ------------------------------------------------------------------ CLI

def read_case_ids(path: Path) -> list[str]:
    data = json.loads(Path(path).read_text())
    ids = data["case_ids"] if isinstance(data, dict) else data
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        raise SystemExit(f"{path}: expected a JSON list of case ids or {{\"case_ids\": [...]}}")
    return ids


def select_cases(cases: list[dict], ids: list[str], split: str) -> list[dict]:
    """The split's cases whose id is in `ids`, in split order; refuses unknown ids."""
    known = {c["id"] for c in cases}
    missing = [i for i in ids if i not in known]
    if missing:
        raise SystemExit(f"case ids not in split {split}: {missing}")
    wanted = set(ids)
    return [c for c in cases if c["id"] in wanted]


def parse_args(argv):
    p = argparse.ArgumentParser(prog="python3 -m rp.runner", description=__doc__.split("\n\n")[0])
    p.add_argument("--split", required=True, choices=sorted(SPLITS))
    p.add_argument("--agent", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--pass", dest="pass_", type=int, required=True)
    p.add_argument("--limit", type=int, default=None, help="only the first N cases of the split")
    p.add_argument("--case-id", default=None)
    p.add_argument("--case-ids-file", default=None,
                   help="JSON file with a list of case ids (or {\"case_ids\": [...]}): run only those cases")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--judge", choices=JUDGE_VERSIONS, default="v2",
                   help="judge version (v2 = scripts/judge_v2.py; v3 = rp.judge_v3 hybrid). Default v2 until the "
                        "pre-registration fixes it; recorded in the manifest and in every row")
    p.add_argument("--judge-samples", type=int, default=1,
                   help="v3 only: checklist draws per turn (>1: temperature 0.7, per-field majority)")
    p.add_argument("--allow-unpriced", action="store_true",
                   help="start even if a billed model has no pricing.json entry (cost cap then undercounts)")
    p.add_argument("--allow-code-change", action="store_true",
                   help="resume a pass file although HEAD differs from its first invocation's git_sha")
    a = p.parse_args(argv)
    a.argv = list(argv) if argv is not None else None
    a.run_id = uuid.uuid4().hex
    return a


def main(argv=None, *, agent_factory=None, judge_fn=None, verifier_client_factory=None, prices=None,
         results_dir: Path | None = None) -> int:
    args = parse_args(argv)
    if args.agent not in AGENTS:
        raise SystemExit(f"unknown --agent {args.agent!r}; choose from {sorted(AGENTS)}")
    if args.config not in CONFIGS:
        raise SystemExit(f"unknown --config {args.config!r}; choose from {list(CONFIGS)}")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    check_allowed(args.split)

    if args.case_id and args.case_ids_file:
        raise SystemExit("--case-id and --case-ids-file are mutually exclusive")
    cases = load_split(args.split, only_id=args.case_id)
    if args.case_id and not cases:
        raise SystemExit(f"no case {args.case_id!r} in split {args.split}")
    if args.case_ids_file:
        cases = select_cases(cases, read_case_ids(Path(args.case_ids_file)), args.split)
    if args.limit is not None:
        cases = cases[:args.limit]
    check_asr_cache(cases, args.split)
    if args.agent in WHISPER_AGENTS:
        check_whisper_cache(cases, args.split)

    load_env_file()
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("refusing to start: OPENROUTER_API_KEY is not set (judge and verifier need it)")
    agent_factory = agent_factory or AGENTS[args.agent]
    if args.judge_samples < 1:
        raise SystemExit("--judge-samples must be >= 1")
    judge_fn = judge_fn or functools.partial(judge_turns, judge=args.judge, samples=args.judge_samples)
    verifier_client_factory = verifier_client_factory or (lambda: UsageClient(OpenRouterClient()))
    if needs_verifier(args.config) and verifier_client_factory() is None:
        raise SystemExit(f"refusing to run config {args.config} without a verifier")
    prices = load_prices() if prices is None else prices
    agent_model = getattr(agent_factory(), "model", None)

    billed = [agent_model, JUDGE_MODEL] + ([VERIFIER_MODEL] if needs_verifier(args.config) else [])
    unpriced = [m for m in billed if m not in prices or all(
        prices[m].get(k) is None for k in ("input_per_mtok", "output_per_mtok", "audio_input_per_mtok",
                                            "audio_output_per_mtok"))]
    if unpriced and not args.allow_unpriced:
        raise SystemExit(f"refusing to start: no prices for {unpriced} in {PRICING_PATH.name} "
                         "(the cost cap could not track them; --allow-unpriced to override)")

    root = Path(results_dir) if results_dir is not None else RESULTS_DIR
    cost_log = root / "cost_log.jsonl"
    guard = CostGuard(cost_log=cost_log, max_case=seed_max_case(cost_log, args.agent),
                      extra_logs=sibling_cost_logs(root))
    if not guard.can_start():
        raise SystemExit(f"refusing to start: cumulative spend ${guard.cumulative:.2f} in {cost_log} plus one "
                         f"case (~${guard.max_case:.2f}) would exceed the ${COST_CAP_USD:.0f} cap "
                         "-- ask the student before continuing")

    folder = root / args.split / args.agent / args.config
    out = folder / f"pass{args.pass_}.jsonl"
    man = write_manifest(folder, args, agent_model)
    done = done_cases(out, {c["id"]: [s["turn"] for s in c["steps"]] for c in cases})
    todo = [c for c in cases if c["id"] not in done]
    print(f"{args.split}/{args.agent}/{args.config}/pass{args.pass_}: {len(cases)} cases selected, "
          f"{len(done)} already done, {len(todo)} to run (workers={args.workers}); "
          f"cumulative spend ${guard.cumulative:.4f} / ${COST_CAP_USD:.0f}", flush=True)

    ctx = Ctx(args, agent_factory, judge_fn, verifier_client_factory, prices, out, cost_log, agent_model, guard,
              git_sha=man["git_sha"])
    ctx.total = len(todo)
    asyncio.run(run_all(todo, ctx, args.workers))

    print(f"finished {ctx.finished}/{len(todo)} cases; cases with agent errors: {ctx.errors}; "
          f"safety valves: {dict(ctx.valves) or 'none'}; cases with unknown cost: {ctx.null_cost_cases}; "
          f"run ${ctx.run_cost:.4f}; cumulative ${guard.cumulative:.4f} / ${COST_CAP_USD:.0f}")
    if guard.stopped:
        what = (f"cap exceeded by in-flight cases (spent ${guard.cumulative:.2f})" if guard.over_cap()
                else f"stopped before exceeding the cap (spent ${guard.cumulative:.2f})")
        print(f"COST CAP ${COST_CAP_USD:.0f}: {what}; {len(todo) - ctx.finished} case(s) not started "
              "-- ask the student before continuing", file=sys.stderr)
        return EXIT_COST_CAP
    return 0


if __name__ == "__main__":
    sys.exit(main())
