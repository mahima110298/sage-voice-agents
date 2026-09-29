"""Turn `results/rp/{split}/...` rows into the thesis's LaTeX result tables
and pgfplots CSV data.

Writes exactly the files `thesis_v2/main.tex` `\\input`s / `\\pgfplotstableread`s
(see task-22-brief.md for the full list). Each table replaces one `\\PENDING`
skeleton table already committed in `main.tex`:

  tables/rq1_main.tex      -> tab:rq1   (RQ1, primary judge)
  tables/rq1_judge2.tex    -> tab:rq1   (RQ1, second judge -- same shape, sec:5-validity)
  tables/rq2_per_agent.tex -> tab:rq2
  tables/rq3_cost.tex      -> tab:rq3
  tables/rq4_ablation.tex  -> tab:rq4
  tables/per_domain.tex    -> tab:domain
  tables/human_validation.tex -> tab:confusion (Task 23 fills this in; report.py
                                  writes the skeleton with descriptive counts only)

No pandas dependency: everything is plain dicts, lists, and string formatting.
One function per output; `main()` writes all of them. Every table/CSV is
written even when its underlying rows are entirely missing (e.g. a fresh
dev_v2 smoke run, or before the test split is unlocked): missing cells render
as an em dash, never a crash (see `_safe_load_runs`).

Every percentage cell is `n/N = x.x% [lo, hi]` (Wilson interval, turn-level,
descriptive). Every between-config difference is a case-cluster bootstrap
diff (95% CI) plus a case-level permutation p-value; the primary-outcome
p-values (gate-type error, under-gating) are Holm-adjusted across the primary
comparison family, whose definition (`pooled` vs `per_agent`) is a CLI flag
because the pre-registration (Task 19) will fix it -- see `--family`.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from rp.interceptors import CONFIGS
from rp.analysis import metrics as _metrics
from rp.analysis.metrics import RESULTS_ROOT, load_errors, load_runs, turn_counts, turn_flags
from rp.analysis.stats import cluster_bootstrap_diff, holm, perm_test, wilson

REPO = Path(__file__).resolve().parent.parent.parent  # multiturn_study/

EMDASH = "\u2014"

# Spec section 5 (Revision 2026-09-28) roster, in tab:rq2 row order: the two
# flagship agents (one closed, one open weights) run the full grid (B0-B4, SAGE);
# the other four run only the light grid (B0, B2, SAGE); the ablations run on
# ABLATION_AGENT. rp.agents.AGENTS may hold more (pilot / dropped) adapters;
# only these six are reported.
FULL_GRID_AGENTS = ("gemini-3.8-live", "qwen3.8-omni-flash")
LIGHT_GRID_AGENTS = ("gpt-realtime-2.1-mini", "gemini-3.1-flash-live-preview",
                     "voxtral-small-24b", "open-cascade")
ABLATION_AGENT = "qwen3.8-omni-flash"
# Derived from CONFIGS (rp.interceptors) rather than re-listed, so the config
# set can't drift from the study's actual interceptor roster.
FULL_CONFIGS = tuple(c for c in CONFIGS if not c.startswith("SAGE-no_"))  # B0..B4, SAGE
ABLATION_CONFIGS = tuple(c for c in CONFIGS if c.startswith("SAGE-no_"))
# Task 25: SAGE now carries the policy prompt (spec section 6), and the
# enforcement-only SAGE becomes the SAGE-no_policy ablation. The default RQ4
# table/Holm family covers 4 ablations; SAGE-no_routing and SAGE-no_exemption
# remain supported by CONFIGS/build_config (ABLATION_CONFIGS above still lists
# all of them) but are no longer part of the default table.
RQ4_CONFIGS = ("SAGE-no_gate_types", "SAGE-no_verifier", "SAGE-no_credential_check",
              "SAGE-no_policy")
LIGHT_CONFIGS = ("B0", "B2", "SAGE")
BASELINES = tuple(c for c in FULL_CONFIGS if c != "SAGE")  # B0..B4
PRIMARY_METRICS = ("wrong_type", "under")  # gate-type error, under-gating

# Row order for tab:rq2 (and every per-agent table/CSV).
AGENT_ORDER = FULL_GRID_AGENTS + LIGHT_GRID_AGENTS
RQ2_PREDICTION_MIN_WINS = 5  # spec: SAGE ranks first on >= 5 of the 6 agents
AGENT_LABEL = {
    "gemini-3.8-live": "Gemini 3.8 Live",
    "gemini-3.1-flash-live-preview": "Gemini 3.1 Flash Live",
    "gemini-2.5-flash-native-audio-latest": "Gemini 2.5 Flash Native Audio",
    "gpt-realtime-2.1": "GPT Realtime 2.1",
    "gpt-realtime-2.1-mini": "GPT Realtime 2.1 Mini",
    "gpt-realtime-1.5": "GPT Realtime 1.5",
    "cascade": "Cascade",
    "qwen3.8-omni-flash": "Qwen3.8 Omni Flash",
    "voxtral-small-24b": "Voxtral Small 24B",
    "open-cascade": "Open Cascade (Whisper + gpt-oss-120b)",
}

ABLATION_LABEL = {
    "SAGE-no_gate_types": "$-$Gate-type selection",
    "SAGE-no_verifier": "$-$Verifier",
    "SAGE-no_routing": "$-$Routing",
    "SAGE-no_credential_check": "$-$Credential check",
    "SAGE-no_exemption": "$-$Exemption",
    "SAGE-no_policy": "$-$Policy prompt",
}
# Task 19a: each ablation's designated predicted-effect metric (spec section 2's
# ablations table / RQ4 prediction), now exposed by rp.analysis.metrics'
# ABLATION_METRICS. This is the single metric Holm-adjusted across the default
# RQ4_CONFIGS family; SAGE-no_exemption and SAGE-no_policy additionally report a
# *secondary* predicted effect (over_escalation / under-gating respectively)
# that is shown with its own (unadjusted) permutation p.
#
# Task 25: SAGE-no_policy's predicted effect is *not* one of ABLATION_METRICS --
# spec section 6 predicts the written policy's removal from the agent prompt
# regresses the pre-registered primary metrics (gate-type error, under-gating),
# same as B0 vs SAGE in RQ1. wrong_type/under are already computed by
# rp.analysis.metrics.turn_counts for every row set, so no metrics.py change is
# needed to report them here.
ABLATION_PRIMARY_METRIC = {
    "SAGE-no_gate_types": "authorize_as_confirm",
    "SAGE-no_verifier": "wrong_content",
    "SAGE-no_routing": "wrong_content",   # predicted to NOT improve -- see rq4_table's note
    "SAGE-no_credential_check": "credential_bypass",
    "SAGE-no_exemption": "delayed_escalation",
    "SAGE-no_policy": "wrong_type",       # gate-type error
}
ABLATION_SECONDARY_METRIC = {
    "SAGE-no_exemption": "over_escalation",
    "SAGE-no_policy": "under",            # under-gating
}
RQ4_COLUMNS = ("authorize_as_confirm", "wrong_content", "credential_bypass",
              "delayed_escalation", "over_escalation", "wrong_type", "under")

DOMAIN_ORDER = ("counseling_support", "ecommerce_support", "education_student_services",
                "finance_billing", "housing_location", "negotiation_procurement", "travel_identity")
DOMAIN_LABEL = {
    "counseling_support": "Counseling", "ecommerce_support": "E-commerce",
    "education_student_services": "Education", "finance_billing": "Finance",
    "housing_location": "Housing", "negotiation_procurement": "Negotiation",
    "travel_identity": "Travel",
}


# --------------------------------------------------------------------------- git / io helpers

def _git_sha() -> str:
    try:
        out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 -- provenance only, never fatal
        return "unknown"


class Loader:
    """Loads rows for one output table, tracking the run_ids and excluded-ERROR
    count that go into that table's header comment (controller ruling 2)."""

    def __init__(self, split: str, judge: str = "primary"):
        self.split = split
        self.judge = judge
        self.run_ids: set[str] = set()
        self.rules_hashes: set[str] = set()
        # Keyed by turn identity (not just counted) so loading the same
        # agent/config more than once in one table (e.g. once for the main
        # body, once inside primary_family for the diff block) doesn't
        # double-count its excluded ERROR rows.
        self._error_keys: set[tuple] = set()

    @property
    def n_errors(self) -> int:
        return len(self._error_keys)

    def load(self, agent: str | None = None, config: str | None = None) -> list[dict]:
        try:
            rows = load_runs(self.split, agent, config, self.judge)
        except FileNotFoundError:
            rows = []
        try:
            errors = load_errors(self.split, agent, config, self.judge)
        except FileNotFoundError:
            errors = []
        for r in rows:
            if r.get("run_id"):
                self.run_ids.add(r["run_id"])
            if r.get("judge_rules_hash"):
                self.rules_hashes.add(r["judge_rules_hash"])
        for r in errors:
            if r.get("run_id"):
                self.run_ids.add(r["run_id"])
            self._error_keys.add((r.get("agent"), r.get("config"), r.get("case_id"),
                                   r.get("turn"), r.get("pass"), r.get("run_id")))
        return rows

    def pooled(self, agents, config: str | None = None) -> list[dict]:
        """Rows for several agents, with case_id prefixed by agent so a
        case-cluster bootstrap/permutation test over the pool treats
        (agent, case_id) as the resampling unit -- case_ids are reused across
        agents (same case script, different agent), so without this prefix
        two different agents' turns for "the same" case_id would be merged
        into one spurious cluster."""
        out = []
        for agent in agents:
            for r in self.load(agent, config):
                r2 = dict(r)
                r2["case_id"] = f"{agent}::{r['case_id']}"
                out.append(r2)
        return out

    def header(self, extra: str = "") -> str:
        ids = ", ".join(sorted(self.run_ids)) if self.run_ids else "none"
        lines = [
            f"% Generated by rp.analysis.report -- git sha {_git_sha()}",
            f"% run_ids: {ids}",
            f"% excluded ERROR turns: {self.n_errors}",
            f"% judge_rules_hash: {', '.join(sorted(self.rules_hashes)) if self.rules_hashes else 'none (v2)'}",
        ]
        if extra:
            lines.append(f"% {extra}")
        return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- formatting

def _num(x: float) -> str:
    if x != x:  # NaN
        return "?"
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.1f}"


def fmt_pct(num: float, den: float) -> str:
    """`n/N = x.x% [lo, hi]` (Wilson interval), or an em dash when den is 0."""
    if not den:
        return EMDASH
    lo, hi = wilson(num, den)
    return f"{_num(num)}/{_num(den)} = {100 * num / den:.1f}\\% [{100 * lo:.1f}, {100 * hi:.1f}]"


def fmt_diff(obs: float, lo: float, hi: float, p: float | None = None) -> str:
    """`+x.x pp [lo, hi], p=...` for a bootstrap difference (+ optional
    Holm-adjusted or raw p-value), or an em dash when the diff is undefined
    (e.g. a metric with zero denominator in both arms)."""
    if obs != obs:
        return EMDASH
    s = f"{100 * obs:+.1f} pp"
    if lo == lo and hi == hi:
        s += f" [{100 * lo:.1f}, {100 * hi:.1f}]"
    if p is not None and p == p:
        s += ", $p<0.001$" if p < 0.001 else f", $p={p:.3f}$"
    return s


def _tabular(header_comment: str, caption: str, label: str, col_spec: str,
             head_rows: list[str], body_rows: list[str], note: str = "") -> str:
    lines = [header_comment, "\\begin{table}[t]", "\\setlength{\\parindent}{0pt}%",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}",
             f"\\begin{{tabular}}{{{col_spec}}}", "\\toprule"]
    lines += head_rows
    lines.append("\\midrule")
    lines += body_rows
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    if note:
        lines.append(f"\\tablenote{{{note}}}")
    lines.append("\\end{table}\n")
    return "\n".join(lines)


# --------------------------------------------------------------------------- primary family (Holm)

def primary_family(split: str, judge: str, family: str):
    """SAGE vs each baseline on the two primary metrics, for the family named
    by `--family` (controller ruling 1):

      per_agent: computed separately for each of the two full-grid agents
                 (2 agents x 5 baselines x 2 metrics = 20 comparisons).
      pooled:    computed once, pooled over both full-grid agents with
                 case_id prefixed by agent (5 baselines x 2 metrics = 10
                 comparisons, matching sec:4-stats's "ten comparisons in all").

    Returns (loader, results) where results maps
    (unit, baseline, metric) -> (obs, lo, hi, p_raw, p_holm); `unit` is an
    agent name for `per_agent` or "pooled" for `pooled`.
    """
    loader = Loader(split, judge)
    raw = {}
    if family == "per_agent":
        for agent in FULL_GRID_AGENTS:
            sage_rows = loader.load(agent, "SAGE")
            for b in BASELINES:
                b_rows = loader.load(agent, b)
                for m in PRIMARY_METRICS:
                    obs, lo, hi = cluster_bootstrap_diff(b_rows, sage_rows, m)
                    p = perm_test(b_rows, sage_rows, m)
                    raw[(agent, b, m)] = (obs, lo, hi, p)
    elif family == "pooled":
        sage_rows = loader.pooled(FULL_GRID_AGENTS, "SAGE")
        for b in BASELINES:
            b_rows = loader.pooled(FULL_GRID_AGENTS, b)
            for m in PRIMARY_METRICS:
                obs, lo, hi = cluster_bootstrap_diff(b_rows, sage_rows, m)
                p = perm_test(b_rows, sage_rows, m)
                raw[("pooled", b, m)] = (obs, lo, hi, p)
    else:
        raise ValueError(f"--family must be 'pooled' or 'per_agent', got {family!r}")

    pvals = {k: v[3] for k, v in raw.items() if v[3] == v[3]}
    adj = holm(pvals)
    results = {k: (*v, adj.get(k, float("nan"))) for k, v in raw.items()}
    return loader, results


# --------------------------------------------------------------------------- RQ1

def rq1_table(split: str, out_path: Path, family: str, judge: str = "primary",
              filename: str = "rq1_main.tex", judge_note: str = "") -> None:
    metrics_cols = [("wrong_type", "Gate-type error"), ("under", "Under-gating"),
                    ("over", "Over-gating"), ("task_success", "Task success"),
                    ("swl", "Sev.-wtd. loss"), ("severe", "\\textsc{severe}")]

    loader = Loader(split, judge)
    body = []
    for config in FULL_CONFIGS:
        rows = loader.pooled(FULL_GRID_AGENTS, config)
        cells = [fmt_pct(*turn_counts(rows)[m]) for m, _ in metrics_cols]
        label = {"B0": "B0 No guard", "B1": "B1 Safety prompt", "B2": "B2 Policy in prompt",
                  "B3": "B3 Confirm-only", "B4": "B4 Verify all", "SAGE": "SAGE"}[config]
        body.append(f"{label} & " + " & ".join(cells) + " \\\\")

    # SAGE-minus-baseline differences (the family this table's Holm block covers).
    fam_loader, fam = primary_family(split, judge, family)
    loader.run_ids |= fam_loader.run_ids
    loader._error_keys |= fam_loader._error_keys
    metric_label = {"wrong_type": "Gate-type error", "under": "Under-gating"}
    if family == "pooled":
        diff_cols = [("pooled", m) for m in PRIMARY_METRICS]
    else:
        diff_cols = [(agent, m) for agent in FULL_GRID_AGENTS for m in PRIMARY_METRICS]
    diff_col_labels = [f"{metric_label[m]} ({unit})" if family == "per_agent" else metric_label[m]
                        for unit, m in diff_cols]

    diff_rows = []
    for b in BASELINES:
        keys = [(unit, b, m) for unit, m in diff_cols]
        cells = [fmt_diff(*fam[k][:3], fam[k][4]) for k in keys]
        diff_rows.append(f"SAGE $-$ {b} & " + " & ".join(cells) + " \\\\")

    head = ["Configuration & \\shortstack{Gate-type\\\\error} & \\shortstack{Under-\\\\gating} & "
            "\\shortstack{Over-\\\\gating} & \\shortstack{Task\\\\success} & "
            "\\shortstack{Sev.-wtd.\\\\loss} & \\textsc{severe} \\\\"]
    note = ("Three passes per configuration and agent. Gate metrics are over gate-scoreable turns "
            "$|E|$, stated per row. \\textsc{severe} is descriptive only." + judge_note)

    table = _tabular(
        loader.header(f"primary comparison family: {family} ({len(fam)} comparisons, Holm-adjusted)"),
        "RQ1: SAGE Versus Five Baselines on the Test Set, Pooled Over the Two Full-Grid Agents"
        + (" (Second Judge)" if judge != "primary" else ""),
        "tab:rq1" if judge == "primary" else "tab:rq1-judge2",
        "@{}lcccccc@{}", head, body, note,
    )
    diff_note = ("SAGE-minus-baseline differences on the two primary outcomes: case-cluster "
                 "bootstrap 95\\% CI and case-level permutation $p$, Holm-adjusted across the "
                 f"{len(fam)}-comparison {family} family.")
    diff_table = _tabular(
        "", f"RQ1 Differences: SAGE Minus Each Baseline ({family} family)",
        "tab:rq1-diff" if judge == "primary" else "tab:rq1-diff-judge2",
        "@{}l" + "c" * len(diff_cols) + "@{}",
        ["Comparison & " + " & ".join(diff_col_labels) + " \\\\"],
        diff_rows, diff_note,
    ) if diff_rows else ""

    (out_path / filename).write_text(table + "\n" + diff_table)


# --------------------------------------------------------------------------- RQ2

def rq2_table(split: str, out_path: Path, judge: str = "primary") -> None:
    loader = Loader(split, judge)
    body = []
    n_win = n_loss = n_tie = 0
    for agent in AGENT_ORDER:
        cells = []
        per_config = {}
        for config in ("B0", "B2", "SAGE"):
            rows = loader.load(agent, config)
            per_config[config] = turn_counts(rows) if rows else None
        for metric in ("wrong_type", "under"):
            for config in ("B0", "B2", "SAGE"):
                tc = per_config[config]
                cells.append(fmt_pct(*tc[metric]) if tc else EMDASH)
        if all(per_config.values()):
            # Per metric: SAGE strictly below the better of B0/B2 ("win" on
            # that metric), strictly above ("loss"), or equal to it ("tie").
            # Ties count neither as a win nor a loss for the sign test --
            # SAGE ranking first requires strictly beating both baselines on
            # both metrics, not merely tying for first.
            worse_on_any = False
            strict_on_all = True
            for m in PRIMARY_METRICS:
                sage_num, sage_den = per_config["SAGE"][m]
                sage_rate = sage_num / sage_den if sage_den else float("nan")
                baseline_rates = []
                for b in ("B0", "B2"):
                    num, den = per_config[b][m]
                    baseline_rates.append(num / den if den else float("inf"))
                best = min(baseline_rates)
                if sage_rate != sage_rate:  # NaN (zero-denominator metric)
                    strict_on_all = False
                    continue
                if sage_rate > best:
                    worse_on_any = True
                elif not (sage_rate < best):
                    strict_on_all = False
            if worse_on_any:
                n_loss += 1
            elif strict_on_all:
                n_win += 1
            else:
                n_tie += 1
        body.append(f"{AGENT_LABEL[agent]} & " + " & ".join(cells) + " \\\\")

    # Exact one-sided sign test over agents that were neither a tie nor
    # missing data: P(X >= n_win) under Binomial(n_win + n_loss, 0.5).
    def _sign_p(k, n):
        if n == 0:
            return float("nan")
        from math import comb
        return sum(comb(n, i) for i in range(k, n + 1)) / (2 ** n)

    n_for_sign = n_win + n_loss
    p_sign = _sign_p(n_win, n_for_sign)
    p_str = f"{p_sign:.3f}" if p_sign == p_sign else "?"
    body.append(f"\\midrule\nAgents where SAGE ranks first & \\multicolumn{{6}}{{c}}"
                f"{{{n_win} of {n_for_sign} (ties: {n_tie}); sign test $p={p_str}$}} \\\\")

    head = [" & \\multicolumn{3}{c}{Gate-type error} & \\multicolumn{3}{c}{Under-gating} \\\\",
            "Agent & B0 & B2 & SAGE & B0 & B2 & SAGE \\\\"]
    note = ("Win: SAGE strictly better than the better of B0 and B2 on both primary metrics "
            "(gate-type error, under-gating). Loss: strictly worse on at least one. Tie: "
            "otherwise; ties are excluded from the sign test. Prediction: SAGE ranks first on "
            f"$\\geq$ {RQ2_PREDICTION_MIN_WINS} of {len(AGENT_ORDER)} agents.")
    table = _tabular(
        loader.header(),
        "RQ2: Gate-Type Error and Under-Gating per Agent for B0, B2, and SAGE",
        "tab:rq2", "@{}p{0.235\\linewidth}cccccc@{}", head, body, note,
    )
    (out_path / "rq2_per_agent.tex").write_text(table)


# --------------------------------------------------------------------------- RQ3 (cost)

def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    idx = min(len(s) - 1, max(0, round(q * (len(s) - 1))))
    return s[idx]


def _cost_stats(rows: list[dict]):
    """(p50 latency, p95 latency, p50 guard_ms, p95 guard_ms, verifier calls/case,
    extra user turns/case) from raw turn rows -- no case_table needed, these
    are per-turn/per-case descriptive stats, not a between-config comparison."""
    if not rows:
        return None
    lat = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
    guard = [r["guard_ms"] for r in rows if r.get("guard_ms") is not None]
    cases = {r["case_id"] for r in rows}
    n_cases = len(cases) or 1
    verifier_calls = sum(r.get("verifier_calls") or 0 for r in rows) / n_cases
    # "Extra user turns": pauses on turns that required execute (a live user
    # would otherwise never have been asked a question) -- gate_expected ==
    # "execute" but the agent produced a non-empty `blocked` list this turn.
    extra_turns = sum(1 for r in rows if r.get("gate_expected") == "execute" and r.get("blocked")) / n_cases
    return {
        "lat_p50": _percentile(lat, 0.5), "lat_p95": _percentile(lat, 0.95),
        "guard_p50": _percentile(guard, 0.5), "guard_p95": _percentile(guard, 0.95),
        "verifier_calls": verifier_calls, "extra_turns": extra_turns,
    }


def _cost_per_100_turns(split: str, agents, config: str, n_turns: int) -> tuple[float, int]:
    """($ per 100 turns, number of partially-priced cases) from cost_log.jsonl,
    pooled over `agents`, for this split/config.

    `cost_usd` is None *by design* (rp.runner.price_usage) whenever any usage
    component of a case's records could not be priced -- summing it directly
    would crash `statistics.mean` the first time a real run hits an unpriced
    model or an unknown usage field. Instead this sums `known_cost_usd`
    (falling back to `cost_usd` only if `known_cost_usd` is absent), the same
    fallback `rp.runner.cumulative_spend` uses, and separately counts how many
    distinct cases had a record with `cost_usd is None` ("partially priced"),
    so that undercount is visible rather than silently swallowed.

    Returns (NaN, 0) if the log is missing, there are no judged turns, or no
    record matches -- never raises."""
    log_path = RESULTS_ROOT / "cost_log.jsonl"
    if not log_path.exists() or not n_turns:
        return float("nan"), 0
    total_known = 0.0
    partial_cases: set = set()
    matched = False
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("split") != split or rec.get("config") != config or rec.get("agent") not in agents:
                continue
            matched = True
            total_known += float(rec.get("known_cost_usd", rec.get("cost_usd")) or 0.0)
            if rec.get("cost_usd") is None:
                partial_cases.add(rec.get("case_id"))
    if not matched:
        return float("nan"), 0
    return (total_known / n_turns) * 100, len(partial_cases)


def rq3_table(split: str, out_path: Path) -> None:
    loader = Loader(split, "primary")
    body = []
    label = {"B0": "B0 No guard", "B1": "B1 Generic safety prompt", "B2": "B2 Policy in prompt",
              "B3": "B3 Confirm-only guardrail", "B4": "B4 Verifier on every call", "SAGE": "SAGE",
              "SAGE-no_routing": "SAGE $-$routing"}
    configs = list(FULL_CONFIGS) + ["SAGE-no_routing"]
    partial_by_config: dict[str, int] = {}
    for config in configs:
        rows = loader.pooled(FULL_GRID_AGENTS, config)
        st = _cost_stats(rows)
        cost, n_partial = _cost_per_100_turns(split, FULL_GRID_AGENTS, config, len(rows))
        if n_partial:
            partial_by_config[config] = n_partial
        if st is None:
            cells = [EMDASH] * 4 + [EMDASH]
        else:
            no_guard = config in ("B0", "B1", "B2")  # these configs intercept nothing
            lat_cells = [EMDASH, EMDASH] if no_guard else [f"{st['lat_p50']:.0f}", f"{st['lat_p95']:.0f}"]
            verif_cell = EMDASH if config in ("B0", "B1", "B2", "B3") else f"{st['verifier_calls']:.2f}"
            cost_cell = f"{cost:.4f}" if cost == cost else EMDASH
            if n_partial:
                cost_cell += "$^{*}$"
            cells = lat_cells + [f"{st['extra_turns']:.2f}", verif_cell, cost_cell]
        body.append(f"{label[config]} & " + " & ".join(cells) + " \\\\")

    # Routing retention / call share: SAGE (routed) vs SAGE-no_routing (always-on).
    # "Detection" = a turn with a verifier block (turn_flags' verifier_block: a
    # `blocked` entry with source == "verifier", not a rules-only block);
    # retention = |detections shared by both arms| / |SAGE-no_routing's detections|,
    # matched by (case_id, turn); call share = pooled verifier_calls, SAGE /
    # SAGE-no_routing. Bootstrapped over cases the same way as stats.cluster_bootstrap_diff.
    routed_rows = loader.pooled(FULL_GRID_AGENTS, "SAGE")
    always_rows = loader.pooled(FULL_GRID_AGENTS, "SAGE-no_routing")
    retention, call_share = _routing_retention(routed_rows, always_rows)
    body.append("\\midrule")
    body.append(f"Routing retention / call share & \\multicolumn{{5}}{{c}}"
                f"{{{retention} / {call_share}}} \\\\")

    head = [" & \\multicolumn{2}{c}{Added latency (ms)} & Extra user & Verifier calls & \\$ per \\\\",
            "Configuration & p50 & p95 & turns per case & per case & 100 turns \\\\"]
    note = ("Added latency is measured per intercepted call from the function-call event to SAGE's "
            "decision; configurations that intercept nothing add none. Extra user turns and verifier "
            "calls are per case, pooled over the two full-grid agents. Cost is known cost "
            "(results/rp/cost_log.jsonl's known\\_cost\\_usd, falling back to cost\\_usd only when "
            "known\\_cost\\_usd is absent) summed over the config's cases and divided by its judged "
            "turn count, times 100.")
    if partial_by_config:
        bits = [f"{label[c]}: {n} case{'s' if n != 1 else ''} partially priced"
                for c, n in partial_by_config.items()]
        note += " $^{*}$ " + "; ".join(bits) + " (cost\\_usd unknown for at least one usage component)."
    table = _tabular(loader.header(), "RQ3: Cost of Each Configuration on the Two Full-Grid Agents",
                     "tab:rq3", "@{}lccccc@{}", head, body, note)
    (out_path / "rq3_cost.tex").write_text(table)


def _routing_retention(routed_rows, always_rows, n=2000, seed=0):
    """(retention pct string, call-share pct string), each with a case-cluster
    bootstrap 95% CI, or an em dash if either arm has no rows.

    "Detection" (Task 19a) is specifically a *verifier* block (turn_flags'
    verifier_block: some `blocked` entry has source == "verifier"), not any
    block -- a rules-only block (e.g. a plain gate-checker readback/passcode
    failure) is not a verifier detection and must not count here or it would
    overstate retention when routing is off but the verifier never fired."""
    import random
    if not routed_rows or not always_rows:
        return EMDASH, EMDASH

    def detections(rows):
        return {(r["case_id"], r["turn"]) for r in rows if turn_flags(r)["verifier_block"]}

    def calls_by_case(rows):
        out = {}
        for r in rows:
            out[r["case_id"]] = out.get(r["case_id"], 0) + (r.get("verifier_calls") or 0)
        return out

    always_det = detections(always_rows)
    routed_det = detections(routed_rows)
    routed_calls = calls_by_case(routed_rows)
    always_calls = calls_by_case(always_rows)
    cases = sorted(set(r["case_id"] for r in routed_rows) & set(r["case_id"] for r in always_rows))
    if not cases or not always_det:
        return EMDASH, EMDASH

    def retention_of(case_set):
        keep_always = {(c, t) for c, t in always_det if c in case_set}
        keep_routed = {(c, t) for c, t in routed_det if c in case_set}
        if not keep_always:
            return float("nan")
        return len(keep_always & keep_routed) / len(keep_always)

    def call_share_of(case_set):
        r = sum(routed_calls.get(c, 0) for c in case_set)
        a = sum(always_calls.get(c, 0) for c in case_set)
        return r / a if a else float("nan")

    obs_r, obs_c = retention_of(set(cases)), call_share_of(set(cases))
    rng = random.Random(seed)
    r_draws, c_draws = [], []
    for _ in range(n):
        sample = [rng.choice(cases) for _ in cases]
        cs = sample  # multiset via list; membership tests below use `in` over list is fine at this scale
        r = retention_of(cs)
        c = call_share_of(cs)
        if r == r:
            r_draws.append(r)
        if c == c:
            c_draws.append(c)

    def _ci(draws, obs):
        if not draws:
            return f"{100*obs:.1f}\\%" if obs == obs else EMDASH
        draws.sort()
        lo = draws[int(0.025 * len(draws))]
        hi = draws[min(int(0.975 * len(draws)), len(draws) - 1)]
        return f"{100*obs:.1f}\\% [{100*lo:.1f}, {100*hi:.1f}]"

    return _ci(r_draws, obs_r), _ci(c_draws, obs_c)


# --------------------------------------------------------------------------- RQ4 (ablation)

def rq4_table(split: str, out_path: Path, ablation_agent: str | None = None) -> None:
    """RQ4 (Task 19a, extended by Task 25): every predicted-effect column of
    tab:ablation is filled from rp.analysis.metrics' ABLATION_METRICS
    (evaluation-only; SAGE/baselines never read these) plus, for SAGE-no_policy,
    the pre-registered primary metrics (gate-type error, under-gating) already
    computed by turn_counts. Each ablation's designated primary metric
    (ABLATION_PRIMARY_METRIC) is Holm-adjusted across the default RQ4_CONFIGS
    family (4 ablations: gate-type selection, verifier, credential check,
    policy prompt); SAGE-no_exemption's and SAGE-no_policy's second predicted
    effect (over_escalation / under, respectively) and SAGE-no_routing's cost
    figures (verifier calls/turn, guard_ms) are reported alongside but are not
    part of that 4-comparison Holm family (they are either a secondary effect
    or a non-proportion descriptive cost stat). SAGE-no_routing and
    SAGE-no_exemption remain supported configs (CONFIGS/build_config/
    ABLATION_CONFIGS) but are not part of this default table."""
    loader = Loader(split, "primary")
    if ablation_agent is None:
        ablation_agent = ABLATION_AGENT

    sage_rows = loader.load(ablation_agent, "SAGE")
    per_ablation_rows = {c: loader.load(ablation_agent, c) for c in RQ4_CONFIGS}

    pvals = {}
    # config -> {metric: (obs, lo, hi, p_raw)}
    cellvals: dict[str, dict[str, tuple]] = {c: {} for c in RQ4_CONFIGS}
    for config in RQ4_CONFIGS:
        rows = per_ablation_rows[config]
        if not rows or not sage_rows:
            continue
        primary = ABLATION_PRIMARY_METRIC[config]
        obs, lo, hi = cluster_bootstrap_diff(sage_rows, rows, primary)  # ablation - SAGE
        p = perm_test(sage_rows, rows, primary)
        pvals[config] = p
        cellvals[config][primary] = (obs, lo, hi, p)
        secondary = ABLATION_SECONDARY_METRIC.get(config)
        if secondary:
            obs2, lo2, hi2 = cluster_bootstrap_diff(sage_rows, rows, secondary)
            p2 = perm_test(sage_rows, rows, secondary)
            cellvals[config][secondary] = (obs2, lo2, hi2, p2)
    adj = holm(pvals)  # the default 4-ablation RQ4 family (one primary p per ablation)

    sage_cost = _cost_stats(sage_rows) if sage_rows else None

    def _cost_cells(rows):
        st = _cost_stats(rows) if rows else None
        if st is None:
            return [EMDASH, EMDASH]
        return [f"{st['verifier_calls']:.2f}", f"{st['guard_p50']:.0f}"]

    body = []
    # SAGE (full) row: show its own descriptive rate for every predicted-effect
    # column, plus its own cost figures as the reference point for SAGE-no_routing.
    row0 = [fmt_pct(*turn_counts(sage_rows)[m]) if sage_rows else EMDASH for m in RQ4_COLUMNS]
    row0 += ([f"{sage_cost['verifier_calls']:.2f}", f"{sage_cost['guard_p50']:.0f}"] if sage_cost
             else [EMDASH, EMDASH])
    body.append("SAGE (full) & " + " & ".join(row0) + " \\\\")
    for config in RQ4_CONFIGS:
        vals = cellvals[config]
        primary = ABLATION_PRIMARY_METRIC[config]
        cells = []
        for m in RQ4_COLUMNS:
            if m in vals:
                obs, lo, hi, p = vals[m]
                p_show = adj.get(config) if m == primary else p  # Holm only on the primary cell
                cells.append(fmt_diff(obs, lo, hi, p_show))
            else:
                cells.append(EMDASH)
        cells += (_cost_cells(per_ablation_rows[config]) if config == "SAGE-no_routing"
                  else [EMDASH, EMDASH])
        body.append(f"{ABLATION_LABEL[config]} & " + " & ".join(cells) + " \\\\")

    head = [" & Confirm instead & Right gate, & Credential & Delayed & Over- & Gate-type & Under- "
            "& Verifier & Guard \\\\",
            "Configuration & of authorize & wrong content & bypasses & escalations & escalations & "
            "error & gating & calls/turn & ms (p50) \\\\"]
    note = (f"Ablation agent: {ablation_agent or EMDASH}. Counts over three passes; case-cluster "
            "bootstrap 95\\% CI and case-level permutation $p$ against full SAGE. Each ablation's "
            "predicted primary effect (confirm-instead-of-authorize for $-$gate-type selection; "
            "right-gate-wrong-content for $-$verifier; credential bypasses for $-$credential check; "
            "gate-type error for $-$policy prompt) is Holm-adjusted across these four ablations. "
            "$-$Policy prompt's second predicted effect (under-gating) is shown with its own "
            "unadjusted $p$ and is not part of that 4-comparison Holm family. SAGE-no_routing "
            "(right-gate-wrong-content, predicted to NOT improve, plus verifier-calls/guard\\_ms "
            "cost) and SAGE-no_exemption (delayed/over-escalations) remain supported ablations but "
            "are not part of this default table.")
    table = _tabular(loader.header(f"ablation agent: {ablation_agent}"),
                     "RQ4: Ablations of SAGE on the Ablation Agent", "tab:rq4",
                     "@{}lccccccccc@{}", head, body, note)
    (out_path / "rq4_ablation.tex").write_text(table)


# --------------------------------------------------------------------------- per-domain

def per_domain_table(split: str, out_path: Path) -> None:
    loader = Loader(split, "primary")
    all_b0 = loader.load(None, "B0")
    all_sage = loader.load(None, "SAGE")
    body = []
    for domain in DOMAIN_ORDER:
        b0_rows = [r for r in all_b0 if r.get("domain") == domain]
        sage_rows = [r for r in all_sage if r.get("domain") == domain]
        n_turns = len({(r["case_id"], r["turn"]) for r in b0_rows + sage_rows})
        cells = []
        for m in ("wrong_type", "under", "task_success"):
            cells.append(fmt_pct(*turn_counts(b0_rows)[m]) if b0_rows else EMDASH)
            cells.append(fmt_pct(*turn_counts(sage_rows)[m]) if sage_rows else EMDASH)
        body.append(f"{DOMAIN_LABEL[domain]} ({n_turns}) & " + " & ".join(cells) + " \\\\")

    head = [" & \\multicolumn{2}{c}{Gate-type error} & \\multicolumn{2}{c}{Under-gating} & "
            "\\multicolumn{2}{c}{Task success} \\\\",
            "Domain (test turns) & B0 & SAGE & B0 & SAGE & B0 & SAGE \\\\"]
    table = _tabular(loader.header(), "Primary Metrics and Task Success by Domain, B0 and SAGE",
                     "tab:domain", "@{}lcccccc@{}", head, body)
    (out_path / "per_domain.tex").write_text(table)


# --------------------------------------------------------------------------- human validation (Task 23 fills in the rest)

def human_validation_table(split: str, out_path: Path) -> None:
    """Skeleton confusion matrix (tab:confusion): counts of (verdict, verdict)
    pairs the judge produced on primary-judge vs judge2-rescored rows of the
    same turns, as a stand-in until Task 23 wires in the actual human labels.
    Never crashes when judge2 rows don't exist yet -- every cell is an em dash."""
    loader = Loader(split, "primary")
    primary_rows = {(r["case_id"], r["turn"], r["agent"], r["config"], r["pass"]): r
                    for r in loader.load()}
    try:
        judge2_rows = load_runs(split, judge="judge2")
        for r in judge2_rows:
            if r.get("run_id"):
                loader.run_ids.add(r["run_id"])
    except FileNotFoundError:
        judge2_rows = []

    verdicts = ("PASS", "FAIL", "SEVERE")
    counts = {(a, b): 0 for a in verdicts for b in verdicts}
    for r in judge2_rows:
        key = (r["case_id"], r["turn"], r["agent"], r["config"], r["pass"])
        p = primary_rows.get(key)
        if p is None:
            continue
        a, b = p["verdict"], r["verdict"]
        if a in verdicts and b in verdicts:
            counts[(a, b)] += 1

    body = []
    label = {"PASS": "\\textsc{pass}", "FAIL": "\\textsc{fail}", "SEVERE": "\\textsc{severe}"}
    for a in verdicts:
        cells = [str(counts[(a, b)]) if any(counts.values()) else EMDASH for b in verdicts]
        body.append(f"{label[a]} & " + " & ".join(cells) + " \\\\")

    head = ["Human $\\backslash$ judge & \\textsc{pass} & \\textsc{fail} & \\textsc{severe} \\\\"]
    note = ("Placeholder: rows/columns are primary-judge vs second-judge-family verdict on the same "
            "turns, pending Task 23's human labels. \\textsc{severe} in the judge column is the trap "
            "matcher's verdict, as in every reported number.")
    table = _tabular(loader.header(), "Adjudicated Human Verdict Versus Judge Verdict on the "
                     "Validation Sample", "tab:confusion", "@{}lccc@{}", head, body, note)
    (out_path / "human_validation.tex").write_text(table)


# --------------------------------------------------------------------------- figure CSVs

def _csv(path: Path, header: list[str], rows: list[list]) -> None:
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join("" if v is None or v != v else str(v) for v in row))
    path.write_text("\n".join(lines) + "\n")


SEVERITY_ORDER = ("low", "medium", "high", "critical")


def heatmap_csv(split: str, out_path: Path) -> None:
    rows = []
    for agent in AGENT_ORDER:
        try:
            all_rows = load_runs(split, agent, "SAGE")
        except FileNotFoundError:
            all_rows = []
        by_tier = {}
        for r in all_rows:
            by_tier.setdefault(r.get("severity"), []).append(r)
        for tier in SEVERITY_ORDER:
            tc = turn_counts(by_tier.get(tier, []))
            num, den = tc["task_success"]
            rate = num / den if den else ""
            rows.append([agent, tier, rate])
    _csv(out_path / "heatmap_agent_severity.csv", ["agent", "tier", "pass_rate"], rows)


def forest_csv(split: str, out_path: Path, family: str) -> None:
    _, fam = primary_family(split, "primary", family)
    rows = []
    for (unit, baseline, metric), (obs, lo, hi, p_raw, p_holm) in sorted(fam.items()):
        comparison = f"SAGE_vs_{baseline}" + ("" if unit == "pooled" else f"_{unit}")
        label = f"SAGE $-$ {baseline}" + ("" if unit == "pooled" else f" ({AGENT_LABEL.get(unit, unit)})")
        rows.append([comparison, label, metric, obs if obs == obs else "", lo if lo == lo else "",
                     hi if hi == hi else ""])
    _csv(out_path / "forest.csv", ["comparison", "label", "metric", "diff", "lo", "hi"], rows)


def cost_safety_csv(split: str, out_path: Path) -> None:
    rows = []
    for config in FULL_CONFIGS:
        for agent in FULL_GRID_AGENTS:
            try:
                agent_rows = load_runs(split, agent, config)
            except FileNotFoundError:
                agent_rows = []
            cost, n_partial = _cost_per_100_turns(split, [agent], config, len(agent_rows))
            tc = turn_counts(agent_rows) if agent_rows else None
            gate_err = (tc["wrong_type"][0] / tc["wrong_type"][1]) if tc and tc["wrong_type"][1] else ""
            label = AGENT_LABEL.get(agent, agent)
            rows.append([config, agent, label, cost if cost == cost else "", gate_err, n_partial])
    _csv(out_path / "cost_safety.csv",
         ["config", "agent", "label", "cost_per_100", "gate_type_error", "partially_priced_cases"], rows)


def failure_stack_csv(split: str, out_path: Path) -> None:
    rows = []
    for agent in AGENT_ORDER:
        label = AGENT_LABEL.get(agent, agent)
        for config in ("B0", "SAGE"):
            try:
                agent_rows = load_runs(split, agent, config)
            except FileNotFoundError:
                agent_rows = []
            tc = turn_counts(agent_rows) if agent_rows else None
            if tc is None:
                rows.append([agent, label, config, "", "", "", "", ""])
                continue
            under = tc["under"][0]
            over = tc["over"][0]
            wrong_type = tc["wrong_type"][0]
            scoreable = tc["under"][1]
            unknown = len(agent_rows) - scoreable
            rows.append([agent, label, config, under, over, wrong_type, "", unknown])
    _csv(out_path / "failure_stack.csv",
         ["agent", "label", "config", "under", "over", "wrong_type", "wrong_content", "unknown"], rows)


def routing_curve_csv(split: str, out_path: Path) -> None:
    """Detections retained vs verifier-call budget fraction. SAGE's routing
    only exposes two operating points from stored rows (routed and
    always-verify); intermediate budget points would need a re-run at each
    trigger-budget, which the runner does not currently support, so only the
    two endpoints are written (0% budget = SAGE, 100% budget = SAGE-no_routing
    is the implicit reference at 100/100) -- see task-22-report.md."""
    routed = []
    always = []
    for agent in FULL_GRID_AGENTS:
        try:
            routed += load_runs(split, agent, "SAGE")
        except FileNotFoundError:
            pass
        try:
            always += load_runs(split, agent, "SAGE-no_routing")
        except FileNotFoundError:
            pass
    retention, call_share = _routing_retention(routed, always)
    rows = [[0.0, 0.0]]
    if retention != EMDASH:
        # Parse back the point estimate out of the formatted string's leading number.
        try:
            r_pct = float(retention.split("\\%")[0])
            c_pct = float(call_share.split("\\%")[0])
            rows.append([c_pct / 100, r_pct / 100])
        except (ValueError, IndexError):
            pass
    rows.append([1.0, 1.0])
    _csv(out_path / "routing_curve.csv", ["call_budget_fraction", "detections_retained"], rows)


def latency_csv(split: str, out_path: Path) -> None:
    rows = []
    for config in FULL_CONFIGS + ABLATION_CONFIGS:
        config_label = ABLATION_LABEL.get(config, config)
        for agent in AGENT_ORDER:
            label = f"{config_label} / {AGENT_LABEL.get(agent, agent)}"
            try:
                agent_rows = load_runs(split, agent, config)
            except FileNotFoundError:
                agent_rows = []
            st = _cost_stats(agent_rows)
            if st is None:
                rows.append([config, agent, label, "", "", "", ""])
            else:
                rows.append([config, agent, label, st["lat_p50"], st["lat_p95"],
                             st["guard_p50"], st["guard_p95"]])
    _csv(out_path / "latency.csv",
         ["config", "agent", "label", "latency_p50_ms", "latency_p95_ms", "guard_p50_ms", "guard_p95_ms"],
         rows)


# --------------------------------------------------------------------------- main

def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python3 -m rp.analysis.report", description=__doc__.split("\n\n")[0])
    p.add_argument("--split", default="test")
    p.add_argument("--out", default="../thesis_v2")
    p.add_argument("--family", choices=("pooled", "per_agent"), default="per_agent",
                    help="Primary-comparison family for Holm adjustment (pre-registration, Task 19, fixes this).")
    p.add_argument("--results-root", default=None,
                   help="read results (and cost_log.jsonl) from this root instead of results/rp, "
                        "e.g. results/rp_v3 for the judge-v3 re-scores. Note: a rejudge root holds no "
                        "pass*.judge2.jsonl files, so the judge2 tables are empty there, and its cost log "
                        "holds judge spend only")
    args = p.parse_args(argv)
    if args.results_root:
        global RESULTS_ROOT
        RESULTS_ROOT = Path(args.results_root)
        _metrics.set_results_root(RESULTS_ROOT)

    out_root = Path(args.out)
    tables_dir = out_root / "tables"
    data_dir = out_root / "data"
    tables_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    rq1_table(args.split, tables_dir, args.family, "primary", "rq1_main.tex")
    rq1_table(args.split, tables_dir, args.family, "judge2", "rq1_judge2.tex",
              judge_note=" Second judge family.")
    rq2_table(args.split, tables_dir)
    rq3_table(args.split, tables_dir)
    rq4_table(args.split, tables_dir)
    per_domain_table(args.split, tables_dir)
    human_validation_table(args.split, tables_dir)

    heatmap_csv(args.split, data_dir)
    forest_csv(args.split, data_dir, args.family)
    cost_safety_csv(args.split, data_dir)
    failure_stack_csv(args.split, data_dir)
    routing_curve_csv(args.split, data_dir)
    latency_csv(args.split, data_dir)

    print(f"wrote tables/*.tex and data/*.csv to {out_root}", file=sys.stderr)


if __name__ == "__main__":
    main()
