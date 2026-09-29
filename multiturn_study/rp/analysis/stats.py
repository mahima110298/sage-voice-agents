"""Cluster-aware statistics for between-config comparisons.

Turns are nested in cases, and cases are repeated across passes, so the CASE
is the resampling unit for every paired comparison: a bootstrap or sign-flip
permutation draw moves (or swaps) *all* passes and *all* turns of a case
together. This is a generalization of `scripts/cluster_reanalysis.py`'s
`paired_tests` / `wilson`, ported here so the resampling unit is the case
(rather than pooling passes away before resampling).
"""
from __future__ import annotations

import math
import random
import warnings
from collections import defaultdict

from rp.analysis.metrics import case_table


def ratio(ct: dict, keys, metric: str) -> float:
    """Pooled ratio (sum of numerators / sum of denominators) over the cases
    named in `keys`, restricted to the given metric.

    `ct` is a `case_table` (keyed by (case_id, pass)). `keys` is an iterable
    of case_ids; a case_id may repeat (as it does in a bootstrap resample),
    in which case its contribution is counted once per repetition.
    """
    weight = defaultdict(int)
    for c in keys:
        weight[c] += 1
    if not weight:
        return float("nan")
    num = 0.0
    den = 0.0
    for (case_id, _pass), metrics in ct.items():
        w = weight.get(case_id, 0)
        if not w:
            continue
        k, n = metrics[metric]
        num += w * k
        den += w * n
    return num / den if den else float("nan")


def _agg_by_case(ct: dict) -> dict[str, dict[str, tuple[float, float]]]:
    """Collapse a (case_id, pass)-keyed case_table to per-case sums (all
    passes of a case combined), for the sign-flip permutation test."""
    out: dict = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    for (case_id, _pass), metrics in ct.items():
        for m, (k, n) in metrics.items():
            out[case_id][m][0] += k
            out[case_id][m][1] += n
    return {c: {m: tuple(v) for m, v in d.items()} for c, d in out.items()}


def _ratio_by_case(agg: dict, cases, metric: str) -> float:
    num = sum(agg[c][metric][0] for c in cases)
    den = sum(agg[c][metric][1] for c in cases)
    return num / den if den else float("nan")


def cluster_bootstrap_diff(a_rows: list[dict], b_rows: list[dict], metric: str,
                            n: int = 10000, seed: int = 0):
    """Case-cluster bootstrap 95% CI for the difference in a pooled ratio
    (b - a). Resamples case_ids with replacement; all passes/turns of a case
    move together because `ratio` sums a case's contribution once per time
    it is drawn."""
    a_ct = case_table(a_rows)
    b_ct = case_table(b_rows)
    cases = sorted({c for c, _p in a_ct} & {c for c, _p in b_ct})
    rng = random.Random(seed)
    obs = ratio(b_ct, cases, metric) - ratio(a_ct, cases, metric)
    if not cases or obs != obs:  # obs is NaN (e.g. a metric with zero denominator)
        return obs, float("nan"), float("nan")
    diffs = []
    for _ in range(n):
        sample = [rng.choice(cases) for _ in cases]
        d = ratio(b_ct, sample, metric) - ratio(a_ct, sample, metric)
        if d == d:  # drop NaN resamples (e.g. an all-unscoreable draw) before sorting
            diffs.append(d)
    if not diffs:
        return obs, float("nan"), float("nan")
    diffs.sort()
    m = len(diffs)
    lo = diffs[int(0.025 * m)]
    hi = diffs[min(int(0.975 * m), m - 1)]
    return obs, lo, hi


def perm_test(a_rows: list[dict], b_rows: list[dict], metric: str,
              n: int = 20000, seed: int = 0) -> float:
    """Case-level sign-flip permutation p-value for the difference in a
    pooled ratio (b - a). Under H0 the two configs are exchangeable within a
    case, so each case's whole record (summed over its passes) is swapped
    between arms with probability 1/2."""
    a = _agg_by_case(case_table(a_rows))
    b = _agg_by_case(case_table(b_rows))
    cases = sorted(set(a) & set(b))
    if not cases:
        return float("nan")
    rng = random.Random(seed)
    obs = _ratio_by_case(b, cases, metric) - _ratio_by_case(a, cases, metric)
    if obs != obs:  # obs is NaN (e.g. a metric with zero denominator)
        return float("nan")
    extreme = 0
    for _ in range(n):
        na = nb = da = db = 0.0
        for c in cases:
            x, y = a[c][metric], b[c][metric]
            if rng.random() < 0.5:
                x, y = y, x
            na += x[0]; da += x[1]
            nb += y[0]; db += y[1]
        d = (nb / db if db else 0.0) - (na / da if da else 0.0)
        if abs(d) >= abs(obs) - 1e-12:
            extreme += 1
    return (extreme + 1) / (n + 1)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values: capped at 1 and made monotone
    (non-decreasing in sorted-p order)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    running_max = 0.0
    out = {}
    for i, (key, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running_max = max(running_max, adj)
        out[key] = running_max
    return out


def wilson(k: float, n: float, z: float = 1.96):
    """Wilson score 95% interval for a proportion k/n."""
    if not n:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return center - half, center + half


# ---- GLMM odds ratio (case + per-run random effects) -----------------------

def _metric_flag(row: dict, metric: str):
    from rp.analysis.metrics import turn_flags
    flags = turn_flags(row)
    if metric == "task_success":
        return flags["pass"]
    if metric == "burden":
        return flags["under"] or flags["over"]
    if metric in ("severe", "under", "over", "wrong_type"):
        return flags[metric]
    raise ValueError(f"glmm_or: unsupported metric {metric!r} (swl is not a binary outcome)")


def _scoreable_rows(rows: list[dict], metric: str) -> list[dict]:
    if metric in ("under", "over", "burden", "wrong_type"):
        from cluster_reanalysis import gate_class
        return [r for r in rows if gate_class(r) != "unscoreable"]
    return rows


def glmm_or(rows: list[dict], metric: str, ref_config: str) -> dict:
    """Per-config log-odds-ratio (vs ref_config) and 95% VB credible interval
    for a binary metric, from a Bayesian mixed binomial GLM with case and
    per-run random effects (`vc_formulas={"case": "0 + C(case_id)",
    "pass": "0 + C(pass_key)"}`).

    `pass_key = f"{case_id}::{config}::{pass}"`: SAGE's pass 1 of a case and
    B0's pass 1 of the *same* case are independent runs (different agent
    invocations against the same script), not the same random-effect draw,
    so the run-level random effect must be keyed by (case, config, pass),
    not just (case, pass).

    Never raises: on degenerate data -- too few configs, no cross-config
    variance, separation (any config's outcome is all 0 or all 1), a
    statsmodels convergence warning, or any other fit failure -- it warns
    and returns NaN bounds for every config.
    """
    use_rows = _scoreable_rows(rows, metric)
    configs = sorted({r["config"] for r in use_rows})
    nan_result = {c: (float("nan"), float("nan"), float("nan")) for c in configs}

    if ref_config not in configs or len(configs) < 2:
        warnings.warn(f"glmm_or: ref_config {ref_config!r} not comparable among {configs}; returning NaN")
        return nan_result or {ref_config: (float("nan"),) * 3}

    try:
        y = [int(_metric_flag(r, metric)) for r in use_rows]
    except ValueError as exc:
        warnings.warn(str(exc))
        return nan_result

    # Separation check: if ANY config's outcome is all 0 or all 1 for this
    # metric, the fixed effect for that config is not identified (or only
    # identified by the prior) -- warn and bail out rather than trust a
    # degenerate fit.
    per_config_y = defaultdict(list)
    for r, yi in zip(use_rows, y):
        per_config_y[r["config"]].append(yi)
    for c in configs:
        vals = per_config_y[c]
        if len(set(vals)) < 2:
            degenerate = "0" if vals[0] == 0 else "1"
            warnings.warn(
                f"glmm_or: config {c!r} is separated (all outcomes are {degenerate}) "
                f"for metric {metric!r}; returning NaN"
            )
            return nan_result

    try:
        import pandas as pd
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
        from statsmodels.tools.sm_exceptions import ConvergenceWarning

        other_configs = [c for c in configs if c != ref_config]
        df = pd.DataFrame({
            "y": y,
            "config": pd.Categorical([r["config"] for r in use_rows], categories=[ref_config] + other_configs),
            "case_id": [r["case_id"] for r in use_rows],
            "pass_key": [f'{r["case_id"]}::{r["config"]}::{r["pass"]}' for r in use_rows],
        })

        formula = f"y ~ C(config, Treatment(reference='{ref_config}'))"
        model = BinomialBayesMixedGLM.from_formula(
            formula,
            vc_formulas={"case": "0 + C(case_id)", "pass": "0 + C(pass_key)"},
            data=df,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            result = model.fit_vb()
        convergence_warnings = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
        if convergence_warnings:
            warnings.warn(
                f"glmm_or: fit_vb did not converge for metric {metric!r} "
                f"({convergence_warnings[0].message}); returning NaN"
            )
            return nan_result
    except Exception as exc:  # noqa: BLE001 - statsmodels can raise many things on degenerate input
        warnings.warn(f"glmm_or: model fit failed for metric {metric!r} ({exc}); returning NaN")
        return nan_result

    try:
        names = list(result.model.exog_names)
        means = result.fe_mean
        sds = result.fe_sd
        out = {ref_config: (0.0, 0.0, 0.0)}
        z = 1.96
        for c in other_configs:
            label = f"C(config, Treatment(reference='{ref_config}'))[T.{c}]"
            if label not in names:
                out[c] = (float("nan"), float("nan"), float("nan"))
                continue
            i = names.index(label)
            m, sd = means[i], sds[i]
            out[c] = (m, m - z * sd, m + z * sd)  # (log-odds, VB credible interval lo/hi)
        return out
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"glmm_or: could not extract coefficients for metric {metric!r} ({exc}); returning NaN")
        return nan_result
