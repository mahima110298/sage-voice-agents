"""Select the 80-of-119 locked-test-set subset used for the pre-registered runs.

Deterministic: same output every time it is run, given the current contents of
`data/test_cases_v3`. Reads metadata only (never modifies anything under
`data/test_cases_v3`).

Stratification: (domain, case-level required gate), where the case-level
required gate is the strictest gate among the case's steps, in the order
execute < clarify < confirm < authorize (i.e. the max over the case's
`gate_expected` values by that order).

Allocation: largest-remainder proportional allocation of 80 seats across
strata, proportional to each stratum's share of the 119 cases.

Sampling: a single `random.Random(20260928)` instance, strata visited in
sorted order, sampling from each stratum's case ids sorted lexically --
so the output does not depend on filesystem iteration order.

Output: multiturn_study/data/test_subset_v3.json (outside data/test_cases_v3,
which stays locked and untouched):
    {"seed": 20260928, "n": 80, "strata": {...}, "case_ids": [sorted]}

Usage:
    python3 -m scripts.select_test_subset
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path

from rp import cases as rp_cases

SEED = 20260928
N_TARGET = 80
GATE_ORDER = {"execute": 0, "clarify": 1, "confirm": 2, "authorize": 3}

STUDY_DIR = Path(__file__).resolve().parent.parent
TEST_CASES_DIR = STUDY_DIR / "data" / "test_cases_v3"
OUTPUT_PATH = STUDY_DIR / "data" / "test_subset_v3.json"


def case_required_gate(case: dict) -> str:
    """The strictest gate among the case's steps (execute < clarify < confirm < authorize)."""
    gates = [step["gate_expected"] for step in case["steps"]]
    return max(gates, key=lambda g: GATE_ORDER[g])


def load_test_cases() -> list[dict]:
    """All 119 locked test cases, read-only (verifies the lock first)."""
    rp_cases.verify_lock(TEST_CASES_DIR)
    return rp_cases.load_split("test")


def build_strata(all_cases: list[dict]) -> dict[tuple[str, str], list[str]]:
    """Maps (domain, required_gate) -> sorted case ids in that stratum."""
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for case in all_cases:
        key = (case["domain"], case_required_gate(case))
        strata[key].append(case["id"])
    return {k: sorted(v) for k, v in strata.items()}


def largest_remainder_allocation(strata: dict[tuple[str, str], list[str]], n_target: int,
                                  n_total: int) -> dict[tuple[str, str], int]:
    """Proportional allocation of n_target seats across strata (sizes from `strata`),
    by the largest-remainder method: floor each stratum's exact quota, then hand
    out the leftover seats to the strata with the largest fractional remainders
    (ties broken by stratum key, for determinism)."""
    quotas = {k: n_target * len(v) / n_total for k, v in strata.items()}
    floors = {k: math.floor(q) for k, q in quotas.items()}
    remainders = {k: quotas[k] - floors[k] for k in strata}
    allocation = dict(floors)
    leftover = n_target - sum(floors.values())
    ranked = sorted(strata, key=lambda k: (-remainders[k], k))
    for key in ranked[:leftover]:
        allocation[key] += 1
    return allocation


def sample_subset(strata: dict[tuple[str, str], list[str]],
                   allocation: dict[tuple[str, str], int]) -> list[str]:
    """One random.Random(SEED) instance, strata visited in sorted order, sampling
    from each stratum's sorted case ids -- deterministic regardless of dict order."""
    rng = random.Random(SEED)
    selected: list[str] = []
    for key in sorted(strata):
        selected.extend(rng.sample(strata[key], allocation[key]))
    return selected


def turn_count(all_cases: list[dict], selected_ids: set[str]) -> int:
    return sum(len(c["steps"]) for c in all_cases if c["id"] in selected_ids)


def print_stratum_table(strata: dict[tuple[str, str], list[str]],
                         allocation: dict[tuple[str, str], int]) -> None:
    print(f"{'domain':<28}{'gate':<12}{'n_full':>8}{'n_test':>8}")
    for key in sorted(strata):
        domain, gate = key
        print(f"{domain:<28}{gate:<12}{len(strata[key]):>8}{allocation[key]:>8}")
    print(f"{'TOTAL':<40}{sum(len(v) for v in strata.values()):>8}{sum(allocation.values()):>8}")


def main() -> None:
    all_cases = load_test_cases()
    n_total = len(all_cases)
    strata = build_strata(all_cases)
    allocation = largest_remainder_allocation(strata, N_TARGET, n_total)
    print_stratum_table(strata, allocation)

    selected = sample_subset(strata, allocation)
    selected_sorted = sorted(selected)
    assert len(selected_sorted) == N_TARGET
    assert len(set(selected_sorted)) == N_TARGET

    n_turns = turn_count(all_cases, set(selected_sorted))
    print(f"\nselected {N_TARGET} of {n_total} cases; {n_turns} turns")

    out = {
        "seed": SEED,
        "n": N_TARGET,
        "strata": {
            f"{domain}|{gate}": {"n_full": len(strata[(domain, gate)]),
                                  "n_test": allocation[(domain, gate)]}
            for domain, gate in sorted(strata)
        },
        "case_ids": selected_sorted,
    }
    OUTPUT_PATH.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
