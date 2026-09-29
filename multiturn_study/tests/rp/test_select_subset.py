from scripts.select_test_subset import (
    N_TARGET,
    build_strata,
    case_required_gate,
    largest_remainder_allocation,
    load_test_cases,
    sample_subset,
)


def test_determinism():
    all_cases = load_test_cases()
    strata = build_strata(all_cases)
    allocation = largest_remainder_allocation(strata, N_TARGET, len(all_cases))
    a = sample_subset(strata, allocation)
    b = sample_subset(strata, allocation)
    assert a == b


def test_selects_80():
    all_cases = load_test_cases()
    strata = build_strata(all_cases)
    allocation = largest_remainder_allocation(strata, N_TARGET, len(all_cases))
    selected = sample_subset(strata, allocation)
    assert len(selected) == N_TARGET


def test_no_duplicates():
    all_cases = load_test_cases()
    strata = build_strata(all_cases)
    allocation = largest_remainder_allocation(strata, N_TARGET, len(all_cases))
    selected = sample_subset(strata, allocation)
    assert len(set(selected)) == len(selected)


def test_every_populated_stratum_keeps_at_least_one_case():
    all_cases = load_test_cases()
    strata = build_strata(all_cases)
    allocation = largest_remainder_allocation(strata, N_TARGET, len(all_cases))
    for key, ids in strata.items():
        assert len(ids) >= 1
        assert allocation[key] >= 1


def test_selected_ids_are_a_subset_of_full_set():
    all_cases = load_test_cases()
    strata = build_strata(all_cases)
    allocation = largest_remainder_allocation(strata, N_TARGET, len(all_cases))
    selected = sample_subset(strata, allocation)
    known = {c["id"] for c in all_cases}
    assert set(selected) <= known


def test_case_required_gate_is_the_strictest_step_gate():
    case = {"steps": [{"gate_expected": "execute"}, {"gate_expected": "confirm"},
                       {"gate_expected": "clarify"}]}
    assert case_required_gate(case) == "confirm"

    case_authorize = {"steps": [{"gate_expected": "authorize"}, {"gate_expected": "execute"}]}
    assert case_required_gate(case_authorize) == "authorize"

    case_single = {"steps": [{"gate_expected": "clarify"}]}
    assert case_required_gate(case_single) == "clarify"
