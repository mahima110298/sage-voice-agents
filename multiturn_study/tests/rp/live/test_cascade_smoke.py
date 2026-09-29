"""Live smoke test for the cascade adapter (costs cents; skipped unless RP_LIVE=1).

  cd multiturn_study && RP_LIVE=1 python3 -m pytest tests/rp/live/test_cascade_smoke.py -s

Runs ONE dev_v2 case (ecommerce_mt_04, the same case the Gemini/OpenAI smokes
use) through configs B0 and SAGE (verifier_obj=None, so SAGE never calls out
to a verifier model -- this smoke only exercises the cascade's own LLM calls).
dev_v2 only -- never the locked test split.
"""
import asyncio
import os

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("RP_LIVE") != "1", reason="live API")


def _run(model, cfg_name, case):
    from rp.agents.cascade import CascadeAdapter
    from rp.interceptors import build_config
    from rp.prompts import BASE_PROMPT
    turns = asyncio.run(CascadeAdapter(model).run(case, build_config(cfg_name, None), BASE_PROMPT))
    print(f"\n== {model} / {cfg_name} / {case['id']}")
    for t in turns:
        print(f"turn {t.turn}: latency_ms={t.latency_ms} guard_ms={t.guard_ms:.2f} error={t.error}")
        print(f"  user:    {t.user_text!r}")
        print(f"  calls:   {[c.as_dict() for c in t.calls]}")
        print(f"  blocked: {t.blocked}")
        print(f"  agent:   {t.agent_text!r}")
        print(f"  usage:   {t.usage}")
    return turns


def test_one_dev_case_runs_b0_and_sage():
    from rp.agents.cascade import CASCADE_LLM
    from rp.cases import load_split
    model = os.environ.get("RP_LIVE_MODEL", CASCADE_LLM)
    case_id = os.environ.get("RP_LIVE_CASE", "ecommerce_mt_04")
    case = load_split("dev_v2", only_id=case_id)[0]

    for cfg_name in (os.environ.get("RP_LIVE_CONFIG") or "B0,SAGE").split(","):
        turns = _run(model, cfg_name, case)
        assert len(turns) == len(case["steps"])
        assert all(t.error is None for t in turns)
        assert all(t.user_text for t in turns) and any(t.agent_text for t in turns)
        assert all(t.latency_ms is not None and t.latency_ms >= 0 for t in turns)
