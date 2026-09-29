# sage-voice-agents

Code, cases and audio for the MS thesis **"The Right Pause"**: SAGE (Severity-Aware Gate Enforcement), a runtime guard that makes voice agents pause the right way (clarify, confirm, or authorize) before risky tool calls.

## Layout
- `multiturn_study/rp/` — SAGE, baselines (B0–B4), agents, judge v3, runner, analysis
- `multiturn_study/scripts/` — test-grid scheduler, progress views, data tools
- `multiturn_study/tests/` — unit tests
- `multiturn_study/data/` — dev (`dev_v2`) and locked test (`test_cases_v3`, with `GATE_POLICY.md`) cases with audio; `test_subset_v3.json` = the 80 pre-registered test cases
- `docs/prereg/` — pre-registered analysis plan; `docs/superpowers/` — design spec and implementation plan
- `accent_sweep/`, `data/speaker_profiles.py` — shared helpers

## Setup
```bash
cd multiturn_study
pip install -r requirements-rp.txt
python3 -m pytest tests/rp -q
```
API keys go in `multiturn_study/.env` (never committed).
