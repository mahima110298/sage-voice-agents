from __future__ import annotations
import json, os, re, time
import requests
from rp.numwords import numbers_in_text, norm
from rp.tools import is_read_only, KEY_ARGS
from rp.types import Call

# Cheapest non-Anthropic model (as of 2026-09-28 OpenRouter listing) among current
# small/fast-tier chat models (openai/*-mini, google/*-flash*) that supports JSON
# output via response_format: json_object, excluding ":batch" variants (this code
# calls the synchronous chat/completions endpoint, not the batch API).
# google/gemini-2.5-flash-lite: $0.10 / 1M input tokens, $0.40 / 1M output tokens.
VERIFIER_MODEL = "google/gemini-2.5-flash-lite"
_SELF_CORR = re.compile(r"(no wait|sorry|i mean|actually|--|—|scratch that|correction)", re.I)

PROMPT = """You check one tool call a voice assistant is about to make, against the conversation.
Answer with JSON only: {{"verdict": "ok"|"wrong_target"|"ambiguous"|"missing_value", "candidates": [...], "reason": "..."}}
- wrong_target: an argument is not what the user meant (e.g. a value the user corrected away from, the other of two records).
- ambiguous: two or more records or readings fit and nothing the user said separates them; list the candidates.
- missing_value: the user never supplied a value the call needs and it was not looked up.
- ok: otherwise.
Policy the assistant must follow:
{policy}
Conversation so far (user words are an automatic transcript):
{transcript}
Tool results so far: {results}
Call about to be made: {call}"""


class OpenRouterClient:
    def complete(self, model, messages):
        r = requests.post("https://openrouter.ai/api/v1/chat/completions", timeout=30,
                          headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
                          json={"model": model, "temperature": 0, "messages": messages,
                                "response_format": {"type": "json_object"}})
        r.raise_for_status()
        d = r.json()
        return {"content": d["choices"][0]["message"]["content"], "usage": d.get("usage", {})}


def _record_matches_call(records: list, call_args: dict) -> bool:
    """True if some record in a >1-length lookup result shares a value (by norm())
    with one of this call's argument values -- i.e. the call's target is one of
    several candidates the lookup returned, not just any unrelated multi-record list."""
    target_vals = {norm(v) for v in call_args.values() if v is not None}
    if not target_vals:
        return False
    for rec in records:
        field_vals = rec.values() if isinstance(rec, dict) else [rec]
        if any(norm(v) in target_vals for v in field_vals):
            return True
    return False


def triggers(call: Call, state, gate: str) -> list[str]:
    if is_read_only(call.function) and call.function not in ("track_order",):
        return []
    out = []
    if gate in ("confirm", "authorize"):
        out.append("gated_action")
    if _SELF_CORR.search(state.current.user_text or ""):
        out.append("self_correction")
    all_turns = list(state.turns) + ([state.current] if state.current else [])
    read_only_tools = {c.function for t in all_turns for c, _ in t.results if is_read_only(c.function)}
    for tool in read_only_tools:
        for res in state.lookup_results(tool):
            if isinstance(res, list) and len(res) > 1 and _record_matches_call(res, call.args):
                out.append("multiple_candidates"); break
    seen = state.mentioned_values()
    for k in KEY_ARGS.get(call.function, ()):
        v = call.args.get(k)
        if v is None:
            continue
        if norm(v) not in seen and not (numbers_in_text(str(v)) & seen):
            out.append("unmentioned_value"); break
    return sorted(set(out))


class Verifier:
    def __init__(self, model: str = VERIFIER_MODEL, client=None):
        self.model, self.client = model, client or OpenRouterClient()

    def check(self, call: Call, state, policy_text: str) -> dict:
        turns = state.turns + ([state.current] if state.current else [])
        transcript = "\n".join(f"User: {t.user_text}\nAssistant: {t.agent_text}" for t in turns)
        results = [(c.function, r.get("result")) for t in turns for c, r in t.results]
        msg = PROMPT.format(policy=policy_text, transcript=transcript, results=json.dumps(results)[:4000],
                            call=json.dumps(call.as_dict()))
        t0 = time.perf_counter()
        try:
            r = self.client.complete(self.model, [{"role": "user", "content": msg}])
            out = json.loads(r["content"])
            verdict = out.get("verdict") if out.get("verdict") in ("ok", "wrong_target", "ambiguous", "missing_value") else "ok"
            res = {"verdict": verdict, "candidates": [str(c) for c in out.get("candidates", [])],
                   "reason": str(out.get("reason", "")), "usage": r.get("usage", {})}
        except Exception as e:  # network or parse failure: rules decision stands
            res = {"verdict": "ok", "candidates": [], "reason": f"unparseable: {str(e)[:80]}", "usage": {}}
        res["latency_ms"] = (time.perf_counter() - t0) * 1000
        return res
