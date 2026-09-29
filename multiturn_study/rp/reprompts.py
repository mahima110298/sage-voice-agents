"""Gate-specific re-prompt messages returned to the agent when SAGE blocks a call.

Each message tells the agent (1) the call did not execute and (2) exactly what
kind of pause to produce next — clarify / confirm / authorize — so the agent
does not just retry the same call.
"""
from __future__ import annotations
from rp.tools import KEY_ARGS
from rp.types import Call


def _values(call: Call) -> str:
    return ", ".join(f"{k}={call.args[k]}" for k in KEY_ARGS.get(call.function, ()) if k in call.args)


def _clarify_detail(call: Call, candidates: list[str], verdict: str | None, reason: str) -> str:
    what = f"{call.function}({_values(call)})"
    suffix = f" ({reason})" if reason else ""
    if verdict == "missing_value":
        return (f"This action was NOT executed. {what} is missing a value it needs{suffix}. "
                "Ask the user for the missing value. Do not call this tool again now.")
    if verdict == "wrong_target":
        return (f"This action was NOT executed. What the user asked for may not match {what}{suffix}. "
                "Say what the user asked for and ask which one they mean. Do not call this tool again now.")
    # ambiguous, or no verdict given: name the candidates and ask which.
    return (f"This action was NOT executed. More than one reading fits: {', '.join(candidates) or 'the target is unclear'}. "
            "Ask the user which one they mean, naming the options. Do not call this tool again now.")


def block_message(gate: str, call: Call, candidates: list[str] = (), verdict: str | None = None,
                   reason: str = "") -> dict:
    if gate == "clarify":
        return {"error": "BLOCKED_BY_POLICY", "required_gate": gate,
                "detail": _clarify_detail(call, candidates, verdict, reason)}
    what = f"{call.function}({_values(call)})"
    if gate == "confirm":
        detail = (f"This action was NOT executed. Before {what}, read back the specific action and its key values "
                  "and ask the user for a yes or no. Do not call this tool again until they agree. "
                  "If their latest message did not answer that question and asked for something else, do what they "
                  "asked now and do not repeat the question in this turn.")
    else:  # authorize
        detail = (f"This action was NOT executed. {what} needs the user's authorization passcode. Read back the "
                  "action and its key values and ask the user to state their passcode. When they do, call "
                  "verify_authorization with exactly the digits they gave as their passcode, then this tool, in the same turn. "
                  "If you already called verify_authorization in this turn and this tool was still blocked, do not call "
                  "either tool again in this turn: tell the user the action did not go through, read back the action and "
                  "its key values, and ask them to state their passcode again.")
    return {"error": "BLOCKED_BY_POLICY", "required_gate": gate, "detail": detail}
