"""Mock backend. Same lookup mapping as the pilot harness, but verify_authorization
checks the case's own passcode (pilot: approved anything)."""
from __future__ import annotations
import re
from rp.types import Call

MOCK_KEY_TO_TOOL = {"open_deals": "list_deals", "enrolled_courses": "list_enrolled_courses",
                    "upcoming_appointments": "list_appointments", "upcoming_checkins": "list_appointments"}
CONTEXT_SCALARS = ("client_id",)


def _digits(v) -> str:
    from rp.numwords import spoken_digits
    s = str(v)
    d = re.sub(r"\D", "", s)
    return d or spoken_digits(s)


class MockBackend:
    def __init__(self, case: dict):
        self._passcode = str(case.get("auth_passcode") or "")
        md = case.get("mock_data") or {}
        self._data = {MOCK_KEY_TO_TOOL.get(k, k): v for k, v in md.items() if k not in CONTEXT_SCALARS}
        self._ctx = {k: v for k, v in md.items() if k in CONTEXT_SCALARS}

    def peek(self, tool: str):
        return self._data.get(tool)

    def respond(self, call: Call) -> dict:
        if call.function == "verify_authorization":
            ok = bool(self._passcode) and _digits(call.args.get("passcode", "")) == self._passcode
            return {"result": {"authorized": ok}}
        if call.function in self._data:
            return {"result": self._data[call.function]}
        return {"result": {"status": "success"}}

    def session_context(self) -> str:
        if not self._ctx:
            return ""
        pairs = ", ".join(f"{k}={v}" for k, v in self._ctx.items())
        return ("\n\n## Session context\nYou are already in an authenticated session with this "
                f"user: {pairs}. Use these values directly when a tool needs them; do NOT ask "
                "the user to state them.")

    def context_values(self) -> dict:
        return dict(self._ctx)
