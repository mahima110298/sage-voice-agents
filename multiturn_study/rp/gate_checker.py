"""Gate satisfaction checks for SAGE's runtime guard: given the gate a tool call
requires (execute / clarify / confirm / authorize), decide from the conversation
whether it has been satisfied.

Known limitation: read-back is checked over the previous agent turn as a whole
(`state.prior_agent_text()`), not per sentence — a read-back phrase and an
unrelated stray number said in the same agent turn are equally "in scope" for
the read-back check.
"""
from __future__ import annotations
import re
from harness_defs import DOMAIN_TOOLS
from rp.numwords import numbers_in_text, norm, spoken_digits, digitized_tokens, parse_date, dates_in_text
from rp.tools import KEY_ARGS, is_read_only
from rp.types import Call

_YES = re.compile(r"\b(yes|yeah|yep|yup|correct|confirmed?|that'?s right|right|go ahead|do it|sure|"
                  r"please do|sounds good|affirmative|ok(ay)?|proceed)\b")
_NO = re.compile(r"\b(no|nope|don'?t|stop|wait|never ?mind|cancel that|hold on|not|actually)\b")
# A modification cue means the user is changing the action, not assenting to it —
# even alongside a yes-word ("sure, but change it to 300").
_MOD = re.compile(r"\b(but|instead|change|make it|rather|different)\b")
# Whether the PREVIOUS agent turn asked the user to state a passcode — gates the
# passcode-digit exemption in assented() (see fix round 3): only a turn that
# actually asked for a passcode can make "seven four two nine" read as a passcode
# rather than a competing number.
_ASKED_PASSCODE = re.compile(r"passcode|pass code|\bpin\b|authori[sz]ation code|\bcode\b", re.IGNORECASE)

# Identifier args that name WHAT an action targets (as opposed to amounts/free text):
# a read-back can point at these indirectly, via a lookup record's descriptive field
# instead of the raw identifier itself (e.g. "cancel the desk lamp order" instead of
# "cancel order QJ881"). "type" and "label" are excluded: both are common words
# ("checking", "group") too likely to appear in unrelated text and collide across
# records, so they are not safe stand-ins for an identifier.
IDENTIFIER_ARGS = {"sku", "order_id", "pnr", "unit_id", "deal_id", "course_code",
                   "appointment_id", "last4"}
DESCRIPTIVE_FIELDS = ("name", "item", "title", "description", "product", "course_name",
                      "provider", "destination")


def _as_float(x):
    try:
        return float(str(x).replace(",", "").replace("$", "").strip())
    except ValueError:
        return None


def assented(text: str, call: Call | None = None, allow_passcode: bool = False) -> bool:
    t = text.lower()
    if not _YES.search(t) or _NO.search(t):
        return False
    if _MOD.search(t):
        return False
    if call is not None:
        allowed = {fv for v in call.args.values() for fv in [_as_float(v)] if fv is not None}
        pc_f = None
        if allow_passcode:
            pc = spoken_passcode(text)
            pc_f = _as_float(pc) if pc else None
        for n in numbers_in_text(text):
            fv = _as_float(n)
            if fv is None or fv < 10:
                continue  # identifiers/amounts in this benchmark are multi-digit;
                          # small numbers are filler ("option 2", "the ... one")
            if pc_f is not None and fv == pc_f:
                continue  # this is the passcode itself, not a mismatched argument
            if fv not in allowed:
                return False
    return True


def _spoken_identifier(nv: str, text: str) -> bool:
    """True if some run of consecutive tokens of the text, with spoken numbers
    turned into digits, spells exactly the identifier ("unit twenty-two C" ->
    "22" "c" -> "22c"). Whole tokens only, so "twenty-two cats" is not "22C"."""
    toks = digitized_tokens(text)
    for i in range(len(toks)):
        acc = ""
        for tok in toks[i:]:
            acc += tok
            if acc == nv:
                return True
            if len(acc) >= len(nv) or not nv.startswith(acc):
                break
    return False


def _date_present(date, text: str) -> bool:
    """A date argument is read back if the text names the same month and day and
    either no year or the same year (the year is rarely spoken)."""
    month, day, year = date
    return any(m == month and d == day and (y is None or year is None or y == year)
               for m, d, y in dates_in_text(text))


def _present(value, text: str) -> bool:
    nums = numbers_in_text(text)
    fv = _as_float(value)
    if fv is not None:  # numeric argument: compare by value, never by prefix
        return any(_as_float(n) == fv for n in nums)
    date = parse_date(value)
    if date is not None:
        return _date_present(date, text)
    nv = norm(value)
    return bool(nv) and (nv in norm(text) or nv in {norm(n) for n in nums} or _spoken_identifier(nv, text))


def _norm_phrase(s) -> str:
    """Lowercase, collapse everything but letters/digits to single spaces — unlike
    rp.numwords.norm (which strips spaces entirely), this keeps word boundaries so a
    descriptive phrase can be matched as whole words, not as a raw substring."""
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def _word_match(value, text_phrase: str) -> bool:
    v = _norm_phrase(value)
    if not v:
        return False
    return bool(re.search(rf"(?<!\S){re.escape(v)}(?!\S)", text_phrase))


def _iter_dicts(obj):
    """Yield every dict found anywhere inside obj (recursing through lists/dicts)."""
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _iter_dicts(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_dicts(item)


def _domain_of(function: str) -> str | None:
    for domain, tools in DOMAIN_TOOLS.items():
        if function in tools:
            return domain
    return None


def _lookup_records(call: Call, state) -> list:
    """Records available to the guard as candidate read-back sources: every read-only
    tool's results seen so far, plus (since a guard may consult the backend directly)
    a peek at the read-only tools belonging to the call's own domain."""
    if state is None:
        return []
    records = []
    all_tools = {tool for domain_tools in DOMAIN_TOOLS.values() for tool in domain_tools}
    for tool in all_tools:
        if is_read_only(tool):
            records.extend(state.lookup_results(tool))
    backend = getattr(state, "backend", None)
    if backend is not None:
        for tool in DOMAIN_TOOLS.get(_domain_of(call.function) or "", ()):
            if is_read_only(tool):
                peeked = backend.peek(tool)
                if peeked is not None:
                    records.append(peeked)
    return records


def _field_value(rec: dict, field: str) -> str:
    v = rec.get(field)
    return _norm_phrase(v) if v is not None else ""


def _field_unique(records: list, field: str, value: str, self_recs: list) -> bool:
    """True if no record OTHER than self_recs shares this normalized field value."""
    return not any(_field_value(r, field) == value for r in records
                   if r not in self_recs and field in r)


def _described(key: str, value, text: str, call: Call, state) -> bool:
    """True if some lookup record's identifier field (key) equals value, and that
    SAME record has a descriptive field whose value is named (as a whole word/phrase)
    in text — a read-back by description ("the desk lamp order") rather than by raw
    identifier. Guards against ambiguity: the descriptive value must be unique to
    that record (not shared by another record with a different identifier), and the
    text must not also name another record — by its identifier or by ITS OWN unique
    description — which would mean the user is pointing at a different record."""
    if key not in IDENTIFIER_ARGS:
        return False
    target = norm(value)
    if not target:
        return False
    text_phrase = _norm_phrase(text)
    records = [rec for group in _lookup_records(call, state) for rec in _iter_dicts(group) if key in rec]
    target_recs = [r for r in records if norm(r[key]) == target]
    if not target_recs:
        return False
    other_recs = [r for r in records if norm(r[key]) != target]

    for field in DESCRIPTIVE_FIELDS:
        trec = next((r for r in target_recs if field in r), None)
        if trec is None:
            continue
        dv = _field_value(trec, field)
        if not dv or not _word_match(dv, text_phrase):
            continue
        if not _field_unique(records, field, dv, target_recs):
            continue  # shared with another record: not a safe stand-in (rule a)
        conflict = False
        for r in other_recs:
            rid = _field_value(r, key)
            if rid and _word_match(rid, text_phrase):
                conflict = True
                break
            for f2 in DESCRIPTIVE_FIELDS:
                if f2 not in r:
                    continue
                rv = _field_value(r, f2)
                if rv and _word_match(rv, text_phrase) and _field_unique(records, f2, rv, [r]):
                    conflict = True
                    break
            if conflict:
                break
        if conflict:
            continue  # text also names another record: ambiguous (rule b)
        return True
    return False


def read_back(call: Call, text: str, state=None) -> bool:
    keys = [k for k in KEY_ARGS.get(call.function, ()) if k in call.args]
    if not text:
        return False
    for k in keys:
        value = call.args[k]
        if _present(value, text) or _described(k, value, text, call, state):
            continue
        return False
    return True


def spoken_passcode(text: str) -> str:
    d = spoken_digits(text)
    if len(d) >= 4:
        return d
    m = re.findall(r"\b\d{4,6}\b", text)
    if m:
        return m[-1]
    spaced = re.findall(r"\b\d(?:[\s,-]+\d){3,5}\b", text)
    return re.sub(r"\D", "", spaced[-1]) if spaced else ""


def _is_action_identifier(pc: str, call: Call) -> bool:
    """True if the spoken digits equal one of the call's own argument values — as a
    digit string, or numerically (so a passcode "2000" is rejected against an amount
    of 2000.0 or "$2,000.00", not just against amounts already spelled as "2000")."""
    pc_f = _as_float(pc)
    for v in call.args.values():
        if re.sub(r"\D", "", str(v)) == pc:
            return True
        fv = _as_float(v)
        if fv is not None and pc_f is not None and fv == pc_f:
            return True
    return False


def _state_changing(call: Call) -> bool:
    return call.function != "verify_authorization" and not is_read_only(call.function)


def is_satisfied(gate: str, call: Call, state, credential_check: bool = True) -> tuple[bool, str]:
    if gate == "execute":
        return True, "execute"
    prev = state.prior_agent_text()
    user = state.current.user_text
    if gate == "clarify":
        return False, "clarification required"
    if not read_back(call, prev, state):
        return False, "no read-back of this action on the previous turn"
    if gate == "confirm" or (gate == "authorize" and not credential_check):
        allow_passcode = (gate == "authorize" and not credential_check and
                          bool(_ASKED_PASSCODE.search(prev)))
        return (True, "confirmed") if assented(user, call, allow_passcode) else (False, "no assent this turn")
    pc = spoken_passcode(user)
    if not pc:
        return False, "no passcode spoken this turn"
    if _is_action_identifier(pc, call):
        return False, "spoken digits are an identifier of the action, not a passcode"
    matched_results = [res for c, res in state.current.results
                        if c.function == "verify_authorization" and
                        re.sub(r"\D", "", str(c.args.get("passcode", ""))) == pc]
    if not matched_results:
        return False, "verify_authorization not called with the spoken passcode"
    if not any(((res or {}).get("result") or {}).get("authorized") is True for res in matched_results):
        return False, "verify_authorization did not confirm this passcode"
    if any(_state_changing(c) and c != call for c in state.current.calls):
        return False, "passcode already released a different action this turn"
    return True, "authorized"
