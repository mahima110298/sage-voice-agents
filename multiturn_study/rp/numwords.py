from __future__ import annotations
import re

_UNITS = {"zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
          "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
          "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
         "eighty": 80, "ninety": 90}
_SCALES = {"hundred": 100, "thousand": 1000, "million": 1_000_000}
_DIGIT = {w: str(v) for w, v in _UNITS.items() if v < 10}
# CJK digit characters: the ASR sometimes writes a spoken passcode in Chinese
# numerals ("三零一六"); each character is one digit word.
_CJK_DIGIT = {c: str(i) for i, c in enumerate("零一二三四五六七八九")}
_CJK_DIGIT["〇"] = "0"
_DIGIT.update(_CJK_DIGIT)
_DIGIT_TOKEN = re.compile(r"[a-z]+|[" + "".join(_CJK_DIGIT) + "]")


def norm(v) -> str:
    s = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", str(v).lower())
    return re.sub(r"[^a-z0-9]", "", s)


def spoken_digits(text: str) -> str:
    """Longest run of single-digit words, as digits."""
    best, run = "", ""
    for w in _DIGIT_TOKEN.findall(text.lower()):
        if w in _DIGIT:
            run += _DIGIT[w]
            best = max(best, run, key=len)
        else:
            run = ""
    return best


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+|\d+", text.lower().replace("-", " "))


def _digit_run_ranges(tokens: list[str]) -> list[tuple[int, int]]:
    """[start, end) index ranges of 2+ consecutive bare digit words (0-9) — a spoken
    digit sequence like "eight two eight one", not a number to sum."""
    ranges, i = [], 0
    while i < len(tokens):
        if tokens[i] in _DIGIT:
            j = i
            while j < len(tokens) and tokens[j] in _DIGIT:
                j += 1
            if j - i >= 2:
                ranges.append((i, j))
            i = j
        else:
            i += 1
    return ranges


def _word_numbers(text: str) -> set[str]:
    # A run of 2+ consecutive bare digit words (0-9) is a digit sequence, not a sum —
    # those token positions are excluded entirely (numbers_in_text reports the run
    # itself via _digit_word_runs), so "eight two eight one" contributes nothing here
    # instead of collapsing into 19 or leaking single-digit fragments.
    # A tens word directly after a teen or unit word (no scale word between) flushes
    # the number in progress: "nineteen eighty five" -> {19, 85}. "twenty one" (tens
    # then unit) and "...hundred twenty" (scale then tens) are unaffected.
    tokens = _tokenize(text)
    exclude = set()
    for s, e in _digit_run_ranges(tokens):
        exclude.update(range(s, e))

    out, total, cur, active, last_kind = set(), 0, 0, False, None
    for idx, w in enumerate(tokens):
        if idx in exclude:
            if active:
                out.add(str(total + cur))
            total, cur, active, last_kind = 0, 0, False, None
            continue
        if w in _UNITS:
            if last_kind == "unit" and active:
                out.add(str(total + cur))
                total, cur = 0, 0
            cur += _UNITS[w]; active = True; last_kind = "unit"
        elif w in _TENS:
            if last_kind == "unit" and active:
                out.add(str(total + cur))
                total, cur = 0, 0
            cur += _TENS[w]; active = True; last_kind = "tens"
        elif w in _SCALES and active:
            if _SCALES[w] == 100:
                cur *= 100
            else:
                total += cur * _SCALES[w]; cur = 0
            last_kind = "scale"
        elif w == "and" and active:
            continue
        else:
            if active:
                out.add(str(total + cur))
            total, cur, active, last_kind = 0, 0, False, None
    if active:
        out.add(str(total + cur))
    return out


def _digit_word_runs(text: str) -> list[str]:
    """Every digit-word run of length >= 2, concatenated to a digit string each —
    not just the longest (that's what spoken_digits, the passcode extractor, is for)."""
    tokens = _tokenize(text)
    return ["".join(_DIGIT[tokens[i]] for i in range(s, e)) for s, e in _digit_run_ranges(tokens)]


def _canonical_decimal(n: str) -> str:
    """Strip trailing fractional zeros, then a trailing dot, so "1590.50",
    "1590.00", and "1590.5" reduce to a shared canonical form."""
    if "." in n:
        n = n.rstrip("0").rstrip(".")
    return n


def numbers_in_text(text: str) -> set[str]:
    found = set()
    for m in re.findall(r"\d[\d,]*(?:\.\d+)?", text):
        found.add(_canonical_decimal(m.replace(",", "")))
    found |= _word_numbers(text)
    found |= set(_digit_word_runs(text))
    sd = spoken_digits(text)
    if len(sd) >= 2:
        found.add(sd)
    return found


def digitized_tokens(text: str) -> list[str]:
    """The text's alphanumeric tokens with spoken numbers turned into digits:
    a run of 2+ bare digit words becomes one digit string ("two two nine" ->
    "229"), any other number-word group becomes its value ("twenty-two" -> "22",
    "one" -> "1"). Other tokens are kept. Lets a read-back of an identifier said
    aloud ("unit twenty-two C", "K P two two nine") be matched token by token."""
    tokens = _tokenize(text)
    digit_runs = {s: e for s, e in _digit_run_ranges(tokens)}
    out, i = [], 0
    while i < len(tokens):
        if i in digit_runs:
            e = digit_runs[i]
            out.append("".join(_DIGIT[t] for t in tokens[i:e])); i = e
            continue
        if tokens[i] in _UNITS or tokens[i] in _TENS:
            j = i
            while j < len(tokens) and j not in digit_runs and (
                    tokens[j] in _UNITS or tokens[j] in _TENS or tokens[j] in _SCALES
                    or (tokens[j] == "and" and j + 1 < len(tokens) and
                        (tokens[j + 1] in _UNITS or tokens[j + 1] in _TENS))):
                j += 1
            nums = _word_numbers(" ".join(tokens[i:j]))
            # one number -> its digits; a group that splits into several numbers
            # ("nineteen eighty five") is ambiguous as an identifier: keep the words
            out.extend(nums if len(nums) == 1 else tokens[i:j])
            i = j
            continue
        out.append(tokens[i]); i += 1
    return out


# ---- dates -----------------------------------------------------------------
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"])}
_MONTHS.update({m[:3]: i for m, i in list(_MONTHS.items())})
_MONTHS["sept"] = 9
_ORD_UNITS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
              "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11,
              "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
              "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19,
              "twentieth": 20, "thirtieth": 30}
_MONTH_RE = "|".join(sorted(_MONTHS, key=len, reverse=True))
_DAY_RE = (r"(\d{1,2})(?:st|nd|rd|th)?|((?:twenty|thirty)[\s-]+(?:" + "|".join(_ORD_UNITS) + r")|"
           + "|".join(_ORD_UNITS) + ")")


def _day_value(num: str | None, word: str | None) -> int | None:
    if num:
        return int(num)
    if word:
        parts = re.split(r"[\s-]+", word)
        if len(parts) == 2:
            return _TENS[parts[0]] + _ORD_UNITS[parts[1]]
        return _ORD_UNITS[parts[0]]
    return None


def dates_in_text(text: str) -> list[tuple[int, int, int | None]]:
    """(month, day, year or None) for every "October 6th[, 2026]" / "6th of October"
    / "October sixth" in the text."""
    t = text.lower()
    out = []
    for m in re.finditer(rf"\b({_MONTH_RE})\.?\s+(?:the\s+)?(?:{_DAY_RE})\b(?:,?\s+(\d{{4}}))?", t):
        d = _day_value(m.group(2), m.group(3))
        out.append((_MONTHS[m.group(1)], d, int(m.group(4)) if m.group(4) else None))
    for m in re.finditer(rf"\b(?:the\s+)?(?:{_DAY_RE})\s+of\s+({_MONTH_RE})\b(?:,?\s+(\d{{4}}))?", t):
        d = _day_value(m.group(1), m.group(2))
        out.append((_MONTHS[m.group(3)], d, int(m.group(4)) if m.group(4) else None))
    return [x for x in out if x[1] is not None and 1 <= x[1] <= 31]


def parse_date(value) -> tuple[int, int, int | None] | None:
    """A tool-argument date ("October 6, 2026", "May 12th", "2026-10-06") as
    (month, day, year or None); None if the value is not a date."""
    s = str(value).strip()
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        return int(m.group(2)), int(m.group(3)), int(m.group(1))
    found = dates_in_text(s)
    return found[0] if len(found) == 1 else None
