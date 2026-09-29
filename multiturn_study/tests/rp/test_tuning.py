"""Task 19b dev-tuning regressions (each test was written failing first).

Iteration 1 (number/passcode parsing), from SAGE failures on dev_v2 pass 1:
  - housing_mt_10: the ASR wrote the spoken passcode in CJK digits ("三零一六"),
    so no passcode was found; and the agent's read-back "unit twenty-two C"
    did not match unit_id "22C".
  - travel_mt_05: read-back "Lisbon on Tuesday, October 6th" did not match
    date "October 6, 2026" (the year is never spoken).
"""
from rp.gate_checker import read_back, spoken_passcode
from rp.numwords import spoken_digits
from rp.types import Call


# --- CJK digit passcodes ---------------------------------------------------

def test_spoken_passcode_cjk_digits():
    assert spoken_passcode("三零一六") == "3016"
    assert spoken_digits("三零一六") == "3016"


def test_spoken_passcode_cjk_zero_variant():
    assert spoken_passcode("九〇二三") == "9023"


def test_spoken_passcode_cjk_too_short_is_not_a_passcode():
    assert spoken_passcode("三") == ""


# --- spoken alphanumeric identifiers --------------------------------------

DEPOSIT = Call("pay_deposit", {"unit_id": "22C", "amount": 2400})


def test_read_back_spoken_alphanumeric_identifier():
    text = ("I can pay the two thousand four hundred dollar deposit for unit twenty-two C. "
            "Please provide your authorization passcode.")
    assert read_back(DEPOSIT, text)


def test_read_back_spoken_alphanumeric_identifier_wrong_letter():
    text = "I can pay the two thousand four hundred dollar deposit for unit twenty-two B."
    assert not read_back(DEPOSIT, text)


def test_read_back_spoken_alphanumeric_identifier_wrong_number():
    text = "I can pay the two thousand four hundred dollar deposit for unit twenty-three C."
    assert not read_back(DEPOSIT, text)


def test_read_back_spoken_digit_by_digit_identifier():
    call = Call("cancel_booking", {"pnr": "KP229"})
    assert read_back(call, "I'll cancel booking K P two two nine. Please state your passcode.")
    assert not read_back(call, "I'll cancel booking K P two two eight. Please state your passcode.")


# --- dates with an unspoken year ------------------------------------------

FLIGHT = Call("book_flight", {"destination": "Lisbon", "date": "October 6, 2026", "nonstop": True})


def test_read_back_date_without_year():
    assert read_back(FLIGHT, "I can book a non-stop flight to Lisbon on Tuesday, October 6th. Passcode?")


def test_read_back_date_wrong_day_rejected():
    assert not read_back(FLIGHT, "I can book a non-stop flight to Lisbon on October 7th. Passcode?")
    assert not read_back(FLIGHT, "I can book a non-stop flight to Lisbon on October 16th. Passcode?")


def test_read_back_date_wrong_year_rejected():
    assert not read_back(FLIGHT, "I can book a flight to Lisbon on October 6th, 2027. Passcode?")


def test_read_back_date_spoken_ordinal_words():
    assert read_back(FLIGHT, "A flight to Lisbon on October sixth. Passcode?")


def test_read_back_iso_date_value():
    call = Call("book_flight", {"destination": "Lisbon", "date": "2026-10-06"})
    assert read_back(call, "A flight to Lisbon on October 6th. Passcode?")
    assert not read_back(call, "A flight to Lisbon on October 9th. Passcode?")


# --- Iteration 2: authorize re-prompt after an already-verified passcode -----
# gemini-3.8-live pass 1: on 6 passcode turns (ecommerce_mt_12 t4, housing_mt_10 t3,
# negotiation_mt_01 t4, negotiation_mt_06 t3, travel_mt_05 t3, travel_mt_07 t3) the
# agent called verify_authorization, the action was blocked, and it retried the
# same call until the 4-round safety valve tripped, leaving the user with no reply.

from rp.reprompts import block_message  # noqa: E402

PAY = Call("pay_deposit", {"unit_id": "22C", "amount": 2400})


def test_authorize_reprompt_says_stop_retrying_after_verified_passcode():
    detail = block_message("authorize", PAY)["detail"]
    low = detail.lower()
    assert "already called verify_authorization" in low
    assert "do not call" in low and "again in this turn" in low
    assert "tell the user" in low


def test_authorize_reprompt_still_gives_the_normal_path():
    low = block_message("authorize", PAY)["detail"].lower()
    assert "passcode" in low and "verify_authorization" in low
    assert "unit_id=22c" in low and "amount=2400" in low


# --- Iteration 3: confirm re-prompt when the user moved on ---------------------
# dev_v2 passes 1/11/12 (both agents): after the agent asked a yes/no question the
# user changed the subject (counseling_mt_02 t3, counseling_mt_12 t3, finance_mt_03
# t3, finance_mt_10 t4, housing_mt_02 t3); the agent retried the pending gated call,
# SAGE blocked it ("no assent"), and the agent repeated the confirmation question
# instead of handling what the user had just asked.

CANCEL = Call("cancel_card", {"last4": "7723"})


def test_confirm_reprompt_handles_a_changed_subject():
    low = block_message("confirm", CANCEL)["detail"].lower()
    assert "did not answer" in low and "asked for something else" in low
    assert "do what they asked" in low and "do not repeat the question" in low


def test_confirm_reprompt_keeps_readback_and_no_retry():
    low = block_message("confirm", CANCEL)["detail"].lower()
    assert "not executed" in low and "read back" in low and "last4=7723" in low
    assert "do not call this tool again until they agree" in low
