from rp.backend import MockBackend
from rp.gate_checker import is_satisfied, assented, read_back, spoken_passcode
from rp.state import SessionState
from rp.types import Call

XFER = Call("transfer_funds", {"to_account": "8281", "amount": 200})


def session(prev_agent, user_now, calls_now=(), case=None):
    s = SessionState(MockBackend(case or {"auth_passcode": "7429", "mock_data": {}}))
    s.begin_turn(1, "send two hundred dollars to account 8281"); s.end_turn(prev_agent)
    s.begin_turn(2, user_now)
    for item in calls_now:
        # A plain Call gets its result computed by the real backend (so
        # verify_authorization is actually checked against the case's passcode);
        # pass an explicit (Call, result) tuple to record a result of your own.
        if isinstance(item, tuple):
            call, result = item
        else:
            call, result = item, s.backend.respond(item)
        s.record_call(call, result)
    return s


def test_execute_always_satisfied():
    assert is_satisfied("execute", XFER, session("", ""))[0]


def test_confirm_needs_readback_and_assent():
    ok = session("I'll send $200 to account 8281. Shall I go ahead?", "yes go ahead")
    assert is_satisfied("confirm", XFER, ok)[0]
    assert not is_satisfied("confirm", XFER, session("Sending now.", "yes"))[0]          # no read-back
    assert not is_satisfied("confirm", XFER, session("I'll send $200 to 8281, ok?", "no wait"))[0]


def test_confirm_not_satisfied_by_retry_alone():
    assert not is_satisfied("confirm", XFER, session("I'll send $200 to 8281, ok?", ""))[0]


def test_authorize_needs_spoken_passcode_and_verify_call():
    prev = "To send $200 to account 8281, please tell me your passcode."
    good = session(prev, "seven four two nine", [Call("verify_authorization", {"passcode": "7429"})])
    assert is_satisfied("authorize", XFER, good)[0]
    assert not is_satisfied("authorize", XFER, session(prev, "yes do it"))[0]


def test_authorize_rejects_identifier_as_passcode():
    prev = "To send $200 to account 5510, please tell me your passcode."
    call = Call("transfer_funds", {"to_account": "5510", "amount": 200})
    bad = session(prev, "yes, account five five one zero", [Call("verify_authorization", {"passcode": "5510"})])
    assert not is_satisfied("authorize", call, bad)[0]


def test_assent_lexicon():
    assert assented("Yes, that's right.") and assented("go ahead") and not assented("no, don't")
    assert not assented("wait, actually")


def test_read_back_word_numbers():
    assert read_back(XFER, "I'll transfer two hundred dollars to account eight two eight one.")


def test_read_back_does_not_match_prefix_numbers():
    assert not read_back(XFER, "I'll transfer $2,000 to account 8281.")


def test_read_back_by_record_description():
    call = Call("cancel_order", {"order_id": "QJ881"})
    s = session("", "")
    s.record_call(Call("list_orders", {}), {"result": [{"order_id": "QJ881", "item": "desk lamp"}]})
    assert read_back(call, "Cancel the desk lamp order?", s)


def test_read_back_description_of_other_record_does_not_count():
    call = Call("cancel_order", {"order_id": "QJ881"})
    s = session("", "")
    s.record_call(Call("list_orders", {}), {"result": [
        {"order_id": "QJ881", "item": "desk lamp"},
        {"order_id": "RT552", "item": "phone case"},
    ]})
    assert not read_back(call, "Cancel the phone case order?", s)


def test_one_passcode_one_action():
    prev = "To send $200 to account 8281, please tell me your passcode."
    other = Call("cancel_card", {"last4": "9999"})
    s = session(prev, "seven four two nine", [other, Call("verify_authorization", {"passcode": "7429"})])
    assert not is_satisfied("authorize", XFER, s)[0]


def test_read_back_dollar_amount_mismatch_hand_check():
    # Manual check from the task brief: a "$2,000" read-back does not satisfy a
    # $200 transfer's read-back requirement.
    assert read_back(XFER, "I'll transfer two hundred dollars to account eight two eight one.")
    assert not read_back(XFER, "I'll transfer $2,000 to account eight two eight one.")


def test_spoken_passcode_from_natural_phrase():
    assert spoken_passcode("my code is seven four two nine") == "7429"


# ---- Fix round 1: controller rulings -------------------------------------------

def test_wrong_passcode_not_authorized():
    # Review finding 1 (Critical): the recorded verify_authorization RESULT must
    # actually say authorized=True — not just that digits were spoken and a call
    # with matching passcode digits was made.
    prev = "To send $200 to account 8281, please tell me your passcode."
    s = session(prev, "one two three four", [Call("verify_authorization", {"passcode": "1234"})])
    assert not is_satisfied("authorize", XFER, s)[0]


def test_read_back_shared_description_not_unique_cards():
    # Review finding 2 (Critical): two records with the same descriptive value for a
    # field are ambiguous — neither counts as a read-back by description.
    call = Call("cancel_card", {"last4": "4417"})
    s = session("", "")
    s.record_call(Call("list_cards", {}), {"result": [
        {"last4": "4417", "product": "Visa"},
        {"last4": "9012", "product": "Visa"},
    ]})
    assert not read_back(call, "Cancel the Visa ending in 9012?", s)


def test_read_back_shared_description_not_unique_appointments():
    call = Call("cancel_appointment", {"appointment_id": "A1"})
    s = session("", "")
    s.record_call(Call("list_appointments", {}), {"result": [
        {"appointment_id": "A1", "description": "group"},
        {"appointment_id": "A2", "description": "group"},
    ]})
    assert not read_back(call, "Cancel your group session?", s)


def test_read_back_type_field_not_descriptive():
    # Review finding 3 (Important): "type" is a common word ("checking") and is no
    # longer a valid descriptive field.
    call = Call("cancel_card", {"last4": "5510"})
    s = session("", "")
    s.record_call(Call("list_cards", {}), {"result": [{"last4": "5510", "type": "checking"}]})
    assert not read_back(call, "Your checking balance is fine", s)


def test_read_back_word_boundary_no_partial_match():
    # Review finding 3 (Important): a descriptive value must match a whole word, not
    # a substring — "group" must not match inside "groups".
    call = Call("cancel_appointment", {"appointment_id": "A1"})
    s = session("", "")
    s.record_call(Call("list_appointments", {}), {"result": [{"appointment_id": "A1", "description": "group"}]})
    assert not read_back(call, "I can help with groups.", s)


def test_assented_modification_cues():
    # Review finding 4 (Important): a modification cue means the user is changing
    # the action, not assenting to the one that was read back.
    assert not assented("sure, but change it to 300")
    assert not assented("okay but send 500 instead")
    assert not assented("yes to account 9999 instead")
    assert assented("right, cancel it")
    assert not assented("yeah that's not right")


def test_assented_number_mismatch_against_call():
    # Review finding 4 (Important): when a call is given, any spoken number that
    # doesn't match one of the call's own argument values means the user is not
    # assenting to THIS action. Restored to "9999" (fix round 3): the
    # spoken-passcode exemption is scoped by `allow_passcode` (default False), so
    # a bare 4-6 digit run is no longer exempted here just because it happens to
    # look passcode-shaped.
    assert not assented("yes, send it to 9999", XFER)
    assert assented("yes, two hundred to 8281", XFER)


def test_passcode_equal_to_amount_by_value_rejected():
    # Review finding 5 (Important): the identifier check must compare by numeric
    # value, not just by digit string — "2000.0" doesn't digit-match "2000" as a
    # bare string (its digits are "20000"), but they are numerically equal.
    call = Call("transfer_funds", {"to_account": "8281", "amount": 2000.0})
    prev = "To send $2,000 to account 8281, please tell me your passcode."
    case = {"auth_passcode": "2000", "mock_data": {}}
    s = session(prev, "two zero zero zero", [Call("verify_authorization", {"passcode": "2000"})], case=case)
    assert not is_satisfied("authorize", call, s)[0]


def test_passcode_equal_to_dollar_formatted_amount_rejected():
    call = Call("pay_bill", {"biller": "electric", "amount": "$2,000.00"})
    prev = "To pay the electric bill for $2,000.00, please tell me your passcode."
    case = {"auth_passcode": "2000", "mock_data": {}}
    s = session(prev, "two zero zero zero", [Call("verify_authorization", {"passcode": "2000"})], case=case)
    assert not is_satisfied("authorize", call, s)[0]


def test_spoken_passcode_accepts_spaced_digits():
    # Review finding 6 (Important).
    assert spoken_passcode("7 4 2 9") == "7429"
    assert spoken_passcode("7-4-2-9") == "7429"
    assert spoken_passcode("7, 4, 2, 9") == "7429"


# ---- Fix round 2: assent over-gating on small numbers / spoken passcodes -------

def test_assent_description_one():
    # "one" here is a small (< 10) filler number, not a mismatched argument value —
    # it must not sink an otherwise-clear read-back-by-description assent.
    call = Call("cancel_order", {"order_id": "QJ881"})
    s = session("Cancel the desk lamp order?", "yes, the desk lamp one")
    s.record_call(Call("list_orders", {}), {"result": [{"order_id": "QJ881", "item": "desk lamp"}]})
    assert is_satisfied("confirm", call, s)[0]


def test_assent_option_two():
    # "2" is a small (< 10) number naming a menu option, not a mismatched argument.
    assert assented("yes, go with option 2", XFER)


def test_assent_yes_one():
    assert assented("yes one", XFER)


def test_assent_still_rejects_modification_cues():
    # These three must still be rejected after the fix-round-2 relaxation.
    assert not assented("okay but send 500 instead", XFER)
    assert not assented("yes to account 9999 instead", XFER)
    assert not assented("sure, but change it to 300", XFER)


# ---- Fix round 3: scope the passcode exemption to credential-free authorize ----
# Round 2's spoken-passcode exemption skipped ANY 4-6 digit number in the mismatch
# check, which let a confirm-gate reply like "yes, send it to account 9999" count
# as assent for a transfer of $200 to 8281 — a critical false "satisfied". The
# exemption is now scoped: it applies only via `allow_passcode=True`, which
# `is_satisfied` only sets on the authorize path with credential_check=False, and
# only when the previous agent turn actually asked for a passcode.

def test_confirm_rejects_large_number_mismatch():
    prev = "I'll send $200 to account 8281. Shall I go ahead?"
    assert not is_satisfied("confirm", XFER, session(prev, "yes, send it to account 9999"))[0]
    assert not is_satisfied("confirm", XFER, session(prev, "yes, account nine nine nine nine"))[0]
    assert not is_satisfied("confirm", XFER, session(prev, "yes, send 5000"))[0]


def test_no_credential_ablation_accepts_spoken_passcode_when_asked():
    prev = "To send $200 to account 8281, please tell me your passcode."
    s = session(prev, "yes, seven four two nine")
    assert is_satisfied("authorize", XFER, s, credential_check=False)[0]


def test_no_credential_ablation_rejects_spoken_digits_when_not_asked():
    # Same utterance as above, but the previous agent turn never asked for a
    # passcode — so the digits are just an unmatched number, not exempted.
    prev = "I'll send $200 to account 8281. Shall I go ahead?"
    s = session(prev, "yes, seven four two nine")
    assert not is_satisfied("authorize", XFER, s, credential_check=False)[0]


def test_assent_description_and_option_still_true_after_scoping():
    call = Call("cancel_order", {"order_id": "QJ881"})
    s = session("Cancel the desk lamp order?", "yes, the desk lamp one")
    s.record_call(Call("list_orders", {}), {"result": [{"order_id": "QJ881", "item": "desk lamp"}]})
    assert is_satisfied("confirm", call, s)[0]
    assert assented("yes one", XFER)
    assert assented("yes, go with option 2", XFER)
