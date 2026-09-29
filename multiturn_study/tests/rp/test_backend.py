from rp.backend import MockBackend
from rp.types import Call

CASE = {"auth_passcode": "7539", "mock_data": {
    "list_accounts": [{"account": "4410", "type": "checking"}], "client_id": "wc_9"}}


def test_passcode_checked():
    b = MockBackend(CASE)
    assert b.respond(Call("verify_authorization", {"passcode": "7539"}))["result"]["authorized"] is True
    assert b.respond(Call("verify_authorization", {"passcode": "4410"}))["result"]["authorized"] is False


def test_passcode_accepts_spaced_or_worded_digits():
    b = MockBackend(CASE)
    assert b.respond(Call("verify_authorization", {"passcode": "7 5 3 9"}))["result"]["authorized"] is True


def test_lookup_and_action():
    b = MockBackend(CASE)
    assert b.respond(Call("list_accounts"))["result"][0]["account"] == "4410"
    assert b.respond(Call("transfer_funds", {"to_account": "1", "amount": 5}))["result"]["status"] == "success"
    assert b.peek("list_accounts")[0]["account"] == "4410"


def test_context_scalars_not_tools():
    b = MockBackend(CASE)
    assert b.peek("client_id") is None
    assert "client_id=wc_9" in b.session_context()


def test_context_values():
    b = MockBackend(CASE)
    assert b.context_values() == {"client_id": "wc_9"}
