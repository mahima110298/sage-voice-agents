from rp.backend import MockBackend
from rp.state import SessionState
from rp.types import Call


def test_state_tracks_turns_and_mentions():
    s = SessionState(MockBackend({"mock_data": {"list_accounts": [{"account": "4410"}]}}))
    s.begin_turn(1, "move five hundred to my savings")
    s.record_call(Call("list_accounts"), {"result": [{"account": "4410"}]})
    s.end_turn("I'll move $500 to account 4410. Is that right?")
    s.begin_turn(2, "yes")
    assert "4410" in s.prior_agent_text()
    assert s.lookup_results("list_accounts") == [[{"account": "4410"}]]
    assert {"500", "4410"} <= s.mentioned_values()


def test_mentioned_values_includes_context_values():
    s = SessionState(MockBackend({"mock_data": {"client_id": "wc_1180"}}))
    assert "wc1180" in s.mentioned_values()
