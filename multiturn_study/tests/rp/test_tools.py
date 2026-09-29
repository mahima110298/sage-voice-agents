from rp.tools import schemas_for, is_read_only, declared_tier, KEY_ARGS, OVERLAY


def test_overlay_applied_to_transfer_funds():
    s = {t["name"]: t for t in schemas_for("finance_billing")}
    assert "Requires an authorization passcode unless" in s["transfer_funds"]["description"]
    assert "CRITICAL severity" in s["transfer_funds"]["description"]  # original text kept


def test_verify_authorization_replaced():
    s = {t["name"]: t for t in schemas_for("finance_billing")}
    assert s["verify_authorization"]["description"].startswith("Verify the passcode the user just stated")


def test_every_domain_has_verify_and_overlay_is_deterministic():
    from harness_defs import DOMAIN_TOOLS
    for d in DOMAIN_TOOLS:
        names = [t["name"] for t in schemas_for(d)]
        assert "verify_authorization" in names
        assert schemas_for(d) == schemas_for(d)


def test_read_only_and_tier():
    assert is_read_only("list_accounts") and not is_read_only("log_mood_entry")
    assert declared_tier("transfer_funds") == "critical"
    assert declared_tier("place_order") == "critical"  # HIGH/CRITICAL reads the higher


def test_key_args_cover_all_state_changing_tools():
    from harness_defs import TOOL_SCHEMAS_REALTIME
    for name in TOOL_SCHEMAS_REALTIME:
        if not is_read_only(name):
            assert name in KEY_ARGS, name


def test_key_args_exist_in_schema():
    from harness_defs import TOOL_SCHEMAS_REALTIME
    for name, keys in KEY_ARGS.items():
        props = TOOL_SCHEMAS_REALTIME[name]["parameters"].get("properties", {})
        assert set(keys) <= set(props), (name, keys, list(props))
