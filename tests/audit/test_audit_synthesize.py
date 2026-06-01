from scripts.audit_synthesize import synthesize_verdict

def test_clean_keep():
    assert synthesize_verdict([], {"discriminates": True}, None)["verdict"] == "KEEP"

def test_single_layer_disqualifying_capped_at_weak():
    assert synthesize_verdict(["untrippable_trap"], {}, None)["verdict"] == "WEAK"

def test_cut_requires_two_layers_with_disqualifying():
    assert synthesize_verdict(["untrippable_trap"], {}, "meaningless")["verdict"] == "CUT"

def test_fix_two_fixable_layers():
    assert synthesize_verdict(["missing_scenario_field"], {"malformed_magnet": True}, None)["verdict"] == "FIX"

def test_low_signal_alone_is_weak():
    assert synthesize_verdict(["naive_passes"], {}, "trivial")["verdict"] == "WEAK"

def test_trippability_unverified_is_low_not_disqualifying():
    # a must_not-trap trippability flag must NOT condemn, even with an L3 misspecified verdict it stays FIX-or-weaker, never CUT
    assert synthesize_verdict(["trippability_unverified"], {}, None)["verdict"] == "WEAK"
    assert synthesize_verdict(["trippability_unverified"], {}, "meaningless")["verdict"] != "CUT"

def test_deny_signal_cannot_drive_fix():
    v = synthesize_verdict([], {"nontrap_all_refuse": True, "never_triggered_trap": True}, "misspecified")
    assert v["verdict"] == "WEAK"

def test_records_layers():
    v = synthesize_verdict(["untrippable_trap"], {}, "meaningless")
    assert "L1" in v["layers"] and "L3" in v["layers"]
