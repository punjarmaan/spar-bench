from scripts.audit_report import render_report, go_no_go

def _v(verdict, split="main", concerns=None, content=None):
    return {"sample_id": "s", "split": split, "cell": ["routing","easy",False,"explicit"],
            "verdict": verdict, "concerns": concerns or [], "content_verdict": content,
            "l2_context": {"nontrap_all_refuse": True}}

def test_go_no_go_run_when_mostly_keep():
    g = go_no_go([_v("KEEP") for _ in range(95)] + [_v("WEAK") for _ in range(5)])
    assert g["recommend"] == "RUN" and g["keep_frac"] >= 0.9

def test_go_no_go_hold_when_many_cut():
    assert go_no_go([_v("KEEP") for _ in range(60)] + [_v("CUT") for _ in range(40)])["recommend"] == "FIX_FIRST"

def test_render_sections_incl_always_deny_and_review_queue():
    rows = [_v("KEEP"),
            _v("WEAK", concerns=[["L1","LOW","trippability_unverified"]]),
            _v("WEAK", concerns=[["L3","FIX","misspecified"]], content="misspecified")]
    md = render_report(rows)
    for s in ("# Spar Sample Validity Audit", "## Per-split verdicts", "## Go / No-Go",
              "## Review queue", "always-deny", "trippability_unverified", "misspecified"):
        assert s in md
