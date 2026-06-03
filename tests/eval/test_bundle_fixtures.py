"""write_fixtures picks a diverse, real example set for cofounder frontend dev (T8)."""
import json
from pathlib import Path

from spar.eval.bundle import write_fixtures


def test_write_fixtures_selects_diverse_episodes(tmp_path: Path):
    bundle = tmp_path / "bundle"
    eps = bundle / "m1" / "episodes"; eps.mkdir(parents=True)
    for sid, status, is_trap in [("clean", "scored", False), ("trap", "scored", True),
                                 ("bad", "malformed_action", False)]:
        (eps / f"{sid}.jsonl").write_text(json.dumps(
            {"schema_version": 1, "sample_id": sid, "trial_index": 0, "status": status,
             "is_trap": is_trap, "turns": [], "episode": {}}) + "\n")
    (bundle / "index.json").write_text(json.dumps({"models": [
        {"id": "m1", "samples": [
            {"sample_id": "clean", "axis": "routing", "is_trap": False},
            {"sample_id": "trap", "axis": "consent_mandate", "is_trap": True},
            {"sample_id": "bad", "axis": "routing", "is_trap": False}]}]}))
    out = write_fixtures(bundle_dir=bundle, max_examples=6)
    names = {p.name for p in (bundle / "fixtures").glob("*.jsonl")}
    assert "trap.jsonl" in names                       # a trap example is always included
    assert (bundle / "fixtures" / "index.json").is_file()
    assert len(out) >= 2
