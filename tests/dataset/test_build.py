"""M7 Task 10: build — projected public splits + isolated Private + canary (F13/F14/F17)."""

from __future__ import annotations

import json

from spar.dataset.build import DIAMOND_CAP, build
from spar.dataset.gold_backbone import is_hand_authored_id
from spar.dataset.plan import TARGET_TRAP_FRACTION
from spar.simulator.enums import Axis
from spar.simulator.schemas import Sample


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _build(tmp_path, **kw):
    pub = tmp_path / "public"
    priv = tmp_path / "private"
    build(public_dir=pub, private_dir=priv, build_seed=kw.get("seed", 12345),
          spar_version="1.0.0")
    return pub, priv


def test_public_dir_holds_three_splits_private_dir_is_separate(tmp_path):
    pub, priv = _build(tmp_path)
    for split in ("lite", "main", "diamond"):
        assert (pub / f"{split}.jsonl").exists()
        assert (pub / f"{split}.manifest.json").exists()
    assert (priv / "private.jsonl").exists()
    assert (priv / "private.manifest.json").exists()
    assert not (pub / "private.jsonl").exists()
    assert not (priv / "main.jsonl").exists()


def test_public_splits_are_projected_no_gold_or_hidden_config(tmp_path):
    pub, _ = _build(tmp_path)
    for split in ("lite", "main", "diamond"):
        for obj in _read_jsonl(pub / f"{split}.jsonl"):
            assert "gold" not in obj
            assert "world_config" not in obj
            assert set(obj) == {"sample_id", "axis", "difficulty", "policy_id",
                                "canary", "seed", "mandate", "cart", "methods"}
            for m in obj["methods"]:
                assert "true_fee_bps" not in m and "approval_prob" not in m


def test_private_holds_full_graded_samples(tmp_path):
    _, priv = _build(tmp_path)
    lines = [ln for ln in (priv / "private.jsonl").read_text().splitlines() if ln.strip()]
    assert lines, "private split is empty"
    for line in lines:
        Sample.model_validate_json(line)   # full graded Sample (JSON path handles Decimals)
        obj = json.loads(line)
        assert "gold" in obj
        assert "world_config" in obj


def test_canary_present_on_every_line_with_spar_prefix(tmp_path):
    pub, priv = _build(tmp_path)
    all_objs = []
    for split in ("lite", "main", "diamond"):
        all_objs += _read_jsonl(pub / f"{split}.jsonl")
    all_objs += _read_jsonl(priv / "private.jsonl")
    assert all_objs
    assert all(o["canary"].startswith("spar:") for o in all_objs)
    assert len({o["canary"] for o in all_objs}) == 1


def test_diamond_split_all_diamond_human_authored_and_capped(tmp_path):
    pub, _ = _build(tmp_path)
    objs = _read_jsonl(pub / "diamond.jsonl")
    # F14: every Diamond line is hand-authored (non-procedural id), capped <=198.
    assert all(is_hand_authored_id(o["sample_id"]) for o in objs)
    assert len(objs) <= DIAMOND_CAP


def test_two_builds_same_seed_have_identical_samples(tmp_path):
    a, _ = _build(tmp_path / "a", seed=777)
    b, _ = _build(tmp_path / "b", seed=777)
    for split in ("lite", "main", "diamond"):
        am = json.loads((a / f"{split}.manifest.json").read_text())
        bm = json.loads((b / f"{split}.manifest.json").read_text())
        assert am["sample_ids_sha256"] == bm["sample_ids_sha256"]


def test_catastrophic_samples_have_stamped_expected_violations(tmp_path):
    # Task 4.3: every built sample whose world_config populates issuer_behavior carries a
    # non-empty stamped gold.expected_violations (derived applicability).
    _, priv = _build(tmp_path)
    lines = [ln for ln in (priv / "private.jsonl").read_text().splitlines() if ln.strip()]
    catastrophic = [s for s in (Sample.model_validate_json(ln) for ln in lines)
                    if s.world_config.issuer_behavior]
    assert catastrophic, "no catastrophic-applicable samples were built"
    for s in catastrophic:
        assert s.gold.expected_violations, f"{s.sample_id} has empty expected_violations"


def test_trap_fraction_within_tolerance_per_split_per_axis(tmp_path):
    pub, priv = _build(tmp_path)
    for split, base in (("lite", pub), ("main", pub), ("private", priv)):
        m = json.loads((base / f"{split}.manifest.json").read_text())
        for axis in Axis:
            cell = m["counts"][axis.value]
            trap = sum(d["trap"] for d in cell.values())
            non_trap = sum(d["non_trap"] for d in cell.values())
            total = trap + non_trap
            if total == 0:
                continue
            frac = trap / total
            assert abs(frac - TARGET_TRAP_FRACTION) <= 0.05, f"{split}/{axis}: {frac}"
