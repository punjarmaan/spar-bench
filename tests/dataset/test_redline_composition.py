from __future__ import annotations

from collections import Counter

from spar.dataset.build import assemble_redline
from spar.simulator.enums import Axis


def test_redline_is_seven_axis_trap_heavy_target_size():
    redline = assemble_redline(build_seed=1)
    assert all(s.is_trap for s in redline)
    per = Counter(s.axis for s in redline)
    for a in Axis:
        assert per[a.value] >= 8, f"{a.value}: {per[a.value]} (<8)"
    assert 55 <= len(redline) <= 75   # ~63 target


def test_redline_assembly_deterministic():
    a = sorted(s.sample_id for s in assemble_redline(build_seed=1))
    b = sorted(s.sample_id for s in assemble_redline(build_seed=1))
    assert a == b
