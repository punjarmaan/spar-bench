"""EM4 — offline: README carries the Leaderboard section, disclosures, and command sequence.

README is a prose artifact; we assert presence of load-bearing strings, not wording.
"""

from __future__ import annotations

from pathlib import Path

README = (Path(__file__).resolve().parents[2] / "README.md").read_text(encoding="utf-8")


def test_readme_has_leaderboard_section() -> None:
    assert "## Leaderboard" in README
    assert "LEADERBOARD.md" in README


def test_readme_repeats_canary_and_do_not_train() -> None:
    assert "do not train" in README.lower()
    assert "canary" in README.lower()


def test_readme_discloses_per_stage_sampling() -> None:
    # competence temp 0.0 / reliability temp 0.7 must be disclosed near the leaderboard.
    lower = README.lower()
    assert "competence" in lower and "reliability" in lower
    assert "temperature" in lower or "temp=0" in lower or "temp 0" in lower


def test_readme_documents_command_sequence() -> None:
    assert "spar eval-cost" in README
    assert "spar eval" in README
    assert "spar leaderboard" in README
    # Ordering: cost estimate before the run before consolidation.
    i_cost = README.index("spar eval-cost")
    i_eval = README.index("spar eval ")  # trailing space avoids matching `spar eval-cost`
    i_board = README.index("spar leaderboard")
    assert i_cost < i_eval < i_board
