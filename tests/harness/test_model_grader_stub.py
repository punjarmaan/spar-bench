from decimal import Decimal

from spar.harness.graders import ModelGrader, SemanticGradeInput
from spar.harness.model_grader import StubModelGrader


def _gi(spend: str) -> SemanticGradeInput:
    return SemanticGradeInput(
        category="coffee_maker",
        p25=Decimal("40"), p50=Decimal("42"), p75=Decimal("50"), p90=Decimal("58"),
        band_lo=Decimal("0"), band_hi=Decimal("40"), overspend_ceiling=Decimal("60"),
        spend=Decimal(spend), hint="not too expensive",
    )


def test_stub_grader_is_a_modelgrader_callable():
    grader: ModelGrader = StubModelGrader()
    assert callable(grader)
    score = grader(_gi("45"))
    assert isinstance(score, float) and 0.0 <= score <= 1.0


def test_stub_grader_passes_low_gray_zone_spend():
    assert StubModelGrader()(_gi("45")) >= 0.5


def test_stub_grader_fails_high_gray_zone_spend():
    assert StubModelGrader()(_gi("59")) < 0.5


def test_stub_grader_is_deterministic():
    g = StubModelGrader()
    assert g(_gi("50")) == g(_gi("50"))


def test_stub_grader_exposes_pinned_grader_model_id():
    assert StubModelGrader().grader_model == "stub-model-grader@1"
