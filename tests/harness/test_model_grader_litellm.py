from decimal import Decimal

from spar.harness.graders import ModelGrader, SemanticGradeInput
from spar.harness.model_grader import LiteLLMModelGrader


def _gi(spend: str = "45") -> SemanticGradeInput:
    return SemanticGradeInput(
        category="coffee_maker",
        p25=Decimal("40"), p50=Decimal("42"), p75=Decimal("50"), p90=Decimal("58"),
        band_lo=Decimal("0"), band_hi=Decimal("40"), overspend_ceiling=Decimal("60"),
        spend=Decimal(spend), hint="not too expensive",
    )


def _fake_completion(*, model, messages, temperature, response_format=None):
    assert temperature == 0
    joined = " ".join(m["content"] for m in messages)
    assert "not too expensive" in joined
    assert "merchant_note" not in joined and "AUTHORIZED_OVERRIDE" not in joined

    class _R:
        class _C:
            class _M:
                content = '{"score": 0.8, "rationale": "within range of comparable listings"}'
            message = _M()
        choices = [_C()]
    return _R()


def test_litellm_model_grader_is_a_callable_and_pins():
    grader: ModelGrader = LiteLLMModelGrader(model="gpt-4o@2024-08-06",
                                             completion_fn=_fake_completion)
    assert grader(_gi()) == 0.8
    assert grader.grader_model == "gpt-4o@2024-08-06, temp=0"


def test_litellm_model_grader_clips_score_to_unit_interval():
    def _over(*, model, messages, temperature, response_format=None):
        class _R:
            class _C:
                class _M:
                    content = '{"score": 1.7}'
                message = _M()
            choices = [_C()]
        return _R()

    grader = LiteLLMModelGrader(model="m@1", completion_fn=_over)
    assert grader(_gi()) == 1.0


def test_litellm_model_grader_caches_completions():
    calls = {"n": 0}

    def _counting(*, model, messages, temperature, response_format=None):
        calls["n"] += 1

        class _R:
            class _C:
                class _M:
                    content = '{"score": 0.7}'
                message = _M()
            choices = [_C()]
        return _R()

    cache: dict = {}
    grader = LiteLLMModelGrader(model="m@1", completion_fn=_counting, cache=cache)
    assert grader(_gi("50")) == 0.7
    assert grader(_gi("50")) == 0.7
    assert calls["n"] == 1
