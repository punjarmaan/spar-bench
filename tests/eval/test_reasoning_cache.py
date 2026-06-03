"""Reasoning content round-trips through the completion cache (enrichment T1)."""
from pathlib import Path

from spar.eval.cache import CompletionCache
from spar.eval.orchestrator import _CachedResponse, _cached_completion_fn, _extract


class _Msg:
    def __init__(self, content, reasoning):
        self.content = content
        self.reasoning_content = reasoning


class _Resp:
    def __init__(self, content, reasoning):
        self.choices = [type("C", (), {"message": _Msg(content, reasoning)})()]
        self.usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 3})()
        self._hidden_params = {"response_cost": 0.001}


def test_extract_captures_reasoning_content():
    payload = _extract(_Resp('{"tool":"abort","args":{}}', "let me think..."))
    assert payload["reasoning_content"] == "let me think..."
    assert payload["content"] == '{"tool":"abort","args":{}}'


def test_cached_response_surfaces_reasoning_on_replay():
    cr = _CachedResponse(
        {"content": "x", "prompt_tokens": 1, "completion_tokens": 1,
         "response_cost": 0.0, "reasoning_content": "cached thoughts"},
        cache_hit=True,
    )
    assert cr.choices[0].message.reasoning_content == "cached thoughts"


def test_old_cache_entry_without_reasoning_yields_none():
    cr = _CachedResponse(
        {"content": "x", "prompt_tokens": 1, "completion_tokens": 1, "response_cost": 0.0},
        cache_hit=True,
    )
    assert cr.choices[0].message.reasoning_content is None


def test_reasoning_round_trips_through_disk_cache(tmp_path: Path):
    cache = CompletionCache(tmp_path)
    calls = []

    def raw(**kw):
        calls.append(kw)
        return _Resp('{"tool":"capture","args":{}}', "step-by-step reasoning")

    fn = _cached_completion_fn(raw, cache, retries=0, sleep=lambda _s: None)
    fn(model="m", messages=[{"role": "user", "content": "hi"}], temperature=0.0)
    resp = fn(model="m", messages=[{"role": "user", "content": "hi"}], temperature=0.0)
    assert len(calls) == 1
    assert resp.cache_hit is True
    assert resp.choices[0].message.reasoning_content == "step-by-step reasoning"
