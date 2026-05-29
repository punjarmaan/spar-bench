"""EM2 Task 2: content-addressed completion cache (resume + reproducibility)."""

from __future__ import annotations

from spar.eval.cache import CompletionCache, cache_key


def test_cache_key_is_stable_and_input_sensitive() -> None:
    msgs = [{"role": "user", "content": "hi"}]
    samp = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 2048}
    k1 = cache_key("m/route", msgs, samp)
    k2 = cache_key("m/route", msgs, samp)
    assert k1 == k2
    assert isinstance(k1, str) and len(k1) >= 8
    # any field change changes the key
    assert cache_key("other/route", msgs, samp) != k1
    assert cache_key("m/route", [{"role": "user", "content": "bye"}], samp) != k1
    assert cache_key("m/route", msgs, {**samp, "temperature": 0.7}) != k1


def test_cache_roundtrip_and_miss(tmp_path) -> None:
    cache = CompletionCache(tmp_path)
    key = cache_key("m/route", [{"role": "user", "content": "x"}], {"temperature": 0.0})
    assert cache.get(key) is None
    cache.put(key, {"choices": [{"message": {"content": "ok"}}], "_response_cost": 0.01})
    again = CompletionCache(tmp_path)  # fresh instance reads from disk (survives restart)
    got = again.get(key)
    assert got is not None
    assert got["choices"][0]["message"]["content"] == "ok"
    assert got["_response_cost"] == 0.01


def test_cache_digest_changes_with_contents(tmp_path) -> None:
    cache = CompletionCache(tmp_path)
    d0 = cache.digest()
    cache.put(cache_key("m", [{"role": "user", "content": "a"}], {}), {"ok": True})
    d1 = cache.digest()
    assert d0 != d1
    assert isinstance(d1, str) and len(d1) >= 8
