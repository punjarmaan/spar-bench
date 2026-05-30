"""Batch orchestrator (design §5.5): drive split x k with the typed failure taxonomy,
infra retries (exp backoff + jitter), cache/resume, confirmed cost + hard cap, per-episode
audit trajectories, and a reproducibility run_manifest. Writes runs/<id>/<split>.results.json
via the unchanged build_results. litellm is never imported at module load (lazy in the agent)."""

from __future__ import annotations

import json
import random
import time
from collections import Counter
from collections.abc import Callable
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

from spar.eval.agent import SCAFFOLD_VERSION, agent_factory
from spar.eval.cache import CompletionCache, cache_key
from spar.eval.cost import CostMeter
from spar.eval.models import ModelConfig
from spar.eval.profile import Profile, StageSampling
from spar.harness.graders import ModelGrader, SampleScore, score
from spar.harness.passk import is_solved
from spar.harness.report import build_results
from spar.harness.runner import EpisodeTrace, run_episode
from spar.harness.user_sim import UserSim
from spar.harness.weights import DEFAULT_WEIGHTS
from spar.policies.loader import load_policy
from spar.simulator.contract import Abort
from spar.simulator.enums import Axis
from spar.simulator.schemas import Sample

_T = TypeVar("_T")

PUBLISHABILITY_FLOOR = 0.98   # design §5.5: scored_fraction >= floor -> "verified", else "partial"


class SampleStatus(str, Enum):
    SCORED = "scored"                  # reached a terminal and graded normally; in denominator
    ERRORED_INFRA = "errored_infra"    # timeout/429/5xx/context-overflow after retries; excluded
    MALFORMED_ACTION = "malformed_action"  # no valid JSON after reformat retry; scored as Abort
    REFUSED = "refused"                # model refused in-band; scored as its Abort/escalate


# Substrings that mark a failure as OURS (infra), not the model's behaviour (design §5.5/§8).
_INFRA_SIGNATURES = (
    "429", "rate limit", "rate_limit", "too many requests",
    "500", "502", "503", "504", "service unavailable", "bad gateway",
    "timeout", "timed out",
    "context_length_exceeded", "context length", "maximum context", "context-overflow",
)


def _is_infra_error(exc: BaseException) -> bool:
    """True for timeout / 429 / 5xx / context-overflow (retryable, never a capability fail)."""
    if isinstance(exc, TimeoutError):
        return True
    text = f"{type(exc).__name__} {exc}".lower()
    return any(sig in text for sig in _INFRA_SIGNATURES)


def _with_retries(
    call: Callable[[], _T],
    *,
    retries: int,
    sleep: Callable[[float], None],
    jitter: Callable[[], float] = random.random,
    base_delay: float = 0.5,
) -> _T:
    """Run `call`, retrying ONLY infra errors with exponential backoff + jitter up to `retries`
    extra attempts. `sleep`/`jitter` are injected so tests run instantly. Non-infra exceptions
    propagate on the first occurrence. Re-raises the last infra error once retries are exhausted."""
    attempt = 0
    while True:
        try:
            return call()
        except BaseException as exc:  # noqa: BLE001 - re-raised below unless retryable
            if not _is_infra_error(exc) or attempt >= retries:
                raise
            delay = base_delay * (2 ** attempt) + jitter()
            sleep(delay)
            attempt += 1


class _CachedMsg:
    def __init__(self, content: str) -> None:
        self.content = content


class _CachedChoice:
    def __init__(self, content: str) -> None:
        self.message = _CachedMsg(content)


class _CachedResponse:
    """Replays a cached completion in the litellm response shape the agent reads."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.choices = [_CachedChoice(payload["content"])]
        self.usage = type("U", (), {
            "prompt_tokens": payload.get("prompt_tokens", 0),
            "completion_tokens": payload.get("completion_tokens", 0),
        })()
        self._hidden_params = {"response_cost": payload.get("response_cost")}


def _extract(resp: Any) -> dict[str, Any]:
    """Pull the cache-storable fields out of any litellm-shaped response object."""
    usage = getattr(resp, "usage", None)
    hidden = getattr(resp, "_hidden_params", {}) or {}
    return {
        "content": resp.choices[0].message.content,
        "prompt_tokens": getattr(usage, "prompt_tokens", 0),
        "completion_tokens": getattr(usage, "completion_tokens", 0),
        "response_cost": hidden.get("response_cost"),
    }


def _cached_completion_fn(
    raw_fn: Callable[..., Any],
    cache: CompletionCache,
    *,
    retries: int,
    sleep: Callable[[float], None],
    trial_index: int = 0,
) -> Callable[..., _CachedResponse]:
    """Wrap a completion_fn with the content-addressed cache + infra retries. A cache hit replays
    the stored response and makes NO underlying call (resume idempotency, design §5.5/§5.6).
    `trial_index` partitions the cache so each pass^k trial samples & resumes independently (B1)."""

    def fn(*, model: str, messages: list[dict[str, Any]], **sampling: Any) -> _CachedResponse:
        key = cache_key(model, messages, sampling, trial_index=trial_index)
        cached = cache.get(key)
        if cached is not None:
            return _CachedResponse(cached)
        resp = _with_retries(
            lambda: raw_fn(model=model, messages=messages, **sampling),
            retries=retries, sleep=sleep,
        )
        payload = _extract(resp)
        cache.put(key, payload)
        return _CachedResponse(payload)

    return fn


def _trial_sampling(sampling: StageSampling, trial_index: int) -> StageSampling:
    """Per-trial sampling for pass^k: offset the seed (when set) so trials diversify even on
    seed-honoring providers; temperature/top_p stay IDENTICAL across trials (fairness — design
    §5.3 pins one temperature per stage, diversity comes from sampling variation, not temp)."""
    if sampling.seed is None:
        return sampling
    return sampling.model_copy(update={"seed": sampling.seed + trial_index})


def _classify(score_obj: SampleScore, trace: EpisodeTrace) -> SampleStatus:
    """Tag the model-behaviour sub-case of a graded sample (design §5.5).

    A graded sample is SCORED unless the agent layer flagged a malformed action or an in-band
    refusal via the terminating Abort it emitted (EM1 sets reason="malformed_action")."""
    last = trace.action_log[-1] if trace.action_log else None
    if isinstance(last, Abort):
        if last.reason == "malformed_action":
            return SampleStatus.MALFORMED_ACTION
        if trace.made_decision and trace.abort_reason != "step_budget_exhausted":
            return SampleStatus.REFUSED
    return SampleStatus.SCORED


def _run_sample(
    sample: Sample,
    model: ModelConfig,
    *,
    sampling: StageSampling,
    completion_fn: Callable[..., Any],
    cache: CompletionCache,
    meter: CostMeter,
    responder: UserSim,
    grader: ModelGrader,
    k: int,
    retries: int,
    sleep: Callable[[float], None],
) -> tuple[SampleStatus, SampleScore | None]:
    """Run one sample's pass^k offline. Infra errors that survive retries -> ERRORED_INFRA (no
    score, excluded from the denominator but counted). Otherwise grade + classify the status and
    accrue confirmed agent cost."""
    binary = sample.axis is not Axis.ROUTING
    policy_text = load_policy(sample.policy_id)
    solved = 0
    last_trace = None
    last_agent = None
    for trial_index in range(k):
        tsamp = _trial_sampling(sampling, trial_index)
        cached_fn = _cached_completion_fn(
            completion_fn, cache, retries=retries, sleep=sleep, trial_index=trial_index
        )
        factory = agent_factory(
            model, policy_text=policy_text, sampling=tsamp,
            completion_fn=cached_fn, mandate_text=sample.mandate.goal,
        )
        agent = factory()
        try:
            trace = run_episode(sample, agent, trial_index=trial_index, user_sim=responder)
        except BaseException as exc:  # noqa: BLE001
            if _is_infra_error(exc):
                # Cost of any calls that DID return before the fatal infra error is still real.
                for u in getattr(agent, "usage", []):
                    meter.record(u, model)
                return SampleStatus.ERRORED_INFRA, None
            raise
        for u in getattr(agent, "usage", []):
            meter.record(u, model)
        if is_solved(score(sample, trace, model_grader=grader).score, binary=binary):
            solved += 1
        last_trace = trace
        last_agent = agent
    assert last_trace is not None and last_agent is not None
    final = score(sample, last_trace, model_grader=grader)
    final.trials_n = k
    final.trials_c = solved
    return _classify(final, last_trace), final


def _grader_id(grader: ModelGrader) -> str | None:
    return getattr(grader, "grader_model", None)


def _responder_id(responder: UserSim) -> str:
    return getattr(responder, "model", type(responder).__name__)


def _write_trajectory(
    path: Path, sample: Sample, trace: EpisodeTrace, status: SampleStatus
) -> None:
    """One JSONL episode transcript per sample (design §5.5 audit log)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "sample_id": sample.sample_id,
        "axis": sample.axis.value,
        "status": status.value,
        "final_state": trace.final_state.value if trace.final_state else None,
        "grade_terminal": trace.grade_terminal.value if trace.grade_terminal else None,
        "actions": [
            {"tool": a.tool, "args": a.model_dump(mode="json", exclude={"tool"})}
            for a in trace.action_log
        ],
        "abort_reason": trace.abort_reason,
        "terminating_action": trace.terminating_action,
    }
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def evaluate_model(
    model: ModelConfig,
    profile: Profile,
    *,
    out_dir: Path,
    cache: CompletionCache,
    budget_usd: float | None,
    concurrency: int,
    responder: UserSim,
    grader: ModelGrader,
    completion_fn: Callable[..., Any] | None = None,
    retries: int = 4,
    sleep: Callable[[float], None] = time.sleep,
    samples_for: Callable[[str], list[Sample]] | None = None,
) -> None:
    """Orchestrate one model over the profile's split x k plan (design §5.5): typed-status tally,
    confirmed cost + hard cap, per-episode audit trajectories, and a reproducibility manifest.
    Writes runs/<id>/<split>.results.json + run_manifest.json. litellm stays lazy (the agent)."""
    if samples_for is None:
        from spar.dataset.loader import load_split

        samples_for = load_split
    if completion_fn is None:
        import litellm

        completion_fn = litellm.completion

    model_dir = Path(out_dir) / model.id
    meter = CostMeter(budget_usd=budget_usd)
    budget_hit = False
    canary: str | None = None
    splits_block: dict[str, dict[str, Any]] = {}

    for plan in profile.plan:
        sampling = profile.competence if plan.stage == "competence" else profile.reliability
        samples = samples_for(plan.split)
        if samples:
            canary = samples[0].canary
        tally: Counter[str] = Counter()
        scores: list[SampleScore] = []
        attempted = 0
        for sample in samples:
            if budget_hit or meter.over_budget():
                break
            attempted += 1
            status, sscore = _run_sample(
                sample, model,
                sampling=sampling, completion_fn=completion_fn,
                cache=cache, meter=meter, responder=responder, grader=grader,
                k=plan.k, retries=retries, sleep=sleep,
            )
            tally[status.value] += 1
            if sscore is not None:
                scores.append(sscore)
                # Replay the classified (last) trial from cache (no new model call) so the audit
                # log matches the reported status (N1).
                last_idx = plan.k - 1
                tsamp = _trial_sampling(sampling, last_idx)
                cached_fn = _cached_completion_fn(
                    completion_fn, cache, retries=retries, sleep=sleep, trial_index=last_idx
                )
                factory = agent_factory(
                    model, policy_text=load_policy(sample.policy_id), sampling=tsamp,
                    completion_fn=cached_fn, mandate_text=sample.mandate.goal,
                )
                trace = run_episode(sample, factory(), trial_index=last_idx, user_sim=responder)
                _write_trajectory(
                    model_dir / "trajectories" / f"{sample.sample_id}.jsonl",
                    sample, trace, status,
                )
            if meter.over_budget():
                budget_hit = True
                break

        scored_count = len(scores)
        scored_fraction = scored_count / attempted if attempted else 0.0
        status_label = (
            "verified"
            if (scored_fraction >= PUBLISHABILITY_FLOOR and not budget_hit)
            else "partial"
        )

        results = build_results(
            scores, split=plan.split, canary=canary or "",
            build_seed=0, weights=DEFAULT_WEIGHTS.as_dict(),
        )
        results["scaffold_version"] = SCAFFOLD_VERSION
        results["status"] = status_label
        results["scored_fraction"] = scored_fraction
        results["status_tally"] = dict(tally)
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / f"{plan.split}.results.json").write_text(
            json.dumps(results, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )

        splits_block[plan.split] = {
            "status": status_label,
            "scored_fraction": scored_fraction,
            "n": attempted,
            "n_scored": scored_count,
            "tally": dict(tally),
            "stage": plan.stage,
            "k": plan.k,
            "published": plan.published,
        }

    manifest = {
        "model": model.id,
        "class": model.cls,
        "route": model.route,
        "spar_version": results["spar_version"],
        "model_version_pin": model.version_pin,
        "scaffold_version": SCAFFOLD_VERSION,
        "canary": canary,
        "build_seed": 0,
        "weights": DEFAULT_WEIGHTS.as_dict(),
        "provenance": "public_self_run",
        "grader_model": _grader_id(grader),
        "responder_model": _responder_id(responder),
        "sampling": {
            "competence": profile.competence.model_dump(),
            "reliability": profile.reliability.model_dump(),
        },
        "cost_usd": meter.spent(),
        "run_date": date.today().isoformat(),
        "cache_digest": cache.digest(),
        "concurrency": concurrency,
        "budget_usd": budget_usd,
        "budget_hit": budget_hit,
        "splits": splits_block,
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )


def evaluate_all(
    models: list[ModelConfig],
    profile: Profile,
    *,
    out_dir: Path,
    cache: CompletionCache,
    budget_usd: float | None = None,
    concurrency: int = 1,
    responder: UserSim,
    grader: ModelGrader,
    only: str | None = None,
    completion_fn: Callable[..., Any] | None = None,
    retries: int = 4,
    sleep: Callable[[float], None] = time.sleep,
    samples_for: Callable[[str], list[Sample]] | None = None,
) -> None:
    """Run a roster of models; `only` filters to a single model id. The budget cap is per-model
    (design §5.6) — each model gets a fresh CostMeter inside evaluate_model."""
    for model in models:
        if only is not None and model.id != only:
            continue
        evaluate_model(
            model, profile,
            out_dir=out_dir, cache=cache, budget_usd=budget_usd, concurrency=concurrency,
            responder=responder, grader=grader, completion_fn=completion_fn,
            retries=retries, sleep=sleep, samples_for=samples_for,
        )
