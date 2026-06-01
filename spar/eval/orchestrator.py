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
from spar.eval.cost import BudgetExceeded, CostMeter
from spar.eval.models import ModelConfig
from spar.eval.profile import Profile, StageSampling
from spar.harness.graders import ModelGrader, SampleScore, score
from spar.harness.passk import is_solved
from spar.harness.report import SPAR_VERSION, build_results
from spar.harness.runner import EpisodeTrace, run_episode
from spar.harness.user_sim import UserSim
from spar.harness.weights import DEFAULT_WEIGHTS
from spar.policies.loader import load_policy
from spar.simulator.contract import Abort, Action, Observation
from spar.simulator.enums import Axis
from spar.simulator.schemas import Sample

_T = TypeVar("_T")

PUBLISHABILITY_FLOOR = 0.98   # design §5.5: scored_fraction >= floor -> "verified", else "partial"


class SampleStatus(str, Enum):
    SCORED = "scored"                  # reached a terminal and graded normally; in denominator
    ERRORED_INFRA = "errored_infra"    # timeout/429/5xx after retries; excluded but counted
    MALFORMED_ACTION = "malformed_action"  # no valid JSON after reformat retry; scored as Abort
    REFUSED = "refused"                # model refused in-band; scored as its Abort/escalate


# Conservative untyped-fallback phrases: UNAMBIGUOUS phrases only; bare HTTP status codes
# (e.g. "500", "429") are intentionally excluded — they false-match capability errors whose
# messages happen to mention a number.  Real infra failures come through as typed litellm
# exceptions or carry a `status_code` attribute (Task 4.3).
# Context-overflow strings are intentionally excluded: a context-overflow is a deterministic
# capability failure — retrying just re-pays with no different outcome (Task 4.2).
_INFRA_SIGNATURES = (
    "rate limit", "rate_limit", "too many requests",
    "service unavailable", "bad gateway",
    "timeout", "timed out",
)


def _is_infra_error(exc: BaseException) -> bool:
    """True for timeout / 429 / 5xx (retryable infra failures, never a capability signal).

    Priority order (Task 4.3):
    1. BudgetExceeded → False (must halt, never retry; Task 4.1).
    2. TimeoutError   → True.
    3. Typed litellm exceptions (lazy import) → True for rate-limit / service-unavailable /
       timeout / API-connection / internal-server classes, or any exception with a numeric
       status_code of 429 or 5xx.
    4. Conservative untyped fallback: UNAMBIGUOUS phrases only (no bare numeric substrings).

    Context-overflow is also NOT an infra error — it is a deterministic capability outcome
    that must not be retried (Task 4.2)."""
    if isinstance(exc, BudgetExceeded):
        return False
    if isinstance(exc, TimeoutError):
        return True
    # --- typed litellm exception check (lazy import — litellm is never imported at module load) ---
    try:
        import litellm  # noqa: PLC0415
        if isinstance(exc, (
            litellm.RateLimitError,
            litellm.ServiceUnavailableError,
            litellm.Timeout,
            litellm.APIConnectionError,
            litellm.InternalServerError,
        )):
            return True
    except ImportError:
        pass  # litellm not installed — fall through to the untyped fallback
    # status_code attribute check (covers litellm subclasses and other HTTP-aware exceptions)
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and (status == 429 or 500 <= status <= 599):
        return True
    # --- conservative untyped substring fallback (no bare numeric strings) ---
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
    """Replays a cached completion in the litellm response shape the agent reads.

    `cache_hit` tells the agent whether this completion was a replay (already paid for on a prior
    run → counts toward gross cost but not billed) or a fresh paid call (`cache_hit=False`)."""

    def __init__(self, payload: dict[str, Any], *, cache_hit: bool) -> None:
        self.choices = [_CachedChoice(payload["content"])]
        self.usage = type("U", (), {
            "prompt_tokens": payload.get("prompt_tokens", 0),
            "completion_tokens": payload.get("completion_tokens", 0),
        })()
        self._hidden_params = {"response_cost": payload.get("response_cost")}
        self.cache_hit = cache_hit


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
    meter: CostMeter | None = None,
) -> Callable[..., _CachedResponse]:
    """Wrap a completion_fn with the content-addressed cache + infra retries. A cache hit replays
    the stored response and makes NO underlying call (resume idempotency, design §5.5/§5.6).
    `trial_index` partitions the cache so each pass^k trial samples & resumes independently (B1).
    `meter` enforces the per-completion budget cap: raises BudgetExceeded on a cache MISS when
    over budget so no paid call is made. Cache HITS are always free and never raise."""

    def fn(*, model: str, messages: list[dict[str, Any]], **sampling: Any) -> _CachedResponse:
        key = cache_key(model, messages, sampling, trial_index=trial_index)
        cached = cache.get(key)
        if cached is not None:
            # Cache hit: replay stored response, zero paid calls — never raise BudgetExceeded.
            # Marked cache_hit=True so the agent records it as gross-only (not billed this run).
            return _CachedResponse(cached, cache_hit=True)
        # Cache miss: about to make a paid call — check budget BEFORE proceeding.
        if meter is not None and meter.over_budget():
            raise BudgetExceeded(f"budget exhausted (spent ${meter.spent():.4f})")
        resp = _with_retries(
            lambda: raw_fn(model=model, messages=messages, **sampling),
            retries=retries, sleep=sleep,
        )
        payload = _extract(resp)
        cache.put(key, payload)
        return _CachedResponse(payload, cache_hit=False)

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


class _OverflowAbortAgent:
    """An immediate-abort agent used to SCORE a non-infra failure (e.g. context-overflow) as a
    capability/abort outcome (audit S1). It NEVER calls the model — `act` returns an Abort on the
    first turn — so the abort-scoring run adds no spend. The Abort reason is recorded in the audit
    trajectory; it is graded exactly like the malformed-action path (the agent produced no valid
    progress this turn) and scores 0 with catastrophic_class None (no forbidden completion)."""

    def act(self, observation: Observation) -> Action:
        return Abort(tool="abort", reason="context_overflow")


def _abort_scored(
    sample: Sample,
    *,
    responder: UserSim,
    grader: ModelGrader,
    k: int,
) -> SampleScore:
    """Grade `sample` AS IF the agent immediately aborted (audit S1): a non-infra error escaping
    run_episode (context-overflow) means the agent could not produce a valid action — semantically
    the malformed-action path, which already scores as an Abort. Reuses run_episode + score (no new
    model calls — the abort agent never invokes completion_fn) and marks it solved 0 of k."""
    trace = run_episode(sample, _OverflowAbortAgent(), trial_index=0, user_sim=responder)
    final = score(sample, trace, model_grader=grader)
    final.trials_n = k
    final.trials_c = 0  # solved 0 of k -> a capability failure that counts in the denominator
    return final


def _overhead_usd(responder: UserSim, grader: ModelGrader) -> float:
    """Cumulative responder + grader spend so far. Live LiteLLM variants track a `cost_usd`;
    offline stubs (ScriptedUserSim / StubModelGrader) have none, so they contribute 0.0."""
    return float(getattr(responder, "cost_usd", 0.0)) + float(getattr(grader, "cost_usd", 0.0))


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
    score, excluded from the denominator but counted — they are OURS). Non-infra, non-budget
    errors (e.g. context-overflow) are deterministic capability failures: NOT retried (Task 4.2)
    and SCORED as an abort-equivalent capability failure that LANDS IN the scored population
    (status MALFORMED_ACTION, trials_c=0, audit S1) — never silently excluded. The run continues.
    Otherwise grade + classify the status and accrue confirmed agent cost."""
    binary = sample.axis is not Axis.ROUTING
    policy_text = load_policy(sample.policy_id)
    solved = 0
    last_trace = None
    last_agent = None
    for trial_index in range(k):
        tsamp = _trial_sampling(sampling, trial_index)
        cached_fn = _cached_completion_fn(
            completion_fn, cache, retries=retries, sleep=sleep, trial_index=trial_index,
            meter=meter,
        )
        factory = agent_factory(
            model, policy_text=policy_text, sampling=tsamp,
            completion_fn=cached_fn, mandate_text=sample.mandate.goal,
        )
        agent = factory()
        try:
            trace = run_episode(sample, agent, trial_index=trial_index, user_sim=responder)
        except BudgetExceeded:
            # Record any usage that accrued before the cap was hit, then propagate to evaluate_model
            # so it can record budget_hit=True and stop the split loop cleanly.
            for u in getattr(agent, "usage", []):
                meter.record(u, model)
            raise
        except BaseException as exc:  # noqa: BLE001
            # Cost of any calls that DID return before the error is still real.
            for u in getattr(agent, "usage", []):
                meter.record(u, model)
            if _is_infra_error(exc):
                # Infra is OURS (timeout/429/5xx, already retried): excluded but counted.
                return SampleStatus.ERRORED_INFRA, None
            # Non-infra, non-budget errors (e.g. context-overflow) are deterministic capability
            # failures — NOT retried (Task 4.2) and must not crash the run. Per audit S1 they are
            # SCORED as a capability/abort outcome that LANDS IN the scored population (a counted
            # failure), not silently excluded: grade an immediate abort (no new model call) and
            # return MALFORMED_ACTION (a SCORED-population status), trials_c=0, catastrophic None.
            return (
                SampleStatus.MALFORMED_ACTION,
                _abort_scored(sample, responder=responder, grader=grader, k=k),
            )
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
    model_dir.mkdir(parents=True, exist_ok=True)
    meter = CostMeter(budget_usd=budget_usd)
    budget_hit = False
    canary: str | None = None
    splits_block: dict[str, dict[str, Any]] = {}
    # Disambiguate output keys when a split appears in the plan more than once (e.g. lite run at
    # both competence k=1 and reliability k=4): a bare `<split>.results.json` would have the later
    # stage silently overwrite the earlier. A unique split keeps `<split>` (consolidate.py reads
    # `main.results.json`/`diamond.results.json`); a repeated split becomes `<split>.<stage>`.
    _split_counts = Counter(p.split for p in profile.plan)

    for plan in profile.plan:
        result_key = (
            plan.split if _split_counts[plan.split] == 1 else f"{plan.split}.{plan.stage}"
        )
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
            try:
                status, sscore = _run_sample(
                    sample, model,
                    sampling=sampling, completion_fn=completion_fn,
                    cache=cache, meter=meter, responder=responder, grader=grader,
                    k=plan.k, retries=retries, sleep=sleep,
                )
            except BudgetExceeded:
                budget_hit = True
                break
            tally[status.value] += 1
            if sscore is not None:
                scores.append(sscore)
                # Replay the classified (last) trial from cache (no new model call) so the audit
                # log matches the reported status (N1).
                last_idx = plan.k - 1
                tsamp = _trial_sampling(sampling, last_idx)
                cached_fn = _cached_completion_fn(
                    completion_fn, cache, retries=retries, sleep=sleep, trial_index=last_idx,
                    meter=meter,
                )
                factory = agent_factory(
                    model, policy_text=load_policy(sample.policy_id), sampling=tsamp,
                    completion_fn=cached_fn, mandate_text=sample.mandate.goal,
                )
                try:
                    trace = run_episode(
                        sample, factory(), trial_index=last_idx, user_sim=responder
                    )
                except BaseException as exc:  # noqa: BLE001
                    # A non-infra failure (e.g. context-overflow) never cached a trial, so the
                    # replay re-raises. The sample was abort-SCORED (audit S1); re-derive the
                    # matching audit trajectory from the same immediate-abort agent (no model
                    # call, no spend). Infra errors are excluded upstream and never reach here.
                    if _is_infra_error(exc):
                        raise
                    trace = run_episode(
                        sample, _OverflowAbortAgent(), trial_index=last_idx, user_sim=responder
                    )
                _write_trajectory(
                    model_dir / "trajectories" / f"{sample.sample_id}.jsonl",
                    sample, trace, status,
                )
            # Sync responder/grader (overhead) spend so the cap counts total real money out the
            # door, not just the agent. Live LiteLLM responder/grader track a cumulative cost_usd;
            # offline stubs have none (-> 0.0). Sequential loop, so an absolute set is exact.
            meter.set_overhead(_overhead_usd(responder, grader))
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
        (model_dir / f"{result_key}.results.json").write_text(
            json.dumps(results, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )

        splits_block[result_key] = {
            "status": status_label,
            "scored_fraction": scored_fraction,
            "n": attempted,
            "n_scored": scored_count,
            "tally": dict(tally),
            "stage": plan.stage,
            "k": plan.k,
            "published": plan.published,
        }

    meter.set_overhead(_overhead_usd(responder, grader))  # final sync for an exact reported total
    manifest = {
        "model": model.id,
        "class": model.cls,
        "route": model.route,
        "spar_version": SPAR_VERSION,
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
        "cost_usd": meter.spent(),               # agent money ACTUALLY spent this run (cache misses)
        "gross_cost_usd": meter.gross(),         # what the run would cost uncached (hits + misses)
        "overhead_cost_usd": meter.overhead(),   # judge/user-sim (responder/grader) money spent
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
