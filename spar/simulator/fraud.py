"""Agent-reactive fraud engine.

`fraud_score(t)` is a monotone, seeded function of accumulated agent behavior:

    fraud_score(t) = clip( base
        + sensitivity * ( a * txn_frequency(t)
                        + b * n_distinct_merchants(t)
                        + c * velocity(t) )
        + noise_seeded , 0.0, 1.0 )

The only randomness is `noise`, drawn from the FRAUD sub-stream. `record_submission` raises
the behavioral terms; `record_wait` (modeling `retry(wait)`) decays the frequency/velocity
terms — the only action that lets the score cool off. Money/velocity are Decimal; the
returned score is a float in [0, 1].
"""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum

from spar.simulator.rng import SubStream, substream

_A_FREQUENCY = 0.15  # per submission/step over baseline
_B_MERCHANTS = 0.10  # per distinct merchant beyond the first
_C_VELOCITY = 0.05  # per normalized velocity unit
_BASE_NO_ATTESTATION = 0.05
_BASE_ATTESTATION = 0.01
_DECAY = 0.5  # fraction the frequency/velocity counters retain per `wait`
_VELOCITY_NORM = Decimal("100")  # one velocity unit per $100 of recent amount/step


class FraudEffect(StrEnum):
    NONE = "none"
    CHALLENGE = "challenge"
    SOFT_BLOCK = "soft_block"
    HARD_BLOCK = "hard_block"


class FraudEngine:
    """Owns the running behavioral counters and computes the hidden fraud score."""

    def __init__(
        self,
        *,
        sample_id: str,
        seed: int,
        trial_index: int,
        sensitivity: float,
        challenge_at: float,
        soft_block_at: float,
        hard_block_at: float,
        attestation_present: bool,
    ) -> None:
        self.sample_id = sample_id
        self.seed = seed
        self.trial_index = trial_index
        self.sensitivity = sensitivity
        self.challenge_at = challenge_at
        self.soft_block_at = soft_block_at
        self.hard_block_at = hard_block_at
        self.base = _BASE_ATTESTATION if attestation_present else _BASE_NO_ATTESTATION
        self._frequency = 0.0
        self._velocity = 0.0
        self._merchants: set[str] = set()

    def record_submission(self, *, merchant: str, amount: Decimal) -> None:
        """A submission/retry that re-pushes authorization — raises every term."""
        self._frequency += 1.0
        self._velocity += float(amount / _VELOCITY_NORM)
        self._merchants.add(merchant)

    def record_wait(self) -> None:
        """`retry(wait)` cool-off: decays the frequency/velocity terms."""
        self._frequency *= _DECAY
        self._velocity *= _DECAY

    def _noise(self, step: int) -> float:
        rng = substream(
            self.sample_id, seed=self.seed, trial_index=self.trial_index,
            stream=SubStream.FRAUD, step=step,
        )
        return float(rng.normal(0.0, 0.02))

    def behavioral_score(self) -> float:
        """NOISE-FREE deterministic component: base + sensitivity * behavioral terms."""
        distinct_beyond_first = max(0, len(self._merchants) - 1)
        behavioral = (
            _A_FREQUENCY * self._frequency
            + _B_MERCHANTS * float(distinct_beyond_first)
            + _C_VELOCITY * self._velocity
        )
        return self.base + self.sensitivity * behavioral

    def fraud_score(self, *, step: int) -> float:
        raw = self.behavioral_score() + self._noise(step)
        return min(1.0, max(0.0, raw))

    def effect_for_score(self, score: float) -> FraudEffect:
        if score >= self.hard_block_at:
            return FraudEffect.HARD_BLOCK
        if score >= self.soft_block_at:
            return FraudEffect.SOFT_BLOCK
        if score >= self.challenge_at:
            return FraudEffect.CHALLENGE
        return FraudEffect.NONE
