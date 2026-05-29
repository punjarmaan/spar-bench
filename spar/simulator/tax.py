"""Destination tax + FX markup + duties helpers for the compliance_tax axis.

Pure functions; all money is Decimal with explicit currency. No RNG, no I/O.
Rounding is half-up to 2 places on each computed component (US sales tax rounds on
the total; VAT-style regimes round per the summed line here — sufficient for v1 gold
with the rounding-tolerant grader in module 40 §3.2).
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict

_CENTS = Decimal("0.01")
TaxRegime = Literal["vat_oss", "ioss", "us_sales", "gst", "none"]


def _q(value: Decimal) -> Decimal:
    return value.quantize(_CENTS, rounding=ROUND_HALF_UP)


class _Strict(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class TaxSpec(_Strict):
    regime: TaxRegime
    rate_bps: int  # destination rate in basis points (e.g. 2000 = 20%)


class FxSpec(_Strict):
    quote_ccy: str
    settle_ccy: str
    reference_rate: Decimal  # ECB/interbank reference
    markup_bps: int          # markup over the reference applied to the converted base


class DutiesSpec(_Strict):
    applies: bool
    rate_bps: int
    de_minimis_value: Decimal  # duties fire only on subtotal strictly above this


class TaxResult(_Strict):
    computed_tax: Decimal
    fx_markup: Decimal
    duties: Decimal
    total: Decimal


def compute_tax(
    *,
    subtotal: Decimal,
    tax: TaxSpec,
    fx: FxSpec | None,
    duties: DutiesSpec | None,
) -> TaxResult:
    """Landed cost = subtotal + destination tax + duties + FX markup (all Decimal)."""
    computed_tax = _q(subtotal * Decimal(tax.rate_bps) / Decimal(10000))
    duties_amt = Decimal("0")
    if duties is not None and duties.applies and subtotal > duties.de_minimis_value:
        duties_amt = _q(subtotal * Decimal(duties.rate_bps) / Decimal(10000))
    base_before_fx = subtotal + computed_tax + duties_amt
    fx_markup = Decimal("0")
    if fx is not None:
        fx_markup = _q(base_before_fx * Decimal(fx.markup_bps) / Decimal(10000))
    total = _q(base_before_fx + fx_markup)
    return TaxResult(
        computed_tax=computed_tax, fx_markup=fx_markup, duties=duties_amt, total=total,
    )


def is_prohibited_combo(geo: str, method: str, combos: list[dict[str, str]]) -> bool:
    """Data-driven (geo, method) legality check — the prohibited table is data, not code."""
    return any(c.get("geo") == geo and c.get("method") == method for c in combos)
