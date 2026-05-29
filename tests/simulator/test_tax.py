from decimal import Decimal

from spar.simulator.tax import (
    TaxSpec, FxSpec, DutiesSpec, compute_tax, is_prohibited_combo,
)


def test_us_sales_tax_on_total():
    spec = TaxSpec(regime="us_sales", rate_bps=875)  # 8.75%
    out = compute_tax(subtotal=Decimal("100.00"), tax=spec, fx=None, duties=None)
    assert out.computed_tax == Decimal("8.75")
    assert out.total == Decimal("108.75")


def test_eu_vat_oss_plus_fx_markup():
    spec = TaxSpec(regime="vat_oss", rate_bps=2000)  # 20%
    fx = FxSpec(quote_ccy="EUR", settle_ccy="USD", reference_rate=Decimal("1.08"),
                markup_bps=200)  # 2% over ECB reference
    out = compute_tax(subtotal=Decimal("100.00"), tax=spec, fx=fx, duties=None)
    assert out.computed_tax == Decimal("20.00")
    assert out.fx_markup == Decimal("2.40")  # 2% of 120.00
    assert out.total == Decimal("122.40")


def test_duties_apply_only_above_de_minimis():
    spec = TaxSpec(regime="ioss", rate_bps=2000)
    under = DutiesSpec(applies=True, rate_bps=1200, de_minimis_value=Decimal("150"))
    out_under = compute_tax(subtotal=Decimal("100"), tax=spec, fx=None, duties=under)
    assert out_under.duties == Decimal("0")  # below the de-minimis threshold
    out_over = compute_tax(subtotal=Decimal("200"), tax=spec, fx=None, duties=under)
    assert out_over.duties == Decimal("24.00")  # 12% of 200, over de-minimis


def test_prohibited_combo_lookup_is_data_driven():
    combos = [{"geo": "IR", "method": "visa"}, {"geo": "KP", "method": "mc"}]
    assert is_prohibited_combo("IR", "visa", combos) is True
    assert is_prohibited_combo("US", "visa", combos) is False
