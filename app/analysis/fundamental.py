"""Fundamental analysis from point-in-time financial facts.

Everything here operates on facts already resolved as-of an analysis timestamp
by ``app.pit.queries.financial_facts_as_of``, so a restated figure cannot leak
backwards. Three rules shape the implementations:

* **Flows and balances are not interchangeable.** Revenue is a duration fact,
  total assets an instant fact. A ratio mixing them (asset turnover) must take
  the flow over the period and the balance at a point, and the code states
  which.
* **A missing input gives a missing ratio.** Not a zero, and not a figure
  computed from a substitute concept, because "we could not compute margin"
  and "margin is 0%" are different statements.
* **Currency and period must match.** Facts in different currencies are not
  combined; a mismatch is reported rather than silently converted at some
  unstated rate.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

from app.analysis.registry import MethodInput, MethodSpec, register_method
from app.db.models.enums import InstrumentType

EQUITY_ONLY = (InstrumentType.EQUITY, InstrumentType.ADR, InstrumentType.REIT)


@dataclass(frozen=True, slots=True)
class PeriodFacts:
    """Facts for one fiscal period, already currency-checked."""

    label: str
    period_end: str
    currency: str | None
    values: dict[str, float]

    def get(self, concept: str) -> float | None:
        return self.values.get(concept)


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    """Divide, or return None. Never substitutes a default."""
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _growth(current: float | None, prior: float | None) -> float | None:
    """Period-over-period growth.

    Returns None when the prior figure is non-positive: growth from a loss to a
    smaller loss is not a meaningful percentage, and reporting one produces the
    nonsense figures that litter screeners.
    """
    if current is None or prior is None or prior <= 0:
        return None
    return (current / prior) - 1.0


@register_method(
    MethodSpec(
        key="profitability",
        version="1.0.0",
        family="fundamental",
        display_name="Profitability and margins",
        description=(
            "Gross, operating and net margins, return on equity and return on "
            "invested capital, from the facts available at the analysis timestamp."
        ),
        purpose="Describe how much of revenue reaches profit, and the return on capital employed.",
        required_inputs=("scalar:revenue",),
        min_observations=0,
        output_schema={
            "gross_margin": "fraction",
            "operating_margin": "fraction",
            "net_margin": "fraction",
            "return_on_equity": "fraction",
            "return_on_invested_capital": "fraction",
        },
        missing_data_policy=(
            "Each ratio is computed only when both of its inputs are present; "
            "otherwise it is null. No concept is substituted for another."
        ),
        limitations=(
            "Accounting definitions differ across issuers and jurisdictions, so "
            "margins are not strictly comparable between companies.",
            "ROE is inflated by leverage and is meaningless or misleading when "
            "equity is small or negative.",
            "ROIC here uses period-end balances, not an average, so it is "
            "distorted by a large acquisition or buyback during the period.",
            "A single period's margin says nothing about its durability.",
        ),
        supported_instrument_types=EQUITY_ONLY,
        supported_resolutions=("1d",),
    )
)
def _profitability(data: MethodInput, params: dict[str, Any]) -> dict[str, Any]:
    f = data.scalars
    revenue = f.get("revenue")
    gross = f.get("gross_profit")
    if gross is None and revenue is not None and f.get("cost_of_revenue") is not None:
        gross = revenue - f["cost_of_revenue"]

    equity = f.get("stockholders_equity")
    net_income = f.get("net_income")
    operating_income = f.get("operating_income")

    # Invested capital = equity + total debt - cash. Stated explicitly because
    # every data vendor defines it differently.
    debt = sum(
        v for k in ("long_term_debt", "short_term_debt") if (v := f.get(k)) is not None
    ) or None
    cash = f.get("cash_and_equivalents")
    invested_capital = None
    if equity is not None and debt is not None:
        invested_capital = equity + debt - (cash or 0.0)

    warnings: list[str] = []
    if equity is not None and equity <= 0:
        warnings.append(
            "stockholders' equity is zero or negative, so return on equity is not "
            "reported: the ratio is undefined rather than large"
        )

    return (
        {
            "gross_margin": _ratio(gross, revenue),
            "operating_margin": _ratio(operating_income, revenue),
            "net_margin": _ratio(net_income, revenue),
            "return_on_equity": (
                _ratio(net_income, equity) if (equity or 0) > 0 else None
            ),
            "return_on_invested_capital": (
                _ratio(operating_income, invested_capital)
                if (invested_capital or 0) > 0
                else None
            ),
            "revenue": revenue,
            "gross_profit": gross,
            "operating_income": operating_income,
            "net_income": net_income,
            "invested_capital": invested_capital,
            "invested_capital_definition": (
                "stockholders_equity + long_term_debt + short_term_debt - cash_and_equivalents"
            ),
        },
        warnings,
    )


@register_method(
    MethodSpec(
        key="balance_sheet_strength",
        version="1.0.0",
        family="fundamental",
        display_name="Balance-sheet strength",
        description=(
            "Leverage, liquidity and coverage measures from the balance sheet and "
            "income statement."
        ),
        purpose="Assess how much financial stress the issuer could absorb.",
        required_inputs=("scalar:total_assets",),
        min_observations=0,
        output_schema={
            "current_ratio": "ratio",
            "debt_to_equity": "ratio",
            "net_debt": "currency",
            "interest_coverage": "ratio",
            "net_debt_to_operating_cash_flow": "ratio",
        },
        missing_data_policy="Null where an input is absent; nothing is imputed.",
        limitations=(
            "Operating leases, pension obligations and off-balance-sheet "
            "commitments are not captured and can be material.",
            "A balance sheet is a snapshot at one date and can be managed around "
            "the reporting date.",
            "Comparable ratios differ sharply by industry: a bank's leverage is "
            "not a software company's.",
            "Interest coverage using reported interest expense understates the "
            "burden for issuers with capitalised interest.",
        ),
        supported_instrument_types=EQUITY_ONLY,
        supported_resolutions=("1d",),
    )
)
def _balance_sheet_strength(data: MethodInput, params: dict[str, Any]) -> dict[str, Any]:
    f = data.scalars
    equity = f.get("stockholders_equity")
    long_term = f.get("long_term_debt")
    short_term = f.get("short_term_debt")
    total_debt = None
    if long_term is not None or short_term is not None:
        total_debt = (long_term or 0.0) + (short_term or 0.0)
    cash = f.get("cash_and_equivalents")
    investments = f.get("short_term_investments")
    liquid = None
    if cash is not None:
        liquid = cash + (investments or 0.0)

    net_debt = (total_debt - liquid) if (total_debt is not None and liquid is not None) else None
    operating_income = f.get("operating_income")
    interest = f.get("interest_expense")
    ocf = f.get("operating_cash_flow")

    warnings: list[str] = []
    if long_term is None and short_term is None:
        warnings.append(
            "no debt concepts were reported; leverage measures are null rather "
            "than zero, since absence of a tag is not evidence of no debt"
        )

    return (
        {
            "current_ratio": _ratio(f.get("current_assets"), f.get("current_liabilities")),
            "debt_to_equity": (
                _ratio(total_debt, equity) if (equity or 0) > 0 else None
            ),
            "debt_to_assets": _ratio(total_debt, f.get("total_assets")),
            "total_debt": total_debt,
            "liquid_assets": liquid,
            "net_debt": net_debt,
            "interest_coverage": (
                _ratio(operating_income, abs(interest)) if interest else None
            ),
            "net_debt_to_operating_cash_flow": (
                _ratio(net_debt, ocf) if (ocf or 0) > 0 and net_debt is not None else None
            ),
            "equity_ratio": _ratio(equity, f.get("total_assets")),
        },
        warnings,
    )


@register_method(
    MethodSpec(
        key="cash_flow_quality",
        version="1.0.0",
        family="fundamental",
        display_name="Cash flow and earnings quality",
        description=(
            "Free cash flow, the ratio of operating cash flow to net income, and "
            "capital-allocation flows."
        ),
        purpose=(
            "Check whether reported profit converts into cash, which is the "
            "single most informative earnings-quality signal."
        ),
        required_inputs=("scalar:operating_cash_flow",),
        min_observations=0,
        output_schema={
            "free_cash_flow": "currency",
            "ocf_to_net_income": "ratio",
            "fcf_margin": "fraction",
            "capex_to_revenue": "fraction",
        },
        missing_data_policy="Null where an input is absent.",
        limitations=(
            "An OCF-to-net-income ratio near 1 is reassuring but not proof of "
            "clean accounting; it can be managed.",
            "Free cash flow here is operating cash flow less capital expenditure "
            "and excludes acquisitions, which can be the larger outflow.",
            "A single period is noisy: working-capital swings move OCF "
            "substantially without any change in business quality.",
        ),
        supported_instrument_types=EQUITY_ONLY,
        supported_resolutions=("1d",),
    )
)
def _cash_flow_quality(data: MethodInput, params: dict[str, Any]) -> dict[str, Any]:
    f = data.scalars
    ocf = f.get("operating_cash_flow")
    capex = f.get("capital_expenditure")
    revenue = f.get("revenue")
    net_income = f.get("net_income")
    # SEC reports capex as a positive payment amount; subtract its magnitude.
    fcf = (ocf - abs(capex)) if (ocf is not None and capex is not None) else None

    warnings: list[str] = []
    ratio = _ratio(ocf, net_income) if (net_income or 0) > 0 else None
    if ratio is not None and ratio < 0.7:
        warnings.append(
            f"operating cash flow is {ratio:.0%} of net income; a persistent "
            "shortfall is an earnings-quality question worth investigating, though "
            "a single period can be explained by working-capital timing"
        )

    return (
        {
            "operating_cash_flow": ocf,
            "capital_expenditure": abs(capex) if capex is not None else None,
            "free_cash_flow": fcf,
            "ocf_to_net_income": ratio,
            "fcf_margin": _ratio(fcf, revenue),
            "capex_to_revenue": (
                _ratio(abs(capex), revenue) if capex is not None else None
            ),
            "dividends_paid": f.get("dividends_paid"),
            "share_repurchase": f.get("share_repurchase"),
            "fcf_definition": "operating_cash_flow - abs(capital_expenditure)",
        },
        warnings,
    )


@register_method(
    MethodSpec(
        key="valuation_ratios",
        version="1.0.0",
        family="fundamental",
        display_name="Valuation ratios",
        description=(
            "Price and enterprise-value multiples computed from the market price "
            "at the analysis timestamp and the most recent facts available then."
        ),
        purpose="Express the price being paid per unit of earnings, sales or cash flow.",
        # Either share count works; the method prefers diluted and falls back.
        required_inputs=("close", "scalar:shares_diluted|scalar:shares_basic"),
        min_observations=1,
        output_schema={
            "price_to_earnings": "ratio",
            "price_to_sales": "ratio",
            "price_to_book": "ratio",
            "ev_to_ebit": "ratio",
            "free_cash_flow_yield": "fraction",
        },
        missing_data_policy=(
            "Requires a share count to form a market capitalisation. Without one "
            "no multiple is reported, since a per-share price alone cannot be "
            "compared to a company-level figure."
        ),
        limitations=(
            "A multiple is not a valuation. A low P/E can reflect a correctly "
            "anticipated earnings decline.",
            "Trailing multiples use already-published figures, so they lag any "
            "change in the business.",
            "P/E is undefined for loss-making companies and is not reported "
            "rather than being shown as a negative number.",
            "Multiples are only comparable within an industry, and even then "
            "capital intensity and accounting choices distort them.",
            "The share count used is the diluted weighted average for the "
            "reporting period, not the count outstanding today.",
        ),
        supported_instrument_types=EQUITY_ONLY,
        supported_resolutions=("1d",),
    )
)
def _valuation_ratios(data: MethodInput, params: dict[str, Any]) -> dict[str, Any]:
    f = data.scalars
    price = float(data.close[-1])
    # The registry guarantees one of these is present; a multiple cannot be
    # formed from a per-share price without a company-level share count.
    shares = f.get("shares_diluted") or f.get("shares_basic")
    warnings: list[str] = []
    if f.get("shares_diluted") is None:
        warnings.append(
            "no diluted share count was reported, so the basic count was used; "
            "multiples are therefore not dilution-adjusted"
        )

    market_cap = price * shares
    net_income = f.get("net_income")
    revenue = f.get("revenue")
    equity = f.get("stockholders_equity")
    operating_income = f.get("operating_income")
    ocf = f.get("operating_cash_flow")
    capex = f.get("capital_expenditure")
    fcf = (ocf - abs(capex)) if (ocf is not None and capex is not None) else None

    debt = None
    if f.get("long_term_debt") is not None or f.get("short_term_debt") is not None:
        debt = (f.get("long_term_debt") or 0.0) + (f.get("short_term_debt") or 0.0)
    cash = f.get("cash_and_equivalents")
    enterprise_value = (
        market_cap + (debt or 0.0) - (cash or 0.0) if debt is not None else None
    )

    if net_income is not None and net_income <= 0:
        warnings.append(
            "net income is zero or negative, so price-to-earnings is not reported: "
            "a negative multiple is not interpretable"
        )

    return (
        {
            "market_cap": market_cap,
            "enterprise_value": enterprise_value,
            "price": price,
            "shares_used": shares,
            "price_to_earnings": (
                _ratio(market_cap, net_income) if (net_income or 0) > 0 else None
            ),
            "price_to_sales": _ratio(market_cap, revenue),
            "price_to_book": _ratio(market_cap, equity) if (equity or 0) > 0 else None,
            "ev_to_ebit": (
                _ratio(enterprise_value, operating_income)
                if enterprise_value is not None and (operating_income or 0) > 0
                else None
            ),
            "free_cash_flow_yield": (
                _ratio(fcf, market_cap) if fcf is not None else None
            ),
            "earnings_yield": (
                _ratio(net_income, market_cap) if net_income is not None else None
            ),
        },
        warnings,
    )


@register_method(
    MethodSpec(
        key="growth_trajectory",
        version="1.0.0",
        family="fundamental",
        display_name="Growth trajectory",
        description=(
            "Period-over-period and multi-period growth in revenue, earnings and "
            "cash flow from a supplied sequence of fiscal periods."
        ),
        purpose="Describe the direction and consistency of the business, not just its level.",
        # Any one of the tracked fiscal series is enough to produce output.
        required_inputs=(
            "series:revenue_by_period|series:net_income_by_period"
            "|series:operating_cash_flow_by_period|series:eps_diluted_by_period",
        ),
        min_observations=0,
        output_schema={
            "revenue_growth_latest": "fraction",
            "revenue_cagr": "fraction_per_year",
            "growth_consistency": "fraction",
        },
        missing_data_policy=(
            "Growth is null where the prior period is missing or non-positive; a "
            "percentage change from a loss is not meaningful."
        ),
        limitations=(
            "Growth rates from a small base are arithmetically large and "
            "economically unremarkable.",
            "Acquisitions make reported growth non-organic, and the facts here "
            "cannot separate the two.",
            "A CAGR over a short span is dominated by the endpoints chosen.",
            "Consecutive periods must be of equal length; mixing a quarter into an "
            "annual series produces nonsense and is the caller's responsibility to "
            "avoid.",
        ),
        supported_instrument_types=EQUITY_ONLY,
        supported_resolutions=("1d",),
        params_schema={"defaults": {"periods_per_year": 1}},
    )
)
def _growth_trajectory(data: MethodInput, params: dict[str, Any]) -> dict[str, Any]:
    periods_per_year = float(params.get("periods_per_year", 1))
    out: dict[str, Any] = {}
    warnings: list[str] = []

    for concept in ("revenue", "net_income", "operating_cash_flow", "eps_diluted"):
        series = data.series.get(f"{concept}_by_period")
        if not series or len(series) < 2:
            out[f"{concept}_growth_latest"] = None
            out[f"{concept}_cagr"] = None
            continue

        values = list(series)
        out[f"{concept}_growth_latest"] = _growth(values[-1], values[-2])

        first, last = values[0], values[-1]
        spans = len(values) - 1
        if first > 0 and last > 0 and spans > 0:
            years = spans / periods_per_year
            out[f"{concept}_cagr"] = ((last / first) ** (1.0 / years)) - 1.0 if years > 0 else None
        else:
            out[f"{concept}_cagr"] = None
            if first <= 0:
                warnings.append(
                    f"{concept} CAGR is not reported because the earliest period is "
                    "zero or negative"
                )

        growths = [
            g
            for a, b in itertools.pairwise(values)
            if (g := _growth(b, a)) is not None
        ]
        out[f"{concept}_growth_periods"] = growths
        out[f"{concept}_growth_consistency"] = (
            sum(1 for g in growths if g > 0) / len(growths) if growths else None
        )

    out["periods_per_year"] = periods_per_year
    return out, warnings
