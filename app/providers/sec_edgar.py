"""SEC EDGAR adapter -- genuinely point-in-time fundamentals.

Why this source carries the fundamentals workload: the XBRL ``companyfacts``
endpoint returns every reported value *together with the filing that reported
it and the date it was filed*. A single fiscal period therefore appears several
times -- once per filing that restated it. Selecting the row with the greatest
``filed`` date not exceeding an analysis timestamp yields the figure a reader
genuinely had on that date.

Concretely: Apple's FY2021 revenue appears in the FY2021 10-K (filed
2021-10-28) and again in the FY2023 10-K (filed 2023-11-03). A 2022 analysis
must see the first, and only the first.

The ``submissions`` endpoint supplies ``acceptanceDateTime`` -- the exact
instant a filing became public -- which is what ``available_at`` is set from
for filings and filing-derived events.

Access terms: the SEC requires a descriptive User-Agent with contact details
and asks for no more than 10 requests/second. Both are enforced here.
"""

from __future__ import annotations

import datetime as dt
import decimal
import json
from dataclasses import dataclass, field

from app.core.config import settings
from app.core.errors import ProviderError
from app.core.logging import get_logger
from app.core.timeutil import UTC, now_utc
from app.providers.base import HealthStatus, ProviderCapabilities
from app.providers.http import ProviderHttpClient

log = get_logger(__name__)

PROVIDER_KEY = "sec_edgar"

CAPABILITIES = ProviderCapabilities(
    provider_key=PROVIDER_KEY,
    display_name="SEC EDGAR (XBRL company facts and submissions)",
    kinds=frozenset({"fundamentals", "filings", "instrument_universe"}),
    resolutions=frozenset(),
    adjustment_bases=frozenset(),
    instrument_types=frozenset(),
    exchanges=frozenset({"XNAS", "XNYS", "ARCX", "BATS"}),
    max_history=None,
    supports_point_in_time=True,
    supports_search=True,
    requires_credential=False,
    rate_limit_per_minute=int(settings.sec_rate_limit_per_second * 60),
    official_api=True,
    redistribution_allowed=True,
    attribution_required=True,
    attribution_text="Fundamental data from SEC EDGAR filings (public domain).",
    licence_summary=(
        "US government work, public domain. The SEC requires a descriptive "
        "User-Agent with contact details and a request rate at or below 10/s."
    ),
    caveats=(
        "Covers SEC registrants only: US-listed issuers and foreign private "
        "issuers that file with the SEC. Non-US listings are not covered.",
        "XBRL facts begin around 2009; earlier filings are text-only.",
        "Issuers use different us-gaap concepts for the same economic quantity, "
        "so concept mapping is explicit and incomplete rather than guessed.",
    ),
)

# Normalised concept -> candidate us-gaap tags, in preference order.
# Order matters: the first tag an issuer actually reports wins, which is why
# these are lists and not a single guess per concept.
CONCEPT_MAP: dict[str, tuple[str, ...]] = {
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    ),
    "cost_of_revenue": ("CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"),
    "gross_profit": ("GrossProfit",),
    "operating_income": ("OperatingIncomeLoss",),
    "net_income": ("NetIncomeLoss", "ProfitLoss"),
    "eps_basic": ("EarningsPerShareBasic",),
    "eps_diluted": ("EarningsPerShareDiluted",),
    "shares_diluted": ("WeightedAverageNumberOfDilutedSharesOutstanding",),
    "shares_basic": ("WeightedAverageNumberOfSharesOutstandingBasic",
                     "WeightedAverageNumberOfSharesOutstanding"),
    "research_development": ("ResearchAndDevelopmentExpense",),
    "operating_expenses": ("OperatingExpenses", "CostsAndExpenses"),
    "total_assets": ("Assets",),
    "total_liabilities": ("Liabilities",),
    "stockholders_equity": (
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ),
    "cash_and_equivalents": (
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ),
    "short_term_investments": ("ShortTermInvestments", "MarketableSecuritiesCurrent"),
    "inventory": ("InventoryNet",),
    "current_assets": ("AssetsCurrent",),
    "current_liabilities": ("LiabilitiesCurrent",),
    "long_term_debt": ("LongTermDebtNoncurrent", "LongTermDebt"),
    "short_term_debt": ("LongTermDebtCurrent", "DebtCurrent"),
    "operating_cash_flow": (
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ),
    "capital_expenditure": (
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ),
    "dividends_paid": ("PaymentsOfDividendsCommonStock", "PaymentsOfDividends"),
    "share_repurchase": ("PaymentsForRepurchaseOfCommonStock",),
    "interest_expense": ("InterestExpense", "InterestIncomeExpenseNet"),
    "income_tax_expense": ("IncomeTaxExpenseBenefit",),
    "depreciation_amortization": (
        "DepreciationDepletionAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
    ),
}

# Units that make sense per concept, so a per-share figure is never stored as
# a currency total.
_PER_SHARE_CONCEPTS = frozenset({"eps_basic", "eps_diluted"})
_SHARE_COUNT_CONCEPTS = frozenset({"shares_basic", "shares_diluted"})


@dataclass(frozen=True, slots=True)
class SecFact:
    """One XBRL fact with the filing that reported it."""

    concept: str
    source_concept: str
    value: decimal.Decimal
    unit: str
    period_start: dt.date | None
    period_end: dt.date
    period_type: str  # duration | instant
    fiscal_year: int | None
    fiscal_period: str | None
    form: str
    accession: str
    filed_date: dt.date
    # Set from the submissions endpoint when available: the exact instant the
    # filing became public.
    accepted_at: dt.datetime | None = None
    frame: str | None = None

    @property
    def available_at(self) -> dt.datetime:
        if self.accepted_at is not None:
            return self.accepted_at
        # Without an acceptance timestamp, assume end of the filing day in UTC.
        # Conservative: it never makes a fact available earlier than it was.
        return dt.datetime.combine(self.filed_date, dt.time(23, 59, 59), tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class SecFiling:
    accession: str
    form: str
    filing_date: dt.date
    accepted_at: dt.datetime | None
    report_date: dt.date | None
    primary_document: str | None
    description: str | None
    is_xbrl: bool
    cik: str

    @property
    def url(self) -> str | None:
        if not self.primary_document:
            return None
        accn = self.accession.replace("-", "")
        return (
            f"https://www.sec.gov/Archives/edgar/data/{int(self.cik)}/{accn}/"
            f"{self.primary_document}"
        )


@dataclass(frozen=True, slots=True)
class SecCompany:
    cik: str
    name: str
    tickers: tuple[str, ...]
    exchanges: tuple[str, ...]
    sic: str | None
    sic_description: str | None
    fiscal_year_end: str | None
    website: str | None
    description: str | None
    ein: str | None = None
    state_of_incorporation: str | None = None
    filings: tuple[SecFiling, ...] = field(default_factory=tuple)


class SecEdgarProvider:
    capabilities = CAPABILITIES

    def __init__(self, client: ProviderHttpClient | None = None) -> None:
        if "set SEC_USER_AGENT" in settings.sec_user_agent:
            log.warning(
                "sec_user_agent_not_configured",
                detail=(
                    "SEC_USER_AGENT should carry a real contact address; the SEC "
                    "throttles or blocks anonymous clients."
                ),
            )
        headers = {"User-Agent": settings.sec_user_agent, "Accept-Encoding": "gzip, deflate"}
        self._data_client = client or ProviderHttpClient(
            base_url=settings.sec_base_url,
            provider_key=PROVIDER_KEY,
            headers=headers,
            rate_limit_per_minute=int(settings.sec_rate_limit_per_second * 60),
            timeout=90.0,
        )
        self._www_client = ProviderHttpClient(
            base_url="https://www.sec.gov",
            provider_key=f"{PROVIDER_KEY}_www",
            headers=headers,
            rate_limit_per_minute=int(settings.sec_rate_limit_per_second * 60),
            timeout=90.0,
        )

    @property
    def key(self) -> str:
        return PROVIDER_KEY

    @staticmethod
    def pad_cik(cik: str | int) -> str:
        return str(int(str(cik).lstrip("CIK").lstrip("0") or 0)).zfill(10)

    def _json(self, client: ProviderHttpClient, path: str) -> dict:
        response = client.get(path)
        if response.status_code == 404:
            raise ProviderError(
                f"SEC has no resource at {path}",
                details={"provider": PROVIDER_KEY, "path": path, "status": 404},
            )
        if response.status_code >= 400:
            raise ProviderError(
                f"SEC request to {path} failed with {response.status_code}",
                details={"provider": PROVIDER_KEY, "path": path,
                         "status": response.status_code},
            )
        try:
            return json.loads(response.text, parse_float=decimal.Decimal)
        except json.JSONDecodeError as exc:
            raise ProviderError(
                f"SEC returned non-JSON for {path}",
                details={"provider": PROVIDER_KEY, "path": path},
            ) from exc

    # --- instrument universe -------------------------------------------------

    def fetch_ticker_universe(self) -> list[tuple[str, str, str]]:
        """Return ``(ticker, cik, company_name)`` for all SEC registrants.

        ~10k rows. This is the free instrument universe that makes search work
        without a paid reference-data subscription.
        """
        payload = self._json(self._www_client, "/files/company_tickers.json")
        out: list[tuple[str, str, str]] = []
        for row in payload.values():
            if not isinstance(row, dict):
                continue
            ticker = row.get("ticker")
            cik = row.get("cik_str")
            title = row.get("title")
            if not ticker or cik is None:
                continue
            out.append((str(ticker).upper(), self.pad_cik(cik), str(title or ticker)))
        return out

    def fetch_company(self, cik: str | int, *, max_filings: int = 200) -> SecCompany:
        padded = self.pad_cik(cik)
        payload = self._json(self._data_client, f"/submissions/CIK{padded}.json")
        recent = (payload.get("filings") or {}).get("recent") or {}
        filings: list[SecFiling] = []
        accessions = recent.get("accessionNumber") or []
        for i in range(min(len(accessions), max_filings)):
            filing_date = _parse_date(_idx(recent.get("filingDate"), i))
            if filing_date is None:
                continue
            filings.append(
                SecFiling(
                    accession=str(accessions[i]),
                    form=str(_idx(recent.get("form"), i) or ""),
                    filing_date=filing_date,
                    accepted_at=_parse_datetime(_idx(recent.get("acceptanceDateTime"), i)),
                    report_date=_parse_date(_idx(recent.get("reportDate"), i)),
                    primary_document=_idx(recent.get("primaryDocument"), i),
                    description=_idx(recent.get("primaryDocDescription"), i),
                    is_xbrl=bool(_idx(recent.get("isXBRL"), i)),
                    cik=padded,
                )
            )
        return SecCompany(
            cik=padded,
            name=str(payload.get("name") or ""),
            tickers=tuple(str(t).upper() for t in (payload.get("tickers") or [])),
            exchanges=tuple(str(e) for e in (payload.get("exchanges") or [])),
            sic=str(payload.get("sic")) if payload.get("sic") else None,
            sic_description=payload.get("sicDescription"),
            fiscal_year_end=payload.get("fiscalYearEnd"),
            website=payload.get("website") or None,
            description=payload.get("description") or None,
            ein=payload.get("ein") or None,
            state_of_incorporation=payload.get("stateOfIncorporation") or None,
            filings=tuple(filings),
        )

    # --- fundamentals --------------------------------------------------------

    def fetch_company_facts(
        self,
        cik: str | int,
        *,
        concepts: tuple[str, ...] | None = None,
        acceptance_index: dict[str, dt.datetime] | None = None,
    ) -> list[SecFact]:
        """Return every reported value for the requested normalised concepts.

        Each ``(period, filing)`` pair becomes one ``SecFact``. Nothing is
        deduplicated here -- the restatement history *is* the point-in-time
        signal, and collapsing it to "the latest value" would destroy it.
        """
        padded = self.pad_cik(cik)
        payload = self._json(self._data_client, f"/api/xbrl/companyfacts/CIK{padded}.json")
        facts_root = payload.get("facts") or {}
        wanted = concepts or tuple(CONCEPT_MAP)
        out: list[SecFact] = []

        for concept in wanted:
            tags = CONCEPT_MAP.get(concept)
            if not tags:
                log.info("sec_concept_unmapped", concept=concept)
                continue
            # Read every spelling, then resolve collisions by tag preference.
            #
            # Stopping at the first tag that yields anything silently truncates
            # history, because issuers change tags mid-life. Tesla reported
            # revenue as `Revenues` through FY2020 and as
            # `RevenueFromContractWithCustomerExcludingAssessedTax` from FY2021:
            # the preferred tag covers only the later years, so breaking there
            # drops FY2019 and FY2020 entirely -- and drops them silently, as a
            # concept absent from a period looks the same as one never reported.
            best: dict[tuple, tuple[int, SecFact]] = {}
            for rank, tag in enumerate(tags):
                block = _find_tag(facts_root, tag)
                if block is None:
                    continue
                for unit, rows in (block.get("units") or {}).items():
                    if not _unit_is_sane(concept, unit):
                        continue
                    for row in rows:
                        fact = _row_to_fact(
                            concept=concept,
                            source_concept=tag,
                            unit=unit,
                            row=row,
                            acceptance_index=acceptance_index or {},
                        )
                        if fact is None:
                            continue
                        # One filing carrying a period under two spellings is a
                        # duplicate, so the preferred tag wins. The same period
                        # in a *later* filing is a revision and must survive,
                        # which is why the accession is part of the key.
                        key = (
                            fact.period_start,
                            fact.period_end,
                            fact.accession,
                            fact.unit,
                        )
                        current = best.get(key)
                        if current is None or rank < current[0]:
                            best[key] = (rank, fact)
            out.extend(fact for _, fact in best.values())

        out.sort(key=lambda f: (f.concept, f.period_end, f.filed_date))
        return out

    def acceptance_index(self, company: SecCompany) -> dict[str, dt.datetime]:
        """Map accession number -> acceptance instant, for exact availability."""
        return {
            f.accession: f.accepted_at
            for f in company.filings
            if f.accepted_at is not None
        }

    def health_check(self) -> HealthStatus:
        started = now_utc()
        try:
            response = self._data_client.get("/submissions/CIK0000320193.json")
        except Exception as exc:
            return HealthStatus(
                provider_key=PROVIDER_KEY, healthy=False, checked_at=started, detail=str(exc)[:500]
            )
        return HealthStatus(
            provider_key=PROVIDER_KEY,
            healthy=response.status_code == 200,
            checked_at=started,
            latency_ms=(now_utc() - started).total_seconds() * 1000,
            status_code=response.status_code,
        )

    def close(self) -> None:
        self._data_client.close()
        self._www_client.close()


def _find_tag(facts_root: dict, tag: str) -> dict | None:
    for taxonomy in ("us-gaap", "ifrs-full", "dei", "srt"):
        block = (facts_root.get(taxonomy) or {}).get(tag)
        if block:
            return block
    return None


def _unit_is_sane(concept: str, unit: str) -> bool:
    if concept in _PER_SHARE_CONCEPTS:
        return unit.endswith("/shares")
    if concept in _SHARE_COUNT_CONCEPTS:
        return unit == "shares"
    # Everything else is a currency amount; reject share counts and ratios that
    # would otherwise be summed into a monetary total.
    return unit in {"USD", "EUR", "GBP", "JPY", "CAD", "CHF", "AUD", "CNY", "KRW", "INR"}


def _row_to_fact(
    *,
    concept: str,
    source_concept: str,
    unit: str,
    row: dict,
    acceptance_index: dict[str, dt.datetime],
) -> SecFact | None:
    value = row.get("val")
    end = _parse_date(row.get("end"))
    filed = _parse_date(row.get("filed"))
    accession = row.get("accn")
    if value is None or end is None or filed is None or not accession:
        return None
    try:
        decimal_value = decimal.Decimal(str(value))
    except decimal.InvalidOperation:
        return None
    if not decimal_value.is_finite():
        return None

    start = _parse_date(row.get("start"))
    return SecFact(
        concept=concept,
        source_concept=source_concept,
        value=decimal_value,
        unit=unit,
        period_start=start,
        period_end=end,
        period_type="duration" if start is not None else "instant",
        fiscal_year=row.get("fy") if isinstance(row.get("fy"), int) else None,
        fiscal_period=row.get("fp") if isinstance(row.get("fp"), str) else None,
        form=str(row.get("form") or ""),
        accession=str(accession),
        filed_date=filed,
        accepted_at=acceptance_index.get(str(accession)),
        frame=row.get("frame") if isinstance(row.get("frame"), str) else None,
    )


def _idx(seq, i):
    if isinstance(seq, list) and i < len(seq):
        return seq[i]
    return None


def _parse_date(value) -> dt.date | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return dt.date.fromisoformat(value[:10])
    except ValueError:
        return None


def _parse_datetime(value) -> dt.datetime | None:
    if not isinstance(value, str) or not value:
        return None
    text = value.replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
