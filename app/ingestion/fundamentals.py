"""SEC fundamentals ingestion.

The point-in-time contract is established here. Each XBRL fact is stored with
``available_at`` taken from the filing's *acceptance* timestamp, and multiple
filings reporting the same period produce multiple rows with increasing
``revision`` -- so the restatement history is preserved rather than collapsed
to "the latest value".
"""

from __future__ import annotations

import datetime as dt
import itertools
import uuid
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.core.timeutil import UTC
from app.db.models.enums import EventCategory, JobStatus
from app.db.models.fundamentals import Filing, FinancialFact, FiscalPeriod
from app.db.models.instrument import Instrument, Issuer
from app.db.models.news import CorporateEvent
from app.db.models.provider import DataSource
from app.providers.sec_edgar import SecCompany, SecEdgarProvider, SecFact

log = get_logger(__name__)


@dataclass
class FundamentalsReport:
    issuer_id: uuid.UUID | None = None
    filings_written: int = 0
    facts_written: int = 0
    facts_unchanged: int = 0
    periods_written: int = 0
    events_written: int = 0
    status: str = JobStatus.SUCCEEDED
    notes: list[str] = field(default_factory=list)


# Duration in days -> period kind. Bounds are generous because fiscal periods
# are 13-week multiples that drift against the calendar.
_PERIOD_KINDS: tuple[tuple[int, int, str], ...] = (
    (330, 400, "FY"),
    (240, 300, "9M"),
    (150, 220, "H"),
    (60, 120, "Q"),
)


def period_kind_for(period_start: dt.date | None, period_end: dt.date) -> tuple[str, int | None]:
    """Classify a reported period by its actual duration.

    A 10-Q reports a discrete quarter *and* a year-to-date cumulative figure
    under the same ``fp`` tag. Classifying by duration is what keeps a 272-day
    nine-month total from being stored as a quarter.
    """
    if period_start is None:
        return "PIT", None
    days = (period_end - period_start).days
    for low, high, kind in _PERIOD_KINDS:
        if low <= days <= high:
            return kind, days
    return "OTHER", days


def _fiscal_year_for(period_end: dt.date, kind: str) -> int:
    """A descriptive fiscal-year label derived from the period end.

    Deliberately simple: the calendar year the period ends in. An issuer whose
    fiscal 2021 ends in January 2022 would label it differently, which is
    exactly why this is a display label and not part of the period's identity.
    """
    return period_end.year


def _fiscal_period_for(
    session: Session, issuer_id: uuid.UUID, fact: SecFact
) -> FiscalPeriod | None:
    """Find or create the period a *duration* fact belongs to.

    Instant facts (balance-sheet items) are handled separately by
    ``_attach_instant``, because they carry no duration of their own and must
    join the flow period that shares their end date.
    """
    kind, days = period_kind_for(fact.period_start, fact.period_end)
    if kind in {"PIT", "OTHER"}:
        return None

    existing = session.execute(
        select(FiscalPeriod).where(
            FiscalPeriod.issuer_id == issuer_id,
            FiscalPeriod.period_end == fact.period_end,
            FiscalPeriod.period_kind == kind,
        )
    ).scalars().first()
    if existing is not None:
        return existing

    assert fact.period_start is not None
    period = FiscalPeriod(
        issuer_id=issuer_id,
        fiscal_year=_fiscal_year_for(fact.period_end, kind),
        fiscal_period=fact.fiscal_period or kind,
        period_kind=kind,
        duration_days=days,
        period_start=fact.period_start,
        period_end=fact.period_end,
        # Calendar alignment lets a September year-end be compared with a
        # December one without pretending they are the same period.
        calendar_quarter_end=_calendar_quarter_end(fact.period_end),
    )
    session.add(period)
    session.flush()
    return period


def _attach_instant(
    session: Session, issuer_id: uuid.UUID, fact: SecFact
) -> FiscalPeriod | None:
    """Attach a balance-sheet fact to the flow period ending on the same date.

    Preference order FY > 9M > H > Q, so an annual balance sheet joins the
    annual income statement. Without this, return-on-equity would divide a
    full-year profit by a balance from a different period end.
    """
    candidates = session.execute(
        select(FiscalPeriod).where(
            FiscalPeriod.issuer_id == issuer_id,
            FiscalPeriod.period_end == fact.period_end,
        )
    ).scalars().all()
    if not candidates:
        return None
    priority = {"FY": 0, "9M": 1, "H": 2, "Q": 3, "PIT": 4, "OTHER": 5}
    return min(candidates, key=lambda p: priority.get(p.period_kind, 9))


def _calendar_quarter_end(day: dt.date) -> dt.date:
    quarter_end_month = ((day.month - 1) // 3) * 3 + 3
    if quarter_end_month == 12:
        return dt.date(day.year, 12, 31)
    next_month = dt.date(day.year, quarter_end_month + 1, 1)
    return next_month - dt.timedelta(days=1)


def ingest_filings(
    session: Session,
    *,
    issuer: Issuer,
    company: SecCompany,
    data_source: DataSource,
    form_types: tuple[str, ...] = ("10-K", "10-Q", "8-K", "20-F", "40-F"),
) -> int:
    """Store filing metadata, which anchors citations and event timelines."""
    written = 0
    for filing in company.filings:
        if filing.form not in form_types:
            continue
        existing = session.execute(
            select(Filing).where(
                Filing.data_source_id == data_source.id,
                Filing.external_id == filing.accession,
            )
        ).scalars().first()
        if existing is not None:
            continue

        accepted = filing.accepted_at or dt.datetime.combine(
            filing.filing_date, dt.time(23, 59, 59), tzinfo=UTC
        )
        session.add(
            Filing(
                issuer_id=issuer.id,
                data_source_id=data_source.id,
                external_id=filing.accession,
                form_type=filing.form,
                filed_at=accepted,
                period_of_report=filing.report_date,
                accession_number=filing.accession,
                primary_document_url=filing.url,
                title=filing.description,
                is_amendment=filing.form.endswith("/A"),
                event_time=dt.datetime.combine(
                    filing.report_date or filing.filing_date, dt.time.min, tzinfo=UTC
                ),
                published_at=accepted,
                # Availability is the acceptance instant, which is when the
                # filing genuinely became public.
                available_at=accepted,
                revision=0,
            )
        )
        written += 1
    session.flush()
    return written


def ingest_filing_events(
    session: Session,
    *,
    issuer: Issuer,
    instrument: Instrument,
    data_source: DataSource,
    limit: int = 60,
) -> int:
    """Turn filings into timeline events the chart can mark."""
    filings = session.execute(
        select(Filing)
        .where(Filing.issuer_id == issuer.id)
        .order_by(Filing.filed_at.desc())
        .limit(limit)
    ).scalars().all()

    category_for = {
        "10-K": EventCategory.EARNINGS,
        "10-Q": EventCategory.EARNINGS,
        "20-F": EventCategory.EARNINGS,
        "40-F": EventCategory.EARNINGS,
        "8-K": EventCategory.FILING,
    }
    written = 0
    for filing in filings:
        category = category_for.get(filing.form_type, EventCategory.FILING)
        existing = session.execute(
            select(CorporateEvent).where(
                CorporateEvent.instrument_id == instrument.id,
                CorporateEvent.category == category,
                CorporateEvent.event_time == filing.filed_at,
                CorporateEvent.data_source_id == data_source.id,
            )
        ).scalars().first()
        if existing is not None:
            continue

        session.add(
            CorporateEvent(
                instrument_id=instrument.id,
                issuer_id=issuer.id,
                data_source_id=data_source.id,
                category=category,
                title=f"{filing.form_type} filed",
                description=(
                    f"{filing.form_type} for the period ending "
                    f"{filing.period_of_report}" if filing.period_of_report
                    else f"{filing.form_type} filing"
                ),
                is_scheduled=filing.form_type in {"10-K", "10-Q", "20-F", "40-F"},
                is_confirmed=True,
                importance=70 if filing.form_type in {"10-K", "20-F"} else 50,
                source_url=filing.primary_document_url,
                filing_id=filing.id,
                # The event is dated when the filing became public, which is
                # also when it became actionable.
                event_time=filing.filed_at,
                published_at=filing.filed_at,
                available_at=filing.filed_at,
                revision=0,
                payload={"form_type": filing.form_type, "accession": filing.accession_number},
            )
        )
        written += 1
    session.flush()
    return written


def ingest_facts(
    session: Session,
    *,
    issuer: Issuer,
    facts: list[SecFact],
    data_source: DataSource,
) -> tuple[int, int]:
    """Store XBRL facts, one row per (period, filing).

    Returns ``(written, unchanged)``. A fact already stored from the same
    filing is unchanged; the same period reported by a *different* filing is a
    new revision, which is the restatement history the PIT layer reads.
    """
    written = 0
    unchanged = 0

    ordered_facts = sorted(facts, key=lambda f: f.filed_date)

    # Pass 1: duration facts establish the periods.
    grouped: dict[tuple[str, uuid.UUID | None], list[SecFact]] = {}
    for fact in ordered_facts:
        if fact.period_type != "duration":
            continue
        period = _fiscal_period_for(session, issuer.id, fact)
        if period is None:
            continue
        grouped.setdefault((fact.concept, period.id), []).append(fact)

    # Pass 2: instant facts join the flow period sharing their end date. Run
    # second so every period a balance sheet could attach to already exists.
    for fact in ordered_facts:
        if fact.period_type != "instant":
            continue
        period = _attach_instant(session, issuer.id, fact)
        if period is None:
            continue
        grouped.setdefault((fact.concept, period.id), []).append(fact)

    for (concept, period_id), group in grouped.items():
        existing_rows = session.execute(
            select(FinancialFact).where(
                FinancialFact.issuer_id == issuer.id,
                FinancialFact.concept == concept,
                FinancialFact.fiscal_period_id == period_id,
                FinancialFact.data_source_id == data_source.id,
            )
        ).scalars().all()
        seen_accessions = {
            (r.raw_payload or {}).get("accession") for r in existing_rows
        }
        next_revision = max((r.revision for r in existing_rows), default=-1) + 1

        for fact in group:
            if fact.accession in seen_accessions:
                unchanged += 1
                continue

            filing = session.execute(
                select(Filing).where(Filing.accession_number == fact.accession)
            ).scalars().first()

            session.add(
                FinancialFact(
                    issuer_id=issuer.id,
                    fiscal_period_id=period_id,
                    filing_id=filing.id if filing else None,
                    data_source_id=data_source.id,
                    concept=fact.concept,
                    source_concept=fact.source_concept,
                    taxonomy="us-gaap",
                    value=fact.value,
                    unit=fact.unit,
                    currency=fact.unit if len(fact.unit) == 3 else None,
                    period_type=fact.period_type,
                    is_restatement=next_revision > 0,
                    event_time=dt.datetime.combine(
                        fact.period_end, dt.time.min, tzinfo=UTC
                    ),
                    published_at=dt.datetime.combine(
                        fact.filed_date, dt.time.min, tzinfo=UTC
                    ),
                    # The instant the figure genuinely became knowable.
                    available_at=fact.available_at,
                    revision=next_revision,
                    raw_payload={
                        "accession": fact.accession,
                        "form": fact.form,
                        "fy": fact.fiscal_year,
                        "fp": fact.fiscal_period,
                        "frame": fact.frame,
                    },
                )
            )
            seen_accessions.add(fact.accession)
            next_revision += 1
            written += 1

    session.flush()
    return written, unchanged


def supersede_older_revisions(session: Session, issuer_id: uuid.UUID) -> int:
    """Mark each superseded fact revision with the successor's availability.

    Without this a point-in-time query would return every vintage of a figure
    rather than the one in force at the cutoff.
    """
    rows = session.execute(
        select(FinancialFact)
        .where(FinancialFact.issuer_id == issuer_id)
        .order_by(
            FinancialFact.concept,
            FinancialFact.fiscal_period_id,
            FinancialFact.revision,
        )
    ).scalars().all()

    by_key: dict[tuple, list[FinancialFact]] = {}
    for row in rows:
        by_key.setdefault((row.concept, row.fiscal_period_id), []).append(row)

    updated = 0
    for group in by_key.values():
        ordered = sorted(group, key=lambda r: (r.available_at, r.revision))
        for current, successor in itertools.pairwise(ordered):
            if current.superseded_at != successor.available_at:
                current.superseded_at = successor.available_at
                updated += 1
        if ordered[-1].superseded_at is not None:
            ordered[-1].superseded_at = None
            updated += 1
    session.flush()
    return updated


def ingest_company_fundamentals(
    session: Session,
    *,
    provider: SecEdgarProvider,
    data_source: DataSource,
    instrument: Instrument,
    cik: str,
) -> FundamentalsReport:
    """Full fundamentals ingestion for one issuer."""
    report = FundamentalsReport()

    company = provider.fetch_company(cik)
    issuer = instrument.issuer
    if issuer is None:
        from app.ingestion.instruments import upsert_issuer

        issuer = upsert_issuer(
            session,
            name=company.name,
            cik=company.cik,
            sic_code=company.sic,
            industry=company.sic_description,
            website=company.website,
            description=company.description,
        )
        instrument.issuer_id = issuer.id
        session.flush()
    else:
        if not issuer.cik:
            issuer.cik = company.cik
        if not issuer.sic_code:
            issuer.sic_code = company.sic
        session.flush()

    report.issuer_id = issuer.id
    report.filings_written = ingest_filings(
        session, issuer=issuer, company=company, data_source=data_source
    )

    acceptance = provider.acceptance_index(company)
    facts = provider.fetch_company_facts(cik, acceptance_index=acceptance)
    if not facts:
        report.notes.append(
            f"SEC returned no mapped XBRL facts for CIK {cik}; the issuer may file "
            "in a taxonomy this build does not map."
        )
        report.status = JobStatus.PARTIAL
        return report

    written, unchanged = ingest_facts(
        session, issuer=issuer, facts=facts, data_source=data_source
    )
    report.facts_written = written
    report.facts_unchanged = unchanged
    supersede_older_revisions(session, issuer.id)
    report.events_written = ingest_filing_events(
        session, issuer=issuer, instrument=instrument, data_source=data_source
    )

    missing_acceptance = sum(1 for f in facts if f.accepted_at is None)
    if missing_acceptance:
        report.notes.append(
            f"{missing_acceptance} fact(s) had no acceptance timestamp in the "
            "submissions index; their availability is set to end of the filing "
            "day, which is conservative."
        )
    return report
