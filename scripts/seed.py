"""Seed the platform with real data from the configured providers.

    python -m scripts.seed                 # default symbol set, 5 years
    python -m scripts.seed --years 10 --symbols AAPL.US,MSFT.US
    python -m scripts.seed --skip-macro --skip-fundamentals

Idempotent: re-running it re-fetches and finds nothing changed rather than
duplicating anything.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys

from app.analysis.registry import sync_registry_to_db
from app.core.config import settings
from app.core.logging import get_logger
from app.core.timeutil import now_utc
from app.db.models.enums import AdjustmentBasis
from app.db.session import session_scope
from app.ingestion.corporate_actions import ingest_corporate_actions
from app.ingestion.fundamentals import ingest_company_fundamentals
from app.ingestion.instruments import register_instrument, upsert_issuer
from app.ingestion.macro import ingest_curated_series
from app.ingestion.prices import backfill_prices
from app.ingestion.sources import register_all_known_sources
from app.providers.eodhd import DEMO_SYMBOLS, EodhdProvider
from app.providers.fred import FredProvider
from app.providers.sec_edgar import SecEdgarProvider

log = get_logger(__name__)

# The EODHD demo token's symbol set. VTI is included because it serves as the
# benchmark; SPY is not available on the demo token.
DEFAULT_SYMBOLS = ("AAPL.US", "MSFT.US", "TSLA.US", "AMZN.US", "VTI.US")


def seed_instrument(session, provider: EodhdProvider, source, symbol: str, years: int) -> dict:
    summary: dict = {"symbol": symbol}
    resolved = provider.resolve_instrument(symbol)
    if resolved is None:
        summary["error"] = "could not be resolved by the provider"
        return summary

    raw = resolved.raw or {}
    issuer = None
    if raw.get("cik") or raw.get("sector"):
        issuer = upsert_issuer(
            session,
            name=resolved.name,
            cik=(raw.get("cik") or None),
            sector=raw.get("sector"),
            industry=raw.get("industry"),
            country=resolved.country,
            website=raw.get("website"),
            description=(raw.get("description") or None),
        )

    instrument = register_instrument(session, resolved, source, issuer=issuer)
    summary["instrument_id"] = str(instrument.id)
    summary["name"] = instrument.name
    summary["exchange"] = instrument.exchange_code

    end = now_utc()
    start = end - dt.timedelta(days=365 * years)
    for basis in (AdjustmentBasis.RAW, AdjustmentBasis.TOTAL_RETURN):
        report = backfill_prices(
            session,
            provider=provider,
            data_source=source,
            instrument=instrument,
            provider_symbol=resolved.provider_symbol,
            start=start,
            end=end,
            resolution="1d",
            adjustment_basis=basis,
        )
        summary[f"{basis}_bars"] = report.bars_inserted + report.bars_unchanged
        summary[f"{basis}_inserted"] = report.bars_inserted
        summary[f"{basis}_gaps"] = report.gaps_recorded
        summary[f"{basis}_rejected"] = report.bars_rejected

    actions = provider.fetch_corporate_actions(
        resolved.provider_symbol, start.date(), end.date()
    )
    action_report = ingest_corporate_actions(
        session, instrument=instrument, data_source=source, actions=actions
    )
    summary["corporate_actions"] = action_report.written
    summary["action_events"] = action_report.events_written
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed the platform with real data")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--macro-years", type=int, default=15)
    parser.add_argument("--skip-prices", action="store_true")
    parser.add_argument("--skip-fundamentals", action="store_true")
    parser.add_argument("--skip-macro", action="store_true")
    args = parser.parse_args(argv)

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    with session_scope() as session:
        sources = register_all_known_sources(session)
        methods = sync_registry_to_db(session)
        print(f"registered {len(sources)} data sources, {methods} new method definitions")

        if settings.eodhd_api_key == "demo":
            unsupported = [s for s in symbols if s.upper() not in DEMO_SYMBOLS]
            if unsupported:
                print(
                    f"note: EODHD_API_KEY is 'demo', which does not serve "
                    f"{', '.join(unsupported)}. Those will fail. Supported on the "
                    f"demo token: {', '.join(sorted(DEMO_SYMBOLS))}"
                )

        if not args.skip_prices:
            print(f"\n--- prices and corporate actions ({args.years}y) ---")
            provider = EodhdProvider()
            for symbol in symbols:
                try:
                    summary = seed_instrument(
                        session, provider, sources["eodhd"], symbol, args.years
                    )
                except Exception as exc:
                    print(f"  {symbol:14s} FAILED: {type(exc).__name__}: {exc}")
                    continue
                if "error" in summary:
                    print(f"  {symbol:14s} {summary['error']}")
                    continue
                print(
                    f"  {summary['symbol']:14s} {summary.get('name', '')[:28]:30s} "
                    f"raw={summary.get('raw_bars', 0):5d} "
                    f"adj={summary.get('total_return_bars', 0):5d} "
                    f"actions={summary.get('corporate_actions', 0):3d} "
                    f"gaps={summary.get('total_return_gaps', 0):2d}"
                )
            provider.close()

        if not args.skip_fundamentals:
            print("\n--- fundamentals (SEC EDGAR, point-in-time) ---")
            sec = SecEdgarProvider()
            from sqlalchemy import select

            from app.db.models.instrument import Instrument

            instruments = session.execute(
                select(Instrument).where(Instrument.issuer_id.isnot(None))
            ).scalars().all()
            for instrument in instruments:
                cik = instrument.issuer.cik if instrument.issuer else None
                if not cik:
                    print(f"  {instrument.symbol:14s} no CIK, skipped")
                    continue
                try:
                    report = ingest_company_fundamentals(
                        session,
                        provider=sec,
                        data_source=sources["sec_edgar"],
                        instrument=instrument,
                        cik=cik,
                    )
                except Exception as exc:
                    print(f"  {instrument.symbol:14s} FAILED: {type(exc).__name__}: {exc}")
                    continue
                print(
                    f"  {instrument.symbol:14s} filings={report.filings_written:3d} "
                    f"facts={report.facts_written:5d} "
                    f"unchanged={report.facts_unchanged:5d} "
                    f"events={report.events_written:3d}"
                )
                for note in report.notes:
                    print(f"      note: {note}")
            sec.close()

        if not args.skip_macro:
            print(f"\n--- macro (FRED, {args.macro_years}y) ---")
            fred = FredProvider()
            if not fred.has_vintages:
                print(
                    "  note: no FRED_API_KEY configured, so only the latest revision "
                    "of each series is available. Point-in-time macro analysis will "
                    "disclose this."
                )
            start = (now_utc() - dt.timedelta(days=365 * args.macro_years)).date()
            reports = ingest_curated_series(
                session, provider=fred, data_source=sources["fred"], start=start
            )
            for report in reports:
                print(
                    f"  {report.series_code:14s} obs={report.observations_written:6d} "
                    f"unchanged={report.observations_unchanged:6d} "
                    f"vintages={report.has_vintages} status={report.status}"
                )
            fred.close()

    print("\nseed complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
