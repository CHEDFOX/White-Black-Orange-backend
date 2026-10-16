"""End-to-end smoke run: register sources, resolve an instrument, ingest, query.

Run it against a live provider to prove the whole path works:

    python -m scripts.ingest_demo AAPL.US
"""

from __future__ import annotations

import datetime as dt
import sys

from app.core.logging import get_logger
from app.core.timeutil import UTC, now_utc
from app.db.models.enums import AdjustmentBasis
from app.db.session import session_scope
from app.ingestion.instruments import register_instrument, upsert_issuer
from app.ingestion.prices import backfill_prices
from app.ingestion.sources import register_all_known_sources
from app.pit.availability import AsOf
from app.pit.queries import price_bars_as_of
from app.providers.registry import get_provider

log = get_logger(__name__)


def main(symbol: str = "AAPL.US", years: int = 3) -> int:
    with session_scope() as session:
        sources = register_all_known_sources(session)
        source = sources["eodhd"]
        provider = get_provider("eodhd")

        resolved = provider.resolve_instrument(symbol)
        if resolved is None:
            print(f"could not resolve {symbol}")
            return 1

        issuer = None
        raw = resolved.raw or {}
        if raw.get("cik") or raw.get("sector"):
            issuer = upsert_issuer(
                session,
                name=resolved.name,
                cik=(raw.get("cik") or None),
                sector=raw.get("sector"),
                industry=raw.get("industry"),
                country=resolved.country,
                website=raw.get("website"),
            )

        instrument = register_instrument(session, resolved, source, issuer=issuer)
        print(f"instrument: {instrument.symbol}@{instrument.exchange_code} "
              f"({instrument.name}) id={instrument.id}")

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
            print(
                f"  {basis:14s} received={report.bars_received} "
                f"inserted={report.bars_inserted} unchanged={report.bars_unchanged} "
                f"revised={report.bars_revised} rejected={report.bars_rejected} "
                f"gaps={report.gaps_recorded} status={report.status}"
            )
            for kind, n in sorted(_counts(report).items()):
                print(f"      finding {kind}: {n}")

        # Point-in-time read: what did the record show a year ago?
        a_year_ago = AsOf(cutoff=end - dt.timedelta(days=365), purpose="demo")
        past = price_bars_as_of(session, instrument.id, a_year_ago, limit=5)
        print(f"  as of {a_year_ago.cutoff.date()}: {past.coverage.rows_returned} bars visible")
        latest = price_bars_as_of(session, instrument.id, AsOf.now())
        print(f"  as of now: {latest.coverage.rows_returned} bars visible")
        if latest.rows:
            last = latest.rows[-1]
            print(f"  most recent bar: {last.event_time.date()} close={last.close}")
    return 0


def _counts(report) -> dict[str, int]:
    out: dict[str, int] = {}
    for f in report.findings:
        out[f.kind] = out.get(f.kind, 0) + 1
    return out


if __name__ == "__main__":
    sys.exit(main(*(sys.argv[1:] or ["AAPL.US"])))
