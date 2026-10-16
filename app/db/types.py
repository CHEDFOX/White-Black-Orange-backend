"""Shared column types that work on PostgreSQL and SQLite."""

from __future__ import annotations

from sqlalchemy import JSON, Numeric
from sqlalchemy.dialects.postgresql import JSONB

# JSONB on PostgreSQL (indexable, compact), plain JSON elsewhere.
JSONType = JSONB().with_variant(JSON(), "sqlite")

# Prices and quantities are stored as exact decimals, never binary floats:
# a float close of 0.1 + 0.2 must not become 0.30000000000000004 in a ledger
# that users reconcile against an exchange.
PriceNumeric = Numeric(20, 8)
VolumeNumeric = Numeric(28, 8)

# Ratios, rates and per-share figures: small magnitudes, high precision.
RatioNumeric = Numeric(20, 10)

# Reported financial and economic quantities. These span an enormous range in
# the same column -- Apple's revenue is 3.8e11 while its EPS is 2.02 -- so the
# type needs headroom for a trillion-dollar total *and* ten decimal places for
# a margin. Numeric(20, 10) caps at 10^10 and overflows on any large-cap
# revenue line, which is how this type came to exist.
FactNumeric = Numeric(38, 10)
