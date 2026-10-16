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
RatioNumeric = Numeric(20, 10)
