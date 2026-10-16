"""News articles, extracted claims and corporate events.

Two guarantees shape this schema:

1. Nothing enters a historical explanation before it was published. Every row
   carries ``published_at``/``available_at`` from the PIT mixin.
2. An LLM may summarise what is stored here, but every figure and citation in a
   report traces to a row with a real source URL and timestamp. There is no
   column in which a language model may record an invented fact.
"""

from __future__ import annotations

import datetime as dt
import uuid

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.db.models.enums import EventCategory
from app.db.models.mixins import PointInTimeMixin
from app.db.types import JSONType


class NewsArticle(Base, PointInTimeMixin):
    __tablename__ = "news_articles"
    __table_args__ = (
        UniqueConstraint(
            "data_source_id", "external_id", "revision",
            name="uq_news_articles_data_source_id_external_id_revision",
        ),
        Index("ix_news_articles_published", "published_at"),
        Index("ix_news_articles_dedupe", "content_hash"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    external_id: Mapped[str] = mapped_column(String(200), nullable=False)
    headline: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str | None] = mapped_column(Text)
    # Full body stored only where the provider licence permits retention.
    body: Mapped[str | None] = mapped_column(Text)
    url: Mapped[str | None] = mapped_column(String(1000))
    publisher: Mapped[str | None] = mapped_column(String(200), index=True)
    author: Mapped[str | None] = mapped_column(String(300))
    language: Mapped[str | None] = mapped_column(String(8))

    # Normalised hash of headline+body used for syndication/duplicate detection.
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    duplicate_of_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("news_articles.id", ondelete="SET NULL")
    )

    # Source-quality score is an explicit, documented heuristic -- not a claim
    # about truthfulness.
    source_quality_score: Mapped[float | None] = mapped_column(Float)
    relevance_score: Mapped[float | None] = mapped_column(Float)

    # Sentiment is a model estimate with its own uncertainty, never a fact.
    sentiment_score: Mapped[float | None] = mapped_column(Float)
    sentiment_confidence: Mapped[float | None] = mapped_column(Float)
    sentiment_model: Mapped[str | None] = mapped_column(String(80))

    raw_payload: Mapped[dict | None] = mapped_column(JSONType)

    entities: Mapped[list[NewsEntityLink]] = relationship(
        back_populates="article", cascade="all, delete-orphan"
    )
    claims: Mapped[list[ExtractedClaim]] = relationship(
        back_populates="article", cascade="all, delete-orphan"
    )


class NewsEntityLink(Base):
    """Links an article to an instrument or issuer, with how the link was made."""

    __tablename__ = "news_entity_links"
    __table_args__ = (
        UniqueConstraint(
            "article_id", "instrument_id", name="uq_news_entity_links_article_id_instrument_id"
        ),
        Index("ix_news_entity_links_instrument", "instrument_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    article_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("news_articles.id", ondelete="CASCADE"), nullable=False
    )
    instrument_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE")
    )
    issuer_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("issuers.id", ondelete="CASCADE")
    )
    # exact_ticker | name_match | provider_tag | manual
    link_method: Mapped[str] = mapped_column(String(24), nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float)
    is_primary_subject: Mapped[bool] = mapped_column(Boolean, default=False)

    article: Mapped[NewsArticle] = relationship(back_populates="entities")


class ExtractedClaim(Base):
    """A statement extracted from an article, with the passage that supports it.

    ``supporting_passage`` must be a verbatim substring of the source. This is
    what makes a citation checkable instead of decorative.
    """

    __tablename__ = "extracted_claims"
    __table_args__ = (Index("ix_extracted_claims_article", "article_id"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    article_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("news_articles.id", ondelete="CASCADE"), nullable=False
    )
    claim_text: Mapped[str] = mapped_column(Text, nullable=False)
    supporting_passage: Mapped[str] = mapped_column(Text, nullable=False)
    passage_offset: Mapped[int | None] = mapped_column(Integer)
    claim_category: Mapped[str | None] = mapped_column(String(40))
    extraction_method: Mapped[str] = mapped_column(String(60), nullable=False)
    extraction_model: Mapped[str | None] = mapped_column(String(80))
    confidence: Mapped[float | None] = mapped_column(Float)
    extracted_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    article: Mapped[NewsArticle] = relationship(back_populates="claims")


class CorporateEvent(Base, PointInTimeMixin):
    """A dated, categorised event attached to an instrument.

    This is the timeline the UI renders as chart markers. It unifies earnings
    dates, filings, corporate actions and macro releases into one query.
    """

    __tablename__ = "corporate_events"
    __table_args__ = (
        UniqueConstraint(
            "instrument_id", "category", "event_time", "data_source_id", "revision",
            name="uq_corporate_events_instrument_category_time_source_revision",
        ),
        Index("ix_corporate_events_timeline", "instrument_id", "event_time"),
        Index("ix_corporate_events_pit", "instrument_id", "available_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    instrument_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE")
    )
    issuer_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("issuers.id", ondelete="CASCADE")
    )
    category: Mapped[str] = mapped_column(String(24), nullable=False, default=EventCategory.OTHER)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)

    # Scheduled events (earnings) have an expected date before they occur; the
    # distinction between "expected" and "confirmed" matters for PIT analysis.
    is_scheduled: Mapped[bool] = mapped_column(Boolean, default=False)
    is_confirmed: Mapped[bool] = mapped_column(Boolean, default=True)
    expected_on: Mapped[dt.date | None] = mapped_column(Date)

    # Rough magnitude hint for UI marker sizing. Explicitly a heuristic.
    importance: Mapped[int] = mapped_column(Integer, default=50, nullable=False)

    source_url: Mapped[str | None] = mapped_column(String(1000))
    filing_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("filings.id", ondelete="SET NULL")
    )
    corporate_action_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("corporate_actions.id", ondelete="SET NULL")
    )
    article_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("news_articles.id", ondelete="SET NULL")
    )
    payload: Mapped[dict] = mapped_column(JSONType, default=dict)
