"""Method registry, computed features, episodes, models, forecasts and backtests.

This module holds the governance spine of the platform. Three invariants are
enforced here rather than left to convention:

* A computed feature records the *method version* and *input data version* that
  produced it, so any number in a report can be recomputed.
* A forecast is append-only and stores its information cutoff. Nothing may
  rewrite a prediction after the outcome is known.
* A backtest records its execution assumptions (costs, slippage, fill rule)
  because a backtest without them is not a result, it is a hope.
"""

from __future__ import annotations

import datetime as dt
import decimal
import uuid

from sqlalchemy import (
    Boolean,
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

from app.db.base import Base, TimestampMixin
from app.db.models.enums import (
    DataClassification,
    ForecastTarget,
    JobStatus,
    ModelLifecycle,
    RegimeLabel,
    ValidationStatus,
)
from app.db.types import JSONType, RatioNumeric


class MethodDefinition(Base, TimestampMixin):
    """Registered analytical method, mirrored from the in-code registry.

    The code registry is the source of truth at runtime; this table is its
    durable record so a feature row from six months ago can still name the
    method and version that produced it even if the code has moved on.
    """

    __tablename__ = "method_definitions"
    __table_args__ = (
        UniqueConstraint("key", "version", name="uq_method_definitions_key_version"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    key: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(20), nullable=False)
    family: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    display_name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    purpose: Mapped[str | None] = mapped_column(Text)
    required_inputs: Mapped[list] = mapped_column(JSONType, default=list)
    supported_instrument_types: Mapped[list] = mapped_column(JSONType, default=list)
    supported_resolutions: Mapped[list] = mapped_column(JSONType, default=list)
    min_observations: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    output_schema: Mapped[dict] = mapped_column(JSONType, default=dict)
    missing_data_policy: Mapped[str] = mapped_column(Text, nullable=False, default="")
    limitations: Mapped[list] = mapped_column(JSONType, default=list)
    references: Mapped[list] = mapped_column(JSONType, default=list)
    validation_status: Mapped[str] = mapped_column(
        String(16), nullable=False, default=ValidationStatus.UNTESTED
    )
    cost_class: Mapped[str] = mapped_column(String(12), nullable=False, default="cheap")
    code_fingerprint: Mapped[str | None] = mapped_column(String(64))


class ComputedFeature(Base):
    """A cached method output, tied to its method version and input window."""

    __tablename__ = "computed_features"
    __table_args__ = (
        UniqueConstraint(
            "instrument_id",
            "method_key",
            "method_version",
            "resolution",
            "as_of",
            "params_hash",
            name="uq_computed_features_instrument_method_version_res_asof_params",
        ),
        Index("ix_computed_features_lookup", "instrument_id", "method_key", "as_of"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    instrument_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE"), nullable=False
    )
    method_key: Mapped[str] = mapped_column(String(80), nullable=False)
    method_version: Mapped[str] = mapped_column(String(20), nullable=False)
    resolution: Mapped[str] = mapped_column(String(8), nullable=False)
    # The information cutoff of the inputs, not the compute time.
    as_of: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    params: Mapped[dict] = mapped_column(JSONType, default=dict)
    params_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    value: Mapped[decimal.Decimal | None] = mapped_column(RatioNumeric)
    values: Mapped[dict | None] = mapped_column(JSONType)
    observations_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    input_window_start: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    input_window_end: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    input_fingerprint: Mapped[str | None] = mapped_column(String(64))
    warnings: Mapped[list] = mapped_column(JSONType, default=list)
    computed_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    classification: Mapped[str] = mapped_column(
        String(16), nullable=False, default=DataClassification.PRODUCTION
    )


class RegimeEstimate(Base):
    """An estimated market regime for a window. Explicitly an estimate."""

    __tablename__ = "regime_estimates"
    __table_args__ = (
        UniqueConstraint(
            "instrument_id", "method_key", "method_version", "window_end",
            name="uq_regime_estimates_instrument_method_version_window_end",
        ),
        Index("ix_regime_estimates_lookup", "instrument_id", "window_end"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    instrument_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE"), nullable=False
    )
    method_key: Mapped[str] = mapped_column(String(80), nullable=False)
    method_version: Mapped[str] = mapped_column(String(20), nullable=False)
    window_start: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    window_end: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    label: Mapped[str] = mapped_column(
        String(20), nullable=False, default=RegimeLabel.INDETERMINATE
    )
    confidence: Mapped[float | None] = mapped_column(Float)
    label_probabilities: Mapped[dict] = mapped_column(JSONType, default=dict)
    diagnostics: Mapped[dict] = mapped_column(JSONType, default=dict)
    computed_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class HistoricalEpisode(Base):
    """An indexed historical window with its feature snapshot and later outcome.

    ``forward_*`` columns are the only place future-relative-to-window data is
    stored, and they are never readable by a point-in-time query for that
    window's own analysis. The similarity engine uses them only after a match is
    made on the backward-looking snapshot.
    """

    __tablename__ = "historical_episodes"
    __table_args__ = (
        UniqueConstraint(
            "instrument_id", "resolution", "window_end", "feature_set_version",
            name="uq_historical_episodes_instrument_res_window_end_featureset",
        ),
        Index("ix_historical_episodes_lookup", "resolution", "feature_set_version", "window_end"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    instrument_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE"), nullable=False
    )
    resolution: Mapped[str] = mapped_column(String(8), nullable=False)
    window_start: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    window_end: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    window_bars: Mapped[int] = mapped_column(Integer, nullable=False)

    feature_set_version: Mapped[str] = mapped_column(String(20), nullable=False)
    # Normalised, scale-free feature vector used for similarity search.
    feature_vector: Mapped[dict] = mapped_column(JSONType, nullable=False)
    regime_label: Mapped[str | None] = mapped_column(String(20))
    event_categories: Mapped[list] = mapped_column(JSONType, default=list)

    forward_returns: Mapped[dict] = mapped_column(JSONType, default=dict)
    forward_max_drawdown: Mapped[dict] = mapped_column(JSONType, default=dict)
    forward_realized_vol: Mapped[dict] = mapped_column(JSONType, default=dict)
    forward_complete_horizons: Mapped[list] = mapped_column(JSONType, default=list)

    dataset_version: Mapped[str | None] = mapped_column(String(40))
    computed_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    classification: Mapped[str] = mapped_column(
        String(16), nullable=False, default=DataClassification.PRODUCTION
    )


class ModelDefinition(Base, TimestampMixin):
    """A forecasting or scoring model, versioned and lifecycle-managed."""

    __tablename__ = "model_definitions"
    __table_args__ = (
        UniqueConstraint("key", "version", name="uq_model_definitions_key_version"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    key: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(20), nullable=False)
    family: Mapped[str] = mapped_column(String(40), nullable=False)
    display_name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    target: Mapped[str] = mapped_column(
        String(24), nullable=False, default=ForecastTarget.LOG_RETURN
    )
    supported_horizons: Mapped[list] = mapped_column(JSONType, default=list)
    hyperparameters: Mapped[dict] = mapped_column(JSONType, default=dict)
    is_baseline: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    lifecycle: Mapped[str] = mapped_column(
        String(16), nullable=False, default=ModelLifecycle.EXPERIMENTAL
    )
    lifecycle_reason: Mapped[str | None] = mapped_column(Text)
    limitations: Mapped[list] = mapped_column(JSONType, default=list)
    min_observations: Mapped[int] = mapped_column(Integer, default=60)
    code_fingerprint: Mapped[str | None] = mapped_column(String(64))

    forecasts: Mapped[list[Forecast]] = relationship(back_populates="model")


class Forecast(Base):
    """An immutable prediction.

    Immutability is enforced in three places: no UPDATE path exists in the
    repository layer, an ORM event listener raises on modification, and a test
    asserts both. ``information_cutoff`` is the contract that makes later
    evaluation honest -- the model saw nothing after this instant.
    """

    __tablename__ = "forecasts"
    __table_args__ = (
        UniqueConstraint(
            "instrument_id",
            "model_definition_id",
            "target",
            "horizon_label",
            "information_cutoff",
            name="uq_forecasts_instrument_model_target_horizon_cutoff",
        ),
        Index("ix_forecasts_lookup", "instrument_id", "target", "information_cutoff"),
        Index("ix_forecasts_resolve", "resolves_at", "evaluated"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    instrument_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE"), nullable=False
    )
    model_definition_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("model_definitions.id", ondelete="RESTRICT"), nullable=False
    )

    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    # The instant beyond which no input was visible to the model.
    information_cutoff: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    # When the outcome becomes observable.
    resolves_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    target: Mapped[str] = mapped_column(String(24), nullable=False)
    horizon_label: Mapped[str] = mapped_column(String(16), nullable=False)
    horizon_bars: Mapped[int] = mapped_column(Integer, nullable=False)
    resolution: Mapped[str] = mapped_column(String(8), nullable=False)

    point_estimate: Mapped[float | None] = mapped_column(Float)
    # Quantiles as {"0.05": -0.08, "0.5": 0.001, ...}. The distribution, not a
    # single number, is the product.
    quantiles: Mapped[dict] = mapped_column(JSONType, default=dict)
    prob_positive: Mapped[float | None] = mapped_column(Float)
    predicted_volatility: Mapped[float | None] = mapped_column(Float)
    distribution_family: Mapped[str | None] = mapped_column(String(40))
    distribution_params: Mapped[dict] = mapped_column(JSONType, default=dict)

    base_price: Mapped[decimal.Decimal | None] = mapped_column(RatioNumeric)
    contributing_factors: Mapped[list] = mapped_column(JSONType, default=list)
    assumptions: Mapped[list] = mapped_column(JSONType, default=list)
    invalidation_conditions: Mapped[list] = mapped_column(JSONType, default=list)
    limitations: Mapped[list] = mapped_column(JSONType, default=list)

    observations_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    input_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    data_quality_flags: Mapped[list] = mapped_column(JSONType, default=list)
    dataset_version: Mapped[str | None] = mapped_column(String(40))
    regime_at_cutoff: Mapped[str | None] = mapped_column(String(20))

    evaluated: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    classification: Mapped[str] = mapped_column(
        String(16), nullable=False, default=DataClassification.PRODUCTION
    )

    model: Mapped[ModelDefinition] = relationship(back_populates="forecasts")
    outcome: Mapped[ForecastOutcome | None] = relationship(
        back_populates="forecast", uselist=False
    )


class ForecastOutcome(Base):
    """The realised outcome of a forecast, written once the horizon closes."""

    __tablename__ = "forecast_outcomes"
    __table_args__ = (
        UniqueConstraint("forecast_id", name="uq_forecast_outcomes_forecast_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    forecast_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("forecasts.id", ondelete="CASCADE"), nullable=False
    )
    evaluated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    realized_value: Mapped[float | None] = mapped_column(Float)
    realized_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))

    error: Mapped[float | None] = mapped_column(Float)
    absolute_error: Mapped[float | None] = mapped_column(Float)
    squared_error: Mapped[float | None] = mapped_column(Float)
    # Continuous Ranked Probability Score: scores the whole distribution, which
    # is why a probabilistic model is not judged on direction alone.
    crps: Mapped[float | None] = mapped_column(Float)
    log_score: Mapped[float | None] = mapped_column(Float)
    pit_value: Mapped[float | None] = mapped_column(Float)
    interval_coverage: Mapped[dict] = mapped_column(JSONType, default=dict)
    direction_correct: Mapped[bool | None] = mapped_column(Boolean)
    benchmark_errors: Mapped[dict] = mapped_column(JSONType, default=dict)
    notes: Mapped[str | None] = mapped_column(Text)

    forecast: Mapped[Forecast] = relationship(back_populates="outcome")


class Backtest(Base):
    """A walk-forward evaluation run with its execution assumptions recorded."""

    __tablename__ = "backtests"
    __table_args__ = (Index("ix_backtests_model", "model_definition_id", "created_at"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    model_definition_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("model_definitions.id", ondelete="CASCADE"), nullable=False
    )
    instrument_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE")
    )
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default=JobStatus.PENDING)

    train_start: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    test_start: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    test_end: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    resolution: Mapped[str] = mapped_column(String(8), nullable=False)
    horizon_label: Mapped[str] = mapped_column(String(16), nullable=False)
    scheme: Mapped[str] = mapped_column(String(32), nullable=False, default="walk_forward")
    refit_every_bars: Mapped[int | None] = mapped_column(Integer)
    min_train_bars: Mapped[int] = mapped_column(Integer, nullable=False, default=250)

    # Execution assumptions. A strategy result without these is not reportable.
    cost_bps_per_trade: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    slippage_bps: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    fill_rule: Mapped[str] = mapped_column(String(40), nullable=False, default="next_bar_open")
    position_sizing: Mapped[str] = mapped_column(String(40), nullable=False, default="unit")

    n_forecasts: Mapped[int] = mapped_column(Integer, default=0)
    metrics: Mapped[dict] = mapped_column(JSONType, default=dict)
    benchmark_metrics: Mapped[dict] = mapped_column(JSONType, default=dict)
    per_regime_metrics: Mapped[dict] = mapped_column(JSONType, default=dict)
    equity_curve: Mapped[dict | None] = mapped_column(JSONType)
    limitations: Mapped[list] = mapped_column(JSONType, default=list)
    dataset_version: Mapped[str | None] = mapped_column(String(40))
    error: Mapped[str | None] = mapped_column(Text)
    classification: Mapped[str] = mapped_column(
        String(16), nullable=False, default=DataClassification.PRODUCTION
    )


class ResearchExperiment(Base):
    """Experiment log for multiple-testing control.

    Every hypothesis test run against the same data is recorded here. Without
    this count, a p-value from the fiftieth variant of a strategy means nothing.
    """

    __tablename__ = "research_experiments"
    __table_args__ = (Index("ix_research_experiments_family", "hypothesis_family", "created_at"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    hypothesis_family: Mapped[str] = mapped_column(String(120), nullable=False)
    hypothesis: Mapped[str] = mapped_column(Text, nullable=False)
    method_key: Mapped[str | None] = mapped_column(String(80))
    method_version: Mapped[str | None] = mapped_column(String(20))
    params: Mapped[dict] = mapped_column(JSONType, default=dict)
    dataset_version: Mapped[str | None] = mapped_column(String(40))
    n_observations: Mapped[int | None] = mapped_column(Integer)
    statistic: Mapped[float | None] = mapped_column(Float)
    p_value: Mapped[float | None] = mapped_column(Float)
    # Number of tests in the family at the time this one ran.
    family_test_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    adjusted_p_value: Mapped[float | None] = mapped_column(Float)
    adjustment_method: Mapped[str | None] = mapped_column(String(40))
    conclusion: Mapped[str | None] = mapped_column(Text)
    notes: Mapped[str | None] = mapped_column(Text)
