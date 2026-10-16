"""Forecast immutability and outcome recording.

These are the tests that make "historical forecasts are evaluated without
hindsight" a checkable claim rather than a promise.
"""

from __future__ import annotations

import datetime as dt
import math
import random

import pytest
from sqlalchemy import func, select

from app.core.errors import ImmutabilityError
from app.core.timeutil import UTC
from app.db.models.research import Forecast, ForecastOutcome
from app.forecasting.base import ForecastContext, get_model
from app.forecasting.store import (
    due_for_evaluation,
    forecast_history,
    record_outcome,
    store_forecast,
)

pytestmark = pytest.mark.database


def make_context(n=400, *, seed=1, cutoff=None):
    """Build a context whose series *ends* at the cutoff.

    Generating timestamps forward from a fixed start and then setting an
    earlier cutoff produces a context the model correctly refuses, since it
    would contain observations past its own information cutoff.
    """
    rng = random.Random(seed)
    closes = [100.0]
    for _ in range(n - 1):
        closes.append(closes[-1] * math.exp(rng.gauss(0, 0.012)))
    end = cutoff or dt.datetime(2024, 2, 5, tzinfo=UTC)
    ts = tuple(end - dt.timedelta(days=n - 1 - i) for i in range(n))
    return ForecastContext(
        timestamps=ts,
        close=tuple(closes),
        information_cutoff=end,
    )


class TestStoreForecast:
    def test_a_forecast_is_stored_with_its_cutoff_and_fingerprint(self, db, instrument):
        model = get_model("random_walk")
        context = make_context()
        dist = model.forecast(context, "5d")
        stored = store_forecast(
            db, instrument_id=instrument.id, model=model,
            context=context, distribution=dist,
        )
        assert stored.information_cutoff == context.information_cutoff
        assert stored.input_fingerprint == context.fingerprint()
        assert stored.resolves_at > stored.information_cutoff
        assert stored.quantiles
        assert stored.evaluated is False

    def test_assumptions_and_invalidation_conditions_are_persisted(self, db, instrument):
        model = get_model("random_walk")
        context = make_context()
        stored = store_forecast(
            db, instrument_id=instrument.id, model=model,
            context=context, distribution=model.forecast(context, "5d"),
        )
        assert stored.assumptions, "a reader must be able to check the assumptions"
        assert stored.invalidation_conditions, (
            "a reader must be able to check whether the forecast still applies"
        )
        assert stored.limitations

    def test_storing_the_same_forecast_twice_is_idempotent(self, db, instrument):
        model = get_model("random_walk")
        context = make_context()
        dist = model.forecast(context, "5d")
        first = store_forecast(
            db, instrument_id=instrument.id, model=model, context=context, distribution=dist
        )
        second = store_forecast(
            db, instrument_id=instrument.id, model=model, context=context, distribution=dist
        )
        assert first.id == second.id
        count = db.execute(select(func.count()).select_from(Forecast)).scalar_one()
        assert count == 1

    def test_a_forecast_without_a_cutoff_is_refused(self, db, instrument):
        model = get_model("random_walk")
        context = make_context()
        dist = model.forecast(context, "5d")
        context.information_cutoff = None
        with pytest.raises(ValueError, match="unfalsifiable"):
            store_forecast(
                db, instrument_id=instrument.id, model=model,
                context=context, distribution=dist,
            )

    def test_a_malformed_distribution_never_reaches_storage(self, db, instrument):
        model = get_model("random_walk")
        context = make_context()
        dist = model.forecast(context, "5d")
        dist.quantiles[0.75] = dist.quantiles[0.25] - 1.0  # break monotonicity
        with pytest.raises(ValueError, match="not monotonic"):
            store_forecast(
                db, instrument_id=instrument.id, model=model,
                context=context, distribution=dist,
            )


class TestImmutability:
    def test_editing_a_stored_forecast_raises(self, db, instrument):
        """The core governance guarantee."""
        model = get_model("random_walk")
        context = make_context()
        stored = store_forecast(
            db, instrument_id=instrument.id, model=model,
            context=context, distribution=model.forecast(context, "5d"),
        )
        stored.point_estimate = 0.99
        with pytest.raises(ImmutabilityError, match="cannot be modified"):
            db.flush()

    def test_editing_the_quantiles_raises(self, db, instrument):
        model = get_model("random_walk")
        context = make_context()
        stored = store_forecast(
            db, instrument_id=instrument.id, model=model,
            context=context, distribution=model.forecast(context, "5d"),
        )
        stored.quantiles = {"0.5": 1.0}
        with pytest.raises(ImmutabilityError):
            db.flush()

    def test_moving_the_information_cutoff_raises(self, db, instrument):
        """Shifting the cutoff after the fact would launder a look-ahead leak."""
        model = get_model("random_walk")
        context = make_context()
        stored = store_forecast(
            db, instrument_id=instrument.id, model=model,
            context=context, distribution=model.forecast(context, "5d"),
        )
        stored.information_cutoff = stored.information_cutoff + dt.timedelta(days=30)
        with pytest.raises(ImmutabilityError, match="information_cutoff"):
            db.flush()

    def test_deleting_a_forecast_raises(self, db, instrument):
        model = get_model("random_walk")
        context = make_context()
        stored = store_forecast(
            db, instrument_id=instrument.id, model=model,
            context=context, distribution=model.forecast(context, "5d"),
        )
        db.delete(stored)
        with pytest.raises(ImmutabilityError, match="append-only"):
            db.flush()

    def test_the_evaluated_latch_is_the_only_permitted_change(self, db, instrument):
        model = get_model("random_walk")
        context = make_context()
        stored = store_forecast(
            db, instrument_id=instrument.id, model=model,
            context=context, distribution=model.forecast(context, "5d"),
        )
        stored.evaluated = True
        db.flush()  # must not raise
        assert stored.evaluated is True


class TestRecordOutcome:
    def _stored(self, db, instrument, horizon="5d"):
        model = get_model("random_walk")
        context = make_context()
        return model, context, store_forecast(
            db, instrument_id=instrument.id, model=model,
            context=context, distribution=model.forecast(context, horizon),
        )

    def test_an_outcome_is_scored_with_crps_and_pit(self, db, instrument):
        _, _, stored = self._stored(db, instrument)
        outcome = record_outcome(
            db, stored,
            realized_value=0.01,
            realized_at=stored.resolves_at + dt.timedelta(hours=1),
        )
        assert outcome.crps is not None and outcome.crps >= 0
        assert outcome.pit_value is not None and 0.0 <= outcome.pit_value <= 1.0
        assert outcome.absolute_error == pytest.approx(0.01)
        assert outcome.interval_coverage

    def test_scoring_before_the_horizon_closes_is_refused(self, db, instrument):
        """Scoring an incomplete horizon would flatter the model."""
        _, _, stored = self._stored(db, instrument)
        with pytest.raises(ValueError, match="incomplete horizon"):
            record_outcome(
                db, stored,
                realized_value=0.01,
                realized_at=stored.resolves_at - dt.timedelta(days=1),
            )

    def test_rescoring_an_evaluated_forecast_is_refused(self, db, instrument):
        """Otherwise historical performance could be curated."""
        _, _, stored = self._stored(db, instrument)
        record_outcome(
            db, stored, realized_value=-0.05,
            realized_at=stored.resolves_at + dt.timedelta(hours=1),
        )
        with pytest.raises(ImmutabilityError, match="already been evaluated"):
            record_outcome(
                db, stored, realized_value=0.05,
                realized_at=stored.resolves_at + dt.timedelta(hours=2),
            )

    def test_a_zero_drift_model_gets_no_directional_score(self, db, instrument):
        _, _, stored = self._stored(db, instrument)
        outcome = record_outcome(
            db, stored, realized_value=0.02,
            realized_at=stored.resolves_at + dt.timedelta(hours=1),
        )
        assert outcome.direction_correct is None, (
            "the random walk makes no directional call, so scoring its sign "
            "would measure nothing"
        )

    def test_a_directional_model_is_scored_on_direction(self, db, instrument):
        model = get_model("historical_drift")
        context = make_context(n=500, seed=5)
        stored = store_forecast(
            db, instrument_id=instrument.id, model=model,
            context=context, distribution=model.forecast(context, "1mo"),
        )
        outcome = record_outcome(
            db, stored, realized_value=0.05,
            realized_at=stored.resolves_at + dt.timedelta(hours=1),
        )
        assert outcome.direction_correct is not None

    def test_the_outcome_marks_the_forecast_evaluated(self, db, instrument):
        _, _, stored = self._stored(db, instrument)
        record_outcome(
            db, stored, realized_value=0.0,
            realized_at=stored.resolves_at + dt.timedelta(hours=1),
        )
        assert stored.evaluated is True
        count = db.execute(select(func.count()).select_from(ForecastOutcome)).scalar_one()
        assert count == 1


class TestEvaluationSweep:
    def test_only_resolved_unevaluated_forecasts_are_returned(self, db, instrument):
        model = get_model("random_walk")
        early = make_context(n=400, seed=1, cutoff=dt.datetime(2023, 6, 1, tzinfo=UTC))
        late = make_context(n=400, seed=2, cutoff=dt.datetime(2030, 1, 1, tzinfo=UTC))

        store_forecast(
            db, instrument_id=instrument.id, model=model,
            context=early, distribution=model.forecast(early, "5d"),
        )
        store_forecast(
            db, instrument_id=instrument.id, model=model,
            context=late, distribution=model.forecast(late, "5d"),
        )
        due = due_for_evaluation(db, as_of=dt.datetime(2024, 1, 1, tzinfo=UTC))
        assert len(due) == 1
        assert due[0].information_cutoff == early.information_cutoff

    def test_an_evaluated_forecast_drops_out_of_the_sweep(self, db, instrument):
        model = get_model("random_walk")
        context = make_context(cutoff=dt.datetime(2023, 6, 1, tzinfo=UTC))
        stored = store_forecast(
            db, instrument_id=instrument.id, model=model,
            context=context, distribution=model.forecast(context, "5d"),
        )
        record_outcome(
            db, stored, realized_value=0.0,
            realized_at=stored.resolves_at + dt.timedelta(hours=1),
        )
        assert due_for_evaluation(db, as_of=dt.datetime(2024, 1, 1, tzinfo=UTC)) == []


class TestForecastHistory:
    def test_history_is_returned_newest_first(self, db, instrument):
        model = get_model("random_walk")
        for day in (1, 15, 28):
            context = make_context(
                n=400, seed=day, cutoff=dt.datetime(2023, 6, day, tzinfo=UTC)
            )
            store_forecast(
                db, instrument_id=instrument.id, model=model,
                context=context, distribution=model.forecast(context, "5d"),
            )
        history = forecast_history(db, instrument_id=instrument.id)
        assert len(history) == 3
        cutoffs = [f.information_cutoff for f in history]
        assert cutoffs == sorted(cutoffs, reverse=True)

    def test_history_can_be_filtered_to_resolved_forecasts(self, db, instrument):
        model = get_model("random_walk")
        resolved_ctx = make_context(cutoff=dt.datetime(2023, 6, 1, tzinfo=UTC))
        pending_ctx = make_context(seed=9, cutoff=dt.datetime(2030, 1, 1, tzinfo=UTC))
        resolved = store_forecast(
            db, instrument_id=instrument.id, model=model,
            context=resolved_ctx, distribution=model.forecast(resolved_ctx, "5d"),
        )
        store_forecast(
            db, instrument_id=instrument.id, model=model,
            context=pending_ctx, distribution=model.forecast(pending_ctx, "5d"),
        )
        record_outcome(
            db, resolved, realized_value=0.0,
            realized_at=resolved.resolves_at + dt.timedelta(hours=1),
        )
        only_resolved = forecast_history(
            db, instrument_id=instrument.id, include_unresolved=False
        )
        assert len(only_resolved) == 1
        assert only_resolved[0].id == resolved.id
