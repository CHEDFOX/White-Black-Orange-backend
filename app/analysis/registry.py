"""Analytical method registry.

Every method the platform can run is declared here with its inputs, minimum
data requirement, output schema, missing-data policy and limitations. Three
things follow from that declaration, and they are the reason the registry
exists rather than a module of loose functions:

1. **A method can refuse.** If the data does not meet ``min_observations`` the
   runner raises ``InsufficientDataError`` instead of returning a number
   computed from four data points.
2. **A result is reproducible.** Each output carries the method key, version
   and a fingerprint of the exact inputs, so any figure in a report can be
   recomputed years later.
3. **Limitations travel with the number.** A report cannot cite a method
   without also having access to what that method cannot tell you.

Versioning rule: a change to a method's output for the same input is a version
bump. Anything else -- a comment, a refactor, a performance fix -- is not.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.core.errors import InsufficientDataError, NotFoundError
from app.core.logging import get_logger
from app.db.models.enums import ValidationStatus

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class MethodSpec:
    """Declaration of one analytical method."""

    key: str
    version: str
    family: str
    display_name: str
    description: str
    # What the method needs. Names match the keys of ``MethodInput``.
    required_inputs: tuple[str, ...]
    # Minimum usable observations. Below this the method refuses to run.
    min_observations: int
    output_schema: dict[str, str]
    missing_data_policy: str
    limitations: tuple[str, ...]
    purpose: str = ""
    supported_instrument_types: tuple[str, ...] = ()
    supported_resolutions: tuple[str, ...] = ("1d",)
    references: tuple[str, ...] = ()
    validation_status: str = ValidationStatus.UNIT_TESTED
    cost_class: str = "cheap"  # cheap | moderate | expensive
    params_schema: dict[str, Any] = field(default_factory=dict)

    @property
    def qualified_key(self) -> str:
        return f"{self.key}@{self.version}"


@dataclass
class MethodInput:
    """Everything a method may read.

    Deliberately a closed set. A method cannot reach into the database or call
    a provider, which is what makes it deterministic and testable: the same
    input always produces the same output.
    """

    # Chronologically ascending. Enforced by the runner, not assumed.
    timestamps: tuple[dt.datetime, ...] = ()
    open: tuple[float, ...] = ()
    high: tuple[float, ...] = ()
    low: tuple[float, ...] = ()
    close: tuple[float, ...] = ()
    volume: tuple[float | None, ...] = ()
    # Optional comparison series, already aligned to ``timestamps``.
    benchmark_close: tuple[float, ...] = ()
    # Macro or fundamental scalars keyed by concept.
    scalars: dict[str, float] = field(default_factory=dict)
    series: dict[str, tuple[float, ...]] = field(default_factory=dict)
    resolution: str = "1d"
    currency: str = "USD"
    trading_days_per_year: int = 252
    as_of: dt.datetime | None = None

    def __len__(self) -> int:
        return len(self.close)


@dataclass
class MethodResult:
    """A method's output plus the provenance needed to reproduce it."""

    method_key: str
    method_version: str
    values: dict[str, Any]
    observations_used: int
    input_fingerprint: str
    warnings: list[str] = field(default_factory=list)
    window_start: dt.datetime | None = None
    window_end: dt.datetime | None = None
    params: dict[str, Any] = field(default_factory=dict)
    limitations: tuple[str, ...] = ()

    @property
    def scalar(self) -> float | None:
        """The primary value, when a method produces exactly one."""
        if len(self.values) == 1:
            only = next(iter(self.values.values()))
            return only if isinstance(only, (int, float)) else None
        return self.values.get("value")


MethodFn = Callable[[MethodInput, dict[str, Any]], dict[str, Any]]

_REGISTRY: dict[str, tuple[MethodSpec, MethodFn]] = {}
_LOADED = False


def _ensure_loaded() -> None:
    """Import the method modules so their decorators have run.

    The registry is populated by import side effect, which means any entry
    point reached before those modules are imported sees an empty registry.
    Relying on the caller to import them first is a trap: it works in a process
    that happens to import a method module for another reason and fails in one
    that does not. Every public accessor funnels through here instead.
    """
    global _LOADED
    if _LOADED:
        return
    # Set the flag first: the method modules import this one, so a re-entrant
    # call during their import must not recurse.
    _LOADED = True
    from app.analysis import (  # noqa: F401
        bubble,
        fundamental,
        regime,
        risk,
        statistics,
        technical,
    )


def register_method(spec: MethodSpec) -> Callable[[MethodFn], MethodFn]:
    """Decorator registering a method implementation against its spec."""

    def decorator(fn: MethodFn) -> MethodFn:
        if spec.key in _REGISTRY:
            existing_spec, _ = _REGISTRY[spec.key]
            if existing_spec.version == spec.version:
                raise ValueError(
                    f"method {spec.key!r} version {spec.version} is already registered; "
                    "a change in output for the same input requires a version bump"
                )
        _REGISTRY[spec.key] = (spec, fn)
        return fn

    return decorator


def get_spec(key: str) -> MethodSpec:
    _ensure_loaded()
    try:
        return _REGISTRY[key][0]
    except KeyError as exc:
        raise NotFoundError(
            f"unknown method {key!r}; registered: {sorted(_REGISTRY)}"
        ) from exc


def all_specs() -> list[MethodSpec]:
    _ensure_loaded()
    return sorted((spec for spec, _ in _REGISTRY.values()), key=lambda s: (s.family, s.key))


def specs_in_family(family: str) -> list[MethodSpec]:
    return [s for s in all_specs() if s.family == family]


def code_fingerprint(key: str) -> str:
    """Hash of the implementation's source, to detect drift from the recorded version."""
    _ensure_loaded()
    _, fn = _REGISTRY[key]
    try:
        source = inspect.getsource(fn)
    except OSError:  # pragma: no cover - source unavailable
        return ""
    return hashlib.sha256(source.encode()).hexdigest()[:16]


def fingerprint_input(data: MethodInput, params: dict[str, Any]) -> str:
    """Stable hash of the exact inputs a result was computed from.

    Includes the window boundaries and the full close series, so two results
    with the same fingerprint genuinely used the same data. This is what makes
    acceptance criterion "reproduce key calculations from recorded inputs"
    checkable rather than aspirational.
    """
    payload = {
        "n": len(data.close),
        "first": data.timestamps[0].isoformat() if data.timestamps else None,
        "last": data.timestamps[-1].isoformat() if data.timestamps else None,
        "close": [f"{c:.10g}" for c in data.close],
        "resolution": data.resolution,
        "params": {k: params[k] for k in sorted(params)},
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def run_method(
    key: str, data: MethodInput, params: dict[str, Any] | None = None
) -> MethodResult:
    """Run a registered method with its declared guarantees enforced."""
    _ensure_loaded()
    spec, fn = _REGISTRY.get(key, (None, None))
    if spec is None or fn is None:
        raise NotFoundError(f"unknown method {key!r}; registered: {sorted(_REGISTRY)}")

    effective = dict(spec.params_schema.get("defaults", {}))
    effective.update(params or {})

    _validate_inputs(spec, data)
    _validate_ordering(data)

    n = _usable_observations(spec, data)
    required = _required_observations(spec, effective)
    if n < required:
        raise InsufficientDataError(
            f"{spec.display_name} needs at least {required} observations for the "
            f"requested parameters but only {n} are available. "
            f"{spec.missing_data_policy}",
            details={
                "method": spec.key,
                "version": spec.version,
                "required": required,
                "available": n,
                "params": effective,
            },
        )

    warnings: list[str] = []
    values = fn(data, effective)
    if isinstance(values, tuple) and len(values) == 2:
        values, extra = values
        warnings.extend(extra)

    return MethodResult(
        method_key=spec.key,
        method_version=spec.version,
        values=values,
        observations_used=n,
        input_fingerprint=fingerprint_input(data, effective),
        warnings=warnings,
        window_start=data.timestamps[0] if data.timestamps else None,
        window_end=data.timestamps[-1] if data.timestamps else None,
        params=effective,
        limitations=spec.limitations,
    )


def _validate_inputs(spec: MethodSpec, data: MethodInput) -> None:
    """Check each declared requirement.

    A requirement written ``"a|b"`` is satisfied by either alternative. Methods
    that genuinely accept a substitute -- a diluted *or* basic share count, any
    one of several fiscal series -- must declare it that way, otherwise the
    registry rejects input the method would have handled correctly and reports a
    missing input that was never actually required.
    """
    for requirement in spec.required_inputs:
        alternatives = requirement.split("|")
        if len(alternatives) > 1:
            if not any(_has_input(data, alt) for alt in alternatives):
                raise InsufficientDataError(
                    f"{spec.display_name} requires at least one of "
                    f"{', '.join(repr(a) for a in alternatives)}, and none was supplied",
                    details={"method": spec.key, "missing_any_of": alternatives},
                )
            continue
        _require_input(spec, data, requirement)


def _has_input(data: MethodInput, name: str) -> bool:
    if name.startswith("series:"):
        return name.split(":", 1)[1] in data.series
    if name.startswith("scalar:"):
        return data.scalars.get(name.split(":", 1)[1]) is not None
    return bool(getattr(data, name, ()))


def _require_input(spec: MethodSpec, data: MethodInput, name: str) -> None:
    if name in {"open", "high", "low", "close", "volume", "benchmark_close"}:
        series = getattr(data, name, ())
        if not series:
            raise InsufficientDataError(
                f"{spec.display_name} requires the {name!r} series, which is empty",
                details={"method": spec.key, "missing_input": name},
            )
        if name != "volume" and len(series) != len(data.close):
            raise InsufficientDataError(
                f"{spec.display_name} requires {name!r} aligned to close "
                f"({len(series)} vs {len(data.close)})",
                details={"method": spec.key, "misaligned_input": name},
            )
    elif name.startswith("series:"):
        wanted = name.split(":", 1)[1]
        if wanted not in data.series:
            raise InsufficientDataError(
                f"{spec.display_name} requires the auxiliary series {wanted!r}",
                details={"method": spec.key, "missing_series": wanted},
            )
    elif name.startswith("scalar:"):
        wanted = name.split(":", 1)[1]
        if wanted not in data.scalars:
            raise InsufficientDataError(
                f"{spec.display_name} requires the scalar {wanted!r}",
                details={"method": spec.key, "missing_scalar": wanted},
            )


def _validate_ordering(data: MethodInput) -> None:
    """Reject out-of-order timestamps.

    Every method here assumes ascending time. A silently unsorted series makes
    a return calculation produce confident nonsense, so it is checked rather
    than trusted.
    """
    ts = data.timestamps
    for i in range(1, len(ts)):
        if ts[i] <= ts[i - 1]:
            raise InsufficientDataError(
                f"timestamps are not strictly ascending at index {i}: "
                f"{ts[i - 1].isoformat()} then {ts[i].isoformat()}",
                details={"index": i},
            )


def _usable_observations(spec: MethodSpec, data: MethodInput) -> int:
    """Count observations the method can actually use.

    A None inside a required series truncates usability rather than being
    skipped, because skipping would silently change the spacing of a time
    series and corrupt any windowed statistic.
    """
    if "volume" in spec.required_inputs:
        usable = 0
        for v in data.volume:
            if v is None:
                break
            usable += 1
        return min(usable, len(data.close))
    return len(data.close)


def _required_observations(spec: MethodSpec, params: dict[str, Any]) -> int:
    """Observations this method needs *for these parameters*.

    ``min_observations`` is the requirement at the spec's default parameters,
    not an absolute floor. A method whose window is parameterised declares
    ``observation_drivers``, and the requirement is then derived from the
    parameters actually supplied: a caller asking for a 50-period RSI must be
    refused a 30-bar series, and a caller asking for a 2-period RSI must not be
    refused a 5-bar one. Treating the declared minimum as a floor in both
    directions would do the second, rejecting a computation that is perfectly
    well defined.

    ``absolute_min`` exists for the separate case of a genuine statistical
    floor below which a result would be meaningless whatever the window.
    """
    drivers = spec.params_schema.get("observation_drivers", ())
    padding = spec.params_schema.get("window_padding", 0)
    absolute_min = spec.params_schema.get("absolute_min", 2)

    driven = 0
    for name in drivers:
        value = params.get(name)
        if isinstance(value, int):
            driven = max(driven, value)

    if not driven:
        return spec.min_observations
    return max(driven + padding, absolute_min)


def registry_snapshot() -> list[dict]:
    """Serialisable view of the registry, for the ``/methods`` endpoint."""
    return [
        {
            "key": s.key,
            "version": s.version,
            "family": s.family,
            "display_name": s.display_name,
            "description": s.description,
            "purpose": s.purpose,
            "required_inputs": list(s.required_inputs),
            "supported_resolutions": list(s.supported_resolutions),
            "supported_instrument_types": list(s.supported_instrument_types),
            "min_observations": s.min_observations,
            "output_schema": s.output_schema,
            "missing_data_policy": s.missing_data_policy,
            "limitations": list(s.limitations),
            "references": list(s.references),
            "validation_status": s.validation_status,
            "cost_class": s.cost_class,
            "params_schema": s.params_schema,
            "code_fingerprint": code_fingerprint(s.key),
        }
        for s in all_specs()
    ]


def sync_registry_to_db(session) -> int:
    """Mirror the in-code registry into ``method_definitions``.

    The code is authoritative at runtime; the table is the durable record so a
    feature row from months ago can still name the method that produced it.
    """
    from sqlalchemy import select

    from app.db.models.research import MethodDefinition

    written = 0
    for spec in all_specs():
        existing = session.execute(
            select(MethodDefinition).where(
                MethodDefinition.key == spec.key,
                MethodDefinition.version == spec.version,
            )
        ).scalars().first()
        fields = {
            "family": spec.family,
            "display_name": spec.display_name,
            "description": spec.description,
            "purpose": spec.purpose,
            "required_inputs": list(spec.required_inputs),
            "supported_instrument_types": list(spec.supported_instrument_types),
            "supported_resolutions": list(spec.supported_resolutions),
            "min_observations": spec.min_observations,
            "output_schema": spec.output_schema,
            "missing_data_policy": spec.missing_data_policy,
            "limitations": list(spec.limitations),
            "references": list(spec.references),
            "validation_status": spec.validation_status,
            "cost_class": spec.cost_class,
            "code_fingerprint": code_fingerprint(spec.key),
        }
        if existing is None:
            session.add(MethodDefinition(key=spec.key, version=spec.version, **fields))
            written += 1
        else:
            for k, v in fields.items():
                setattr(existing, k, v)
    session.flush()
    return written


def load_all_methods() -> None:
    """Explicitly populate the registry. Accessors do this themselves."""
    _ensure_loaded()
