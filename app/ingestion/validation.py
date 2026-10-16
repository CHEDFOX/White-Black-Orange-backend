"""Market-data validation.

Validation runs *before* storage and its findings are kept, not discarded. The
reason is the product promise: a report must be able to say "this window
contains two sessions with no data and one bar whose high is below its low"
rather than drawing a clean chart over broken input.

Three rules govern every validator here:

1. **Never repair by invention.** A missing price is missing. Nothing is
   forward-filled, interpolated or zeroed.
2. **Separate "absent" from "zero".** No reported volume and zero shares traded
   are different facts and are stored differently (NULL vs 0).
3. **Reject, don't coerce.** A bar that fails a structural check is excluded
   from the stored series and recorded as an issue, so the series stays sound
   and the defect stays visible.
"""

from __future__ import annotations

import datetime as dt
import decimal
import math
from collections import Counter
from dataclasses import dataclass, field

from app.core.timeutil import TradingCalendar, resolution_seconds
from app.db.models.enums import DataQualityKind, DataQualitySeverity
from app.providers.base import ProviderBar

# A single-session move beyond this is flagged for review. It is deliberately
# generous: real instruments do move 50% in a day on buyout news, so this
# catches unadjusted splits and decimal errors, not volatility.
DEFAULT_OUTLIER_LOG_RETURN = 0.40


@dataclass(frozen=True, slots=True)
class ValidationFinding:
    kind: str
    severity: str
    detail: str
    event_time: dt.datetime | None = None
    window_start: dt.datetime | None = None
    window_end: dt.datetime | None = None
    context: dict = field(default_factory=dict)

    @property
    def is_fatal(self) -> bool:
        """Whether the affected bar must be excluded from the stored series."""
        return self.severity == DataQualitySeverity.ERROR


@dataclass
class ValidationResult:
    accepted: list[ProviderBar] = field(default_factory=list)
    rejected: list[tuple[ProviderBar, ValidationFinding]] = field(default_factory=list)
    findings: list[ValidationFinding] = field(default_factory=list)

    @property
    def accepted_count(self) -> int:
        return len(self.accepted)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)

    def findings_by_kind(self) -> dict[str, int]:
        return dict(Counter(f.kind for f in self.findings))

    def errors(self) -> list[ValidationFinding]:
        return [f for f in self.findings if f.severity == DataQualitySeverity.ERROR]

    def warnings(self) -> list[ValidationFinding]:
        return [f for f in self.findings if f.severity == DataQualitySeverity.WARNING]


def check_bar_structure(bar: ProviderBar) -> ValidationFinding | None:
    """Structural checks on a single bar. A failure excludes the bar."""
    prices = {"open": bar.open, "high": bar.high, "low": bar.low, "close": bar.close}

    for name, value in prices.items():
        if value is None:
            return ValidationFinding(
                kind=DataQualityKind.OHLC_INCONSISTENT,
                severity=DataQualitySeverity.ERROR,
                detail=f"{name} is missing",
                event_time=bar.event_time,
            )
        if not value.is_finite():
            return ValidationFinding(
                kind=DataQualityKind.OHLC_INCONSISTENT,
                severity=DataQualitySeverity.ERROR,
                detail=f"{name} is not finite ({value})",
                event_time=bar.event_time,
            )
        if value <= 0:
            return ValidationFinding(
                kind=DataQualityKind.NEGATIVE_PRICE,
                severity=DataQualitySeverity.ERROR,
                detail=f"{name} is not positive ({value})",
                event_time=bar.event_time,
                context={name: str(value)},
            )

    if bar.high < bar.low:
        return ValidationFinding(
            kind=DataQualityKind.OHLC_INCONSISTENT,
            severity=DataQualitySeverity.ERROR,
            detail=f"high {bar.high} is below low {bar.low}",
            event_time=bar.event_time,
            context={"high": str(bar.high), "low": str(bar.low)},
        )

    # The open and close must lie inside the day's range, by definition of a
    # bar. A violation means the fields were mismatched upstream.
    for name in ("open", "close"):
        value = prices[name]
        if value > bar.high or value < bar.low:
            return ValidationFinding(
                kind=DataQualityKind.OHLC_INCONSISTENT,
                severity=DataQualitySeverity.ERROR,
                detail=(
                    f"{name} {value} lies outside the bar range "
                    f"[{bar.low}, {bar.high}]"
                ),
                event_time=bar.event_time,
                context={name: str(value), "low": str(bar.low), "high": str(bar.high)},
            )

    if bar.volume is not None and bar.volume < 0:
        return ValidationFinding(
            kind=DataQualityKind.OHLC_INCONSISTENT,
            severity=DataQualitySeverity.ERROR,
            detail=f"volume is negative ({bar.volume})",
            event_time=bar.event_time,
        )

    if bar.bar_close_time <= bar.event_time:
        return ValidationFinding(
            kind=DataQualityKind.TIMESTAMP_MISALIGNED,
            severity=DataQualitySeverity.ERROR,
            detail=(
                f"bar_close_time {bar.bar_close_time.isoformat()} is not after "
                f"event_time {bar.event_time.isoformat()}"
            ),
            event_time=bar.event_time,
        )

    return None


def check_zero_volume(bar: ProviderBar) -> ValidationFinding | None:
    """Flag a zero-volume bar that nonetheless shows price movement.

    Zero volume with a flat bar is a legitimate halted or illiquid session.
    Zero volume with a price range is contradictory and worth surfacing.
    """
    if bar.volume is None or bar.volume != 0:
        return None
    if bar.high == bar.low:
        return ValidationFinding(
            kind=DataQualityKind.ZERO_VOLUME,
            severity=DataQualitySeverity.INFO,
            detail="zero volume with a flat bar: likely a halt or an illiquid session",
            event_time=bar.event_time,
        )
    return ValidationFinding(
        kind=DataQualityKind.ZERO_VOLUME,
        severity=DataQualitySeverity.WARNING,
        detail=(
            f"zero volume but the bar has a range of {bar.high - bar.low}; "
            "price moved with no reported trading"
        ),
        event_time=bar.event_time,
        context={"range": str(bar.high - bar.low)},
    )


def find_duplicate_timestamps(bars: list[ProviderBar]) -> list[ValidationFinding]:
    """Detect repeated event times, distinguishing identical from conflicting.

    An exact repeat is a harmless retransmission. Two different bars claiming
    the same instant is a real conflict that must not be silently resolved by
    insertion order.
    """
    seen: dict[dt.datetime, ProviderBar] = {}
    findings: list[ValidationFinding] = []
    for bar in bars:
        prior = seen.get(bar.event_time)
        if prior is None:
            seen[bar.event_time] = bar
            continue
        identical = (
            prior.open == bar.open
            and prior.high == bar.high
            and prior.low == bar.low
            and prior.close == bar.close
            and prior.volume == bar.volume
        )
        findings.append(
            ValidationFinding(
                kind=DataQualityKind.DUPLICATE_TIMESTAMP,
                severity=(
                    DataQualitySeverity.INFO if identical else DataQualitySeverity.ERROR
                ),
                detail=(
                    "exact duplicate bar received twice"
                    if identical
                    else (
                        "two different bars claim the same timestamp: "
                        f"close {prior.close} vs {bar.close}"
                    )
                ),
                event_time=bar.event_time,
                context={"identical": identical},
            )
        )
    return findings


def find_outlier_returns(
    bars: list[ProviderBar], *, threshold: float = DEFAULT_OUTLIER_LOG_RETURN
) -> list[ValidationFinding]:
    """Flag implausible close-to-close moves.

    The dominant cause of a large spurious move is an unadjusted split, so the
    finding reports the implied ratio: a return near ``ln(1/2)`` on a 2:1 split
    is diagnostic, and the message says so rather than just "outlier".
    """
    findings: list[ValidationFinding] = []
    ordered = sorted(bars, key=lambda b: b.event_time)
    for prev, cur in zip(ordered, ordered[1:], strict=False):
        if prev.close <= 0 or cur.close <= 0:
            continue
        ratio = float(cur.close / prev.close)
        log_return = math.log(ratio)
        if abs(log_return) < threshold:
            continue
        context = {
            "log_return": round(log_return, 6),
            "ratio": round(ratio, 6),
            "previous_close": str(prev.close),
            "close": str(cur.close),
        }
        hint = _split_hint(ratio)
        findings.append(
            ValidationFinding(
                kind=DataQualityKind.OUTLIER_RETURN,
                severity=DataQualitySeverity.WARNING,
                detail=(
                    f"close-to-close log return of {log_return:+.4f} "
                    f"({prev.close} -> {cur.close})" + (f". {hint}" if hint else "")
                ),
                event_time=cur.event_time,
                context=context,
            )
        )
    return findings


def _split_hint(ratio: float) -> str | None:
    """Name the split a suspicious ratio looks like, if any."""
    candidates = {
        "2:1": 0.5, "3:1": 1 / 3, "4:1": 0.25, "5:1": 0.2, "7:1": 1 / 7,
        "10:1": 0.1, "20:1": 0.05, "1:2 reverse": 2.0, "1:5 reverse": 5.0,
        "1:10 reverse": 10.0,
    }
    for label, expected in candidates.items():
        if abs(ratio - expected) / expected < 0.03:
            return (
                f"This is close to a {label} split ratio; the series may be "
                "unadjusted across a corporate action rather than genuinely volatile"
            )
    return None


def check_timestamp_alignment(
    bars: list[ProviderBar], resolution: str, calendar: TradingCalendar
) -> list[ValidationFinding]:
    """Verify bar timestamps sit on a sensible grid for the resolution."""
    findings: list[ValidationFinding] = []
    if resolution in {"1wk", "1mo"}:
        return findings

    step = resolution_seconds(resolution)
    for bar in bars:
        if resolution == "1d":
            local_date = bar.event_time.astimezone(calendar.tz).date()
            session = calendar.session(local_date)
            if session is None:
                findings.append(
                    ValidationFinding(
                        kind=DataQualityKind.TIMESTAMP_MISALIGNED,
                        severity=DataQualitySeverity.WARNING,
                        detail=(
                            f"a daily bar is dated {local_date}, which the "
                            f"{calendar.name} calendar treats as non-trading"
                        ),
                        event_time=bar.event_time,
                        context={"calendar": calendar.name,
                                 "calendar_coverage": calendar.coverage_note},
                    )
                )
            elif bar.event_time != session.open_utc:
                findings.append(
                    ValidationFinding(
                        kind=DataQualityKind.TIMESTAMP_MISALIGNED,
                        severity=DataQualitySeverity.INFO,
                        detail=(
                            f"daily bar timestamp {bar.event_time.isoformat()} does not "
                            f"equal the session open {session.open_utc.isoformat()}"
                        ),
                        event_time=bar.event_time,
                    )
                )
            continue

        # Intraday: the open must land on a multiple of the interval.
        if int(bar.event_time.timestamp()) % step != 0:
            findings.append(
                ValidationFinding(
                    kind=DataQualityKind.TIMESTAMP_MISALIGNED,
                    severity=DataQualitySeverity.WARNING,
                    detail=(
                        f"intraday bar at {bar.event_time.isoformat()} is not aligned "
                        f"to the {resolution} grid"
                    ),
                    event_time=bar.event_time,
                )
            )
    return findings


@dataclass(frozen=True, slots=True)
class SessionGap:
    gap_start: dt.datetime
    gap_end: dt.datetime
    expected_sessions: int
    missing_dates: tuple[dt.date, ...]


def expected_sessions_in_window(
    calendar: TradingCalendar, start: dt.datetime, end: dt.datetime
) -> list[dt.date]:
    """Sessions whose opening instant falls within ``[start, end]``.

    Selection is by absolute UTC interval, deliberately not by converting the
    window boundaries to local dates. Converting is the tempting shortcut and
    it is wrong: midnight UTC on 2023-05-10 is 20:00 on 2023-05-09 in New York,
    so a date-based window silently pulls in the previous session and reports a
    phantom gap for a day the caller never asked about.
    """
    # Widen the scan by a day on each side so a session whose local date sits
    # outside the UTC window is still considered, then filter on the instant.
    scan_start = (start - dt.timedelta(days=1)).astimezone(calendar.tz).date()
    scan_end = (end + dt.timedelta(days=1)).astimezone(calendar.tz).date()
    return [
        session.date
        for session in calendar.sessions_between(scan_start, scan_end)
        if start <= session.open_utc <= end
    ]


def find_session_gaps(
    bars: list[ProviderBar],
    calendar: TradingCalendar,
    *,
    start: dt.datetime,
    end: dt.datetime,
) -> list[SessionGap]:
    """Find trading sessions in [start, end] with no daily bar.

    Consecutive missing sessions are merged into one gap, because "2023-03-15
    through 2023-03-17 are missing" is actionable where three separate findings
    are noise.
    """
    expected = expected_sessions_in_window(calendar, start, end)
    if not expected:
        return []
    have = {b.event_time.astimezone(calendar.tz).date() for b in bars}
    missing = [d for d in expected if d not in have]
    if not missing:
        return []

    gaps: list[SessionGap] = []
    run: list[dt.date] = [missing[0]]
    expected_index = {d: i for i, d in enumerate(expected)}
    for day in missing[1:]:
        if expected_index[day] == expected_index[run[-1]] + 1:
            run.append(day)
        else:
            gaps.append(_gap_from_run(run, calendar))
            run = [day]
    gaps.append(_gap_from_run(run, calendar))
    return gaps


def _gap_from_run(run: list[dt.date], calendar: TradingCalendar) -> SessionGap:
    first = calendar.session(run[0])
    last = calendar.session(run[-1])
    start_ts = first.open_utc if first else dt.datetime.combine(
        run[0], dt.time.min, tzinfo=calendar.tz
    )
    end_ts = last.close_utc if last else dt.datetime.combine(
        run[-1], dt.time.max, tzinfo=calendar.tz
    )
    return SessionGap(
        gap_start=start_ts,
        gap_end=end_ts,
        expected_sessions=len(run),
        missing_dates=tuple(run),
    )


def validate_bars(
    bars: list[ProviderBar],
    *,
    resolution: str,
    calendar: TradingCalendar,
    outlier_threshold: float = DEFAULT_OUTLIER_LOG_RETURN,
    check_gaps_between: tuple[dt.datetime, dt.datetime] | None = None,
) -> ValidationResult:
    """Run the full validation suite over a batch of bars."""
    result = ValidationResult()

    # Pass 1: per-bar structure. Failures are excluded from the series.
    for bar in bars:
        finding = check_bar_structure(bar)
        if finding is not None:
            result.rejected.append((bar, finding))
            result.findings.append(finding)
            continue
        volume_finding = check_zero_volume(bar)
        if volume_finding is not None:
            result.findings.append(volume_finding)
        result.accepted.append(bar)

    # Pass 2: cross-bar checks, on accepted bars only -- a rejected bar must not
    # contaminate the return series used for outlier detection.
    duplicates = find_duplicate_timestamps(result.accepted)
    result.findings.extend(duplicates)
    conflicting = {
        f.event_time for f in duplicates if f.severity == DataQualitySeverity.ERROR
    }
    if conflicting:
        # Keep the first bar for each conflicting instant and reject the rest;
        # picking arbitrarily would hide the conflict.
        kept: set[dt.datetime] = set()
        survivors: list[ProviderBar] = []
        for bar in result.accepted:
            if bar.event_time in conflicting:
                if bar.event_time in kept:
                    result.rejected.append(
                        (
                            bar,
                            ValidationFinding(
                                kind=DataQualityKind.DUPLICATE_TIMESTAMP,
                                severity=DataQualitySeverity.ERROR,
                                detail=(
                                    "rejected: a different bar already claims this "
                                    "timestamp"
                                ),
                                event_time=bar.event_time,
                            ),
                        )
                    )
                    continue
                kept.add(bar.event_time)
            survivors.append(bar)
        result.accepted = survivors
    else:
        # Exact duplicates collapse to one row; the unique constraint would do
        # this anyway, and doing it here keeps the counts honest.
        deduped: dict[dt.datetime, ProviderBar] = {}
        for bar in result.accepted:
            deduped.setdefault(bar.event_time, bar)
        result.accepted = sorted(deduped.values(), key=lambda b: b.event_time)

    result.findings.extend(find_outlier_returns(result.accepted, threshold=outlier_threshold))
    result.findings.extend(
        check_timestamp_alignment(result.accepted, resolution, calendar)
    )

    if check_gaps_between is not None and resolution == "1d":
        start, end = check_gaps_between
        for gap in find_session_gaps(result.accepted, calendar, start=start, end=end):
            result.findings.append(
                ValidationFinding(
                    kind=DataQualityKind.MISSING_SESSION,
                    severity=DataQualitySeverity.WARNING,
                    detail=(
                        f"{gap.expected_sessions} expected trading session(s) have no "
                        f"bar: {', '.join(d.isoformat() for d in gap.missing_dates[:5])}"
                        + (" ..." if len(gap.missing_dates) > 5 else "")
                    ),
                    window_start=gap.gap_start,
                    window_end=gap.gap_end,
                    context={
                        "missing_dates": [d.isoformat() for d in gap.missing_dates],
                        "calendar": calendar.name,
                        "calendar_coverage": calendar.coverage_note,
                    },
                )
            )

    result.accepted.sort(key=lambda b: b.event_time)
    return result


def reconcile_bars(
    left: list[ProviderBar],
    right: list[ProviderBar],
    *,
    left_source: str,
    right_source: str,
    tolerance: decimal.Decimal = decimal.Decimal("0.005"),
) -> list[ValidationFinding]:
    """Compare two providers' bars for the same series.

    Disagreement beyond ``tolerance`` (relative) is recorded rather than
    resolved: which source is right is a question for a human or an explicit
    priority rule, not for whichever adapter happened to run second.
    """
    by_time_left = {b.event_time: b for b in left}
    findings: list[ValidationFinding] = []
    for bar in right:
        other = by_time_left.get(bar.event_time)
        if other is None:
            continue
        for field_name in ("open", "high", "low", "close"):
            a = getattr(other, field_name)
            b = getattr(bar, field_name)
            if a == 0:
                continue
            rel = abs((b - a) / a)
            if rel > tolerance:
                findings.append(
                    ValidationFinding(
                        kind=DataQualityKind.PROVIDER_DISAGREEMENT,
                        severity=DataQualitySeverity.WARNING,
                        detail=(
                            f"{field_name} differs by {rel:.2%} between "
                            f"{left_source} ({a}) and {right_source} ({b})"
                        ),
                        event_time=bar.event_time,
                        context={
                            "field": field_name,
                            left_source: str(a),
                            right_source: str(b),
                            "relative_difference": float(rel),
                        },
                    )
                )
    return findings
