"""Replicate, distribution, and subject-level evaluation for the temporal lab."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import sqrt
from statistics import fmean

from umwelt.errors import DataError
from umwelt.observations import (
    BehavioralState,
    ObservationDataset,
    ObservationSource,
    TimeSeries,
)


def quantile(values: Sequence[float], probability: float) -> float | None:
    """Return a deterministic linearly interpolated empirical quantile."""

    if not 0 <= probability <= 1:
        raise DataError("Quantile probability must be between zero and one.")
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return float(ordered[lower] + fraction * (ordered[upper] - ordered[lower]))


def empirical_distribution_distances(
    first: Sequence[float], second: Sequence[float]
) -> dict[str, float | None]:
    """Return exact empirical 1D Wasserstein and two-sample KS distances."""

    if not first or not second:
        return {"wasserstein": None, "kolmogorov_smirnov": None}
    left = sorted(float(value) for value in first)
    right = sorted(float(value) for value in second)
    i = 0
    j = 0
    cdf_left = 0.0
    cdf_right = 0.0
    wasserstein = 0.0
    ks = 0.0
    position = min(left[0], right[0])

    while i < len(left) or j < len(right):
        while i < len(left) and left[i] == position:
            i += 1
        while j < len(right) and right[j] == position:
            j += 1
        cdf_left = i / len(left)
        cdf_right = j / len(right)
        ks = max(ks, abs(cdf_left - cdf_right))
        candidates = []
        if i < len(left):
            candidates.append(left[i])
        if j < len(right):
            candidates.append(right[j])
        if not candidates:
            break
        next_position = min(candidates)
        wasserstein += abs(cdf_left - cdf_right) * (next_position - position)
        position = next_position

    return {
        "wasserstein": wasserstein,
        "kolmogorov_smirnov": ks,
    }


def _profile_rmse(
    recorded: Sequence[float | None], synthetic: Sequence[float | None]
) -> float | None:
    squared = [
        (float(synthetic_value) - float(recorded_value)) ** 2
        for recorded_value, synthetic_value in zip(recorded, synthetic, strict=True)
        if recorded_value is not None and synthetic_value is not None
    ]
    return sqrt(fmean(squared)) if squared else None


@dataclass(slots=True)
class _Accumulator:
    observed: int
    sleep: int
    activity_sum: float
    nonzero_activity: int
    transition_exposure: int
    state_changes: int
    phase_observations: list[int]
    phase_sleep: list[int]
    phase_activity: list[float]
    bouts: dict[str, list[float]]
    autocorrelation: dict[int, list[float]]


def _empty_accumulator(
    phase_bins: int, autocorrelation_lags: tuple[int, ...]
) -> _Accumulator:
    return _Accumulator(
        observed=0,
        sleep=0,
        activity_sum=0.0,
        nonzero_activity=0,
        transition_exposure=0,
        state_changes=0,
        phase_observations=[0] * phase_bins,
        phase_sleep=[0] * phase_bins,
        phase_activity=[0.0] * phase_bins,
        bouts={"wake": [], "sleep": []},
        autocorrelation={lag: [0.0] * 6 for lag in autocorrelation_lags},
    )


def _phase(timestamp, phase_bins: int) -> int:
    seconds = (
        timestamp.hour * 3_600
        + timestamp.minute * 60
        + timestamp.second
        + timestamp.microsecond / 1_000_000
    )
    return min(int(seconds * phase_bins / 86_400), phase_bins - 1)


def _regular_epoch_seconds(dataset: ObservationDataset) -> int:
    expected: int | None = None
    for series in dataset.series:
        for previous, current in zip(series.timestamps, series.timestamps[1:]):
            delta = (current - previous).total_seconds()
            if delta <= 0 or not delta.is_integer():
                raise DataError(
                    "Temporal evaluation requires positive whole-second epochs."
                )
            if expected is None:
                expected = int(delta)
            elif delta != expected:
                raise DataError("Temporal evaluation requires one regular time grid.")
    if expected is None:
        raise DataError("Temporal evaluation requires at least two timestamps.")
    return expected


def _accumulate_series(
    series: TimeSeries,
    *,
    phase_bins: int,
    autocorrelation_lags: tuple[int, ...],
    epoch_seconds: int,
) -> _Accumulator:
    accumulator = _empty_accumulator(phase_bins, autocorrelation_lags)
    previous_state: BehavioralState | None = None
    bout_state: BehavioralState | None = None
    bout_epochs = 0

    def finish_bout() -> None:
        nonlocal bout_state, bout_epochs
        if bout_state is not None and bout_epochs:
            accumulator.bouts[bout_state.value].append(
                float(bout_epochs * epoch_seconds)
            )
        bout_state = None
        bout_epochs = 0

    for index, (timestamp, state, activity) in enumerate(
        zip(series.timestamps, series.states, series.activities, strict=True)
    ):
        if state is None or activity is None:
            finish_bout()
            previous_state = None
            continue
        accumulator.observed += 1
        accumulator.sleep += state is BehavioralState.SLEEP
        accumulator.activity_sum += activity
        accumulator.nonzero_activity += activity > 0
        phase = _phase(timestamp, phase_bins)
        accumulator.phase_observations[phase] += 1
        accumulator.phase_sleep[phase] += state is BehavioralState.SLEEP
        accumulator.phase_activity[phase] += activity

        if previous_state is not None:
            accumulator.transition_exposure += 1
            accumulator.state_changes += state is not previous_state
        previous_state = state

        if bout_state is None:
            bout_state = state
            bout_epochs = 1
        elif state is bout_state:
            bout_epochs += 1
        else:
            finish_bout()
            bout_state = state
            bout_epochs = 1

        y = 1.0 if state is BehavioralState.SLEEP else 0.0
        for lag in autocorrelation_lags:
            if index < lag:
                continue
            prior_state = series.states[index - lag]
            if prior_state is None:
                continue
            x = 1.0 if prior_state is BehavioralState.SLEEP else 0.0
            values = accumulator.autocorrelation[lag]
            values[0] += 1
            values[1] += x
            values[2] += y
            values[3] += x * x
            values[4] += y * y
            values[5] += x * y
    finish_bout()
    return accumulator


def _merge(target: _Accumulator, source: _Accumulator) -> None:
    target.observed += source.observed
    target.sleep += source.sleep
    target.activity_sum += source.activity_sum
    target.nonzero_activity += source.nonzero_activity
    target.transition_exposure += source.transition_exposure
    target.state_changes += source.state_changes
    for index in range(len(target.phase_observations)):
        target.phase_observations[index] += source.phase_observations[index]
        target.phase_sleep[index] += source.phase_sleep[index]
        target.phase_activity[index] += source.phase_activity[index]
    for state in ("wake", "sleep"):
        target.bouts[state].extend(source.bouts[state])
    for lag, values in target.autocorrelation.items():
        for index, value in enumerate(source.autocorrelation[lag]):
            values[index] += value


def _bout_summary(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "q90": None}
    ordered = sorted(values)
    return {
        "mean": fmean(values),
        "median": quantile(ordered, 0.5),
        "q90": quantile(ordered, 0.9),
    }


def _snapshot(accumulator: _Accumulator, epoch_seconds: int) -> dict[str, object]:
    if not accumulator.observed:
        raise DataError("Temporal evaluation requires present observations.")
    scalars: dict[str, float | None] = {
        "sleep_fraction": accumulator.sleep / accumulator.observed,
        "wake_fraction": 1.0 - accumulator.sleep / accumulator.observed,
        "mean_activity": accumulator.activity_sum / accumulator.observed,
        "nonzero_activity_fraction": accumulator.nonzero_activity
        / accumulator.observed,
        "state_changes_per_hour": (
            accumulator.state_changes
            * 3_600
            / (accumulator.transition_exposure * epoch_seconds)
            if accumulator.transition_exposure
            else None
        ),
    }
    for state in ("wake", "sleep"):
        for statistic, value in _bout_summary(accumulator.bouts[state]).items():
            scalars[f"{state}_bout_{statistic}_seconds"] = value
    for lag, values in accumulator.autocorrelation.items():
        count, sum_x, sum_y, sum_x2, sum_y2, sum_xy = values
        covariance = count * sum_xy - sum_x * sum_y
        variance_x = count * sum_x2 - sum_x * sum_x
        variance_y = count * sum_y2 - sum_y * sum_y
        denominator = sqrt(variance_x * variance_y)
        scalars[f"sleep_autocorrelation_lag_{lag}"] = (
            covariance / denominator if count >= 2 and denominator else None
        )
    return {
        "scalars": scalars,
        "phase_profile": {
            "sleep_fraction": [
                accumulator.phase_sleep[index] / count if count else None
                for index, count in enumerate(accumulator.phase_observations)
            ],
            "mean_activity": [
                accumulator.phase_activity[index] / count if count else None
                for index, count in enumerate(accumulator.phase_observations)
            ],
        },
    }


@dataclass(frozen=True, slots=True)
class EvaluationUnit:
    """Metrics and transient bout samples for one recorded comparison unit."""

    snapshot: dict[str, object]
    bouts: dict[str, list[float]]


@dataclass(frozen=True, slots=True)
class RecordedReference:
    """Precomputed recorded summaries, kept separate from synthetic outputs."""

    aggregate: EvaluationUnit
    subjects: dict[str, EvaluationUnit]
    phase_bins: int
    autocorrelation_lags: tuple[int, ...]
    epoch_seconds: int

    def to_dict(self) -> dict[str, object]:
        """Return recorded derived metrics without retaining raw bout samples."""

        return {
            "schema_version": "umwelt.temporal-recorded-reference.v1",
            "source_category": ObservationSource.RECORDED.value,
            "phase_bins": self.phase_bins,
            "autocorrelation_lags": list(self.autocorrelation_lags),
            "epoch_seconds": self.epoch_seconds,
            "aggregate": self.aggregate.snapshot,
            "subjects": {
                subject_id: unit.snapshot for subject_id, unit in self.subjects.items()
            },
        }


def build_recorded_reference(
    dataset: ObservationDataset,
    *,
    phase_bins: int,
    autocorrelation_lags: Iterable[int],
) -> RecordedReference:
    """Precompute aggregate and individual metrics for recorded subjects."""

    if any(
        series.source is not ObservationSource.RECORDED for series in dataset.series
    ):
        raise DataError("A recorded reference may contain only recorded observations.")
    if not dataset.series:
        raise DataError("A recorded reference requires at least one subject.")
    if not 1 <= phase_bins <= 1_440:
        raise DataError("Temporal evaluation phase bins must be between 1 and 1440.")
    lags = tuple(autocorrelation_lags)
    if not lags or any(lag <= 0 for lag in lags) or len(lags) != len(set(lags)):
        raise DataError("Temporal autocorrelation lags must be positive and unique.")
    epoch_seconds = _regular_epoch_seconds(dataset)
    aggregate = _empty_accumulator(phase_bins, lags)
    subjects = {}
    for series in dataset.series:
        accumulator = _accumulate_series(
            series,
            phase_bins=phase_bins,
            autocorrelation_lags=lags,
            epoch_seconds=epoch_seconds,
        )
        _merge(aggregate, accumulator)
        subjects[series.subject_id] = EvaluationUnit(
            snapshot=_snapshot(accumulator, epoch_seconds),
            bouts=accumulator.bouts,
        )
    return RecordedReference(
        aggregate=EvaluationUnit(
            snapshot=_snapshot(aggregate, epoch_seconds), bouts=aggregate.bouts
        ),
        subjects=subjects,
        phase_bins=phase_bins,
        autocorrelation_lags=lags,
        epoch_seconds=epoch_seconds,
    )


def _compare_unit(
    recorded: EvaluationUnit,
    synthetic_snapshot: Mapping[str, object],
    synthetic_bouts: Mapping[str, Sequence[float]],
) -> dict[str, object]:
    discrepancies = {}
    for state in ("wake", "sleep"):
        distances = empirical_distribution_distances(
            recorded.bouts[state], synthetic_bouts[state]
        )
        discrepancies[f"{state}_bout_wasserstein_seconds"] = distances["wasserstein"]
        discrepancies[f"{state}_bout_ks"] = distances["kolmogorov_smirnov"]
    for variable in ("sleep_fraction", "mean_activity"):
        label = "sleep" if variable == "sleep_fraction" else "activity"
        discrepancies[f"{label}_phase_rmse"] = _profile_rmse(
            recorded.snapshot["phase_profile"][variable],
            synthetic_snapshot["phase_profile"][variable],
        )
    return {"metrics": synthetic_snapshot, "discrepancies": discrepancies}


def evaluate_synthetic_replicate(
    dataset: ObservationDataset,
    recorded: RecordedReference,
) -> dict[str, object]:
    """Evaluate one synthetic replicate globally and by source subject."""

    if any(
        series.source is not ObservationSource.SYNTHETIC for series in dataset.series
    ):
        raise DataError(
            "Synthetic replicate evaluation requires synthetic observations."
        )
    subject_ids = {series.subject_id for series in dataset.series}
    if subject_ids != set(recorded.subjects):
        raise DataError(
            "Synthetic replicate subjects must exactly match recorded reference subjects."
        )
    aggregate_accumulator = _empty_accumulator(
        recorded.phase_bins, recorded.autocorrelation_lags
    )
    subjects = {}
    for series in dataset.series:
        accumulator = _accumulate_series(
            series,
            phase_bins=recorded.phase_bins,
            autocorrelation_lags=recorded.autocorrelation_lags,
            epoch_seconds=recorded.epoch_seconds,
        )
        _merge(aggregate_accumulator, accumulator)
        subjects[series.subject_id] = _compare_unit(
            recorded.subjects[series.subject_id],
            _snapshot(accumulator, recorded.epoch_seconds),
            accumulator.bouts,
        )
    aggregate = _compare_unit(
        recorded.aggregate,
        _snapshot(aggregate_accumulator, recorded.epoch_seconds),
        aggregate_accumulator.bouts,
    )
    return {"aggregate": aggregate, "subjects": subjects}


def predictive_summary(
    values: Sequence[float],
    reference: float | None,
    interval: tuple[float, float],
) -> dict[str, object]:
    """Summarize a synthetic replicate distribution without inferential language."""

    if not values:
        return {
            "replicates": 0,
            "reference": reference,
            "median": None,
            "lower": None,
            "upper": None,
            "reference_within_interval": None,
        }
    lower = quantile(values, interval[0])
    median = quantile(values, 0.5)
    upper = quantile(values, interval[1])
    inside = (
        lower <= reference <= upper
        if reference is not None and lower is not None and upper is not None
        else None
    )
    return {
        "replicates": len(values),
        "reference": reference,
        "median": median,
        "lower": lower,
        "upper": upper,
        "reference_within_interval": inside,
    }


def _summarize_scope(
    scopes: Sequence[Mapping[str, object]],
    recorded_snapshot: Mapping[str, object],
    interval: tuple[float, float],
) -> dict[str, object]:
    metric_summaries = {}
    for name, reference in recorded_snapshot["scalars"].items():
        values = [
            float(scope["metrics"]["scalars"][name])
            for scope in scopes
            if scope["metrics"]["scalars"][name] is not None
        ]
        metric_summaries[name] = predictive_summary(values, reference, interval)

    discrepancy_summaries = {}
    for name in scopes[0]["discrepancies"]:
        values = [
            float(scope["discrepancies"][name])
            for scope in scopes
            if scope["discrepancies"][name] is not None
        ]
        discrepancy_summaries[name] = predictive_summary(values, 0.0, interval)

    profiles = {}
    for variable in ("sleep_fraction", "mean_activity"):
        phases = []
        for phase, reference in enumerate(recorded_snapshot["phase_profile"][variable]):
            values = [
                float(scope["metrics"]["phase_profile"][variable][phase])
                for scope in scopes
                if scope["metrics"]["phase_profile"][variable][phase] is not None
            ]
            phases.append(
                {"phase": phase, **predictive_summary(values, reference, interval)}
            )
        profiles[variable] = phases
    return {
        "metrics": metric_summaries,
        "discrepancies": discrepancy_summaries,
        "phase_profiles": profiles,
    }


def summarize_model_replicates(
    records: Sequence[Mapping[str, object]],
    recorded: RecordedReference,
    *,
    interval: tuple[float, float],
) -> dict[str, object]:
    """Aggregate replicate records into global and subject predictive summaries."""

    if not records:
        raise DataError("Model summaries require at least one replicate record.")
    aggregate = _summarize_scope(
        [record["evaluation"]["aggregate"] for record in records],
        recorded.aggregate.snapshot,
        interval,
    )
    subjects = {}
    for subject_id, reference in recorded.subjects.items():
        subjects[subject_id] = _summarize_scope(
            [record["evaluation"]["subjects"][subject_id] for record in records],
            reference.snapshot,
            interval,
        )
    return {"aggregate": aggregate, "subjects": subjects}
