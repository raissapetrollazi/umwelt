"""Quantitative descriptions and comparisons for v0.1 time series."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from math import sqrt
from statistics import fmean, pstdev

from umwelt.errors import DataError
from umwelt.observations import (
    BehavioralState,
    ObservationDataset,
    ObservationSource,
    TimeSeries,
)


def _epoch_seconds(dataset: ObservationDataset) -> int:
    intervals: Counter[int] = Counter()
    for series in dataset.series:
        for previous, current in zip(series.timestamps, series.timestamps[1:]):
            delta = (current - previous).total_seconds()
            if delta <= 0 or not delta.is_integer():
                raise DataError("Evaluation requires positive whole-second epochs.")
            intervals[int(delta)] += 1
    if not intervals or len(intervals) != 1:
        raise DataError("Evaluation requires one regular observation interval.")
    return next(iter(intervals))


def _phase(timestamp, phase_bins: int) -> int:
    seconds = (
        timestamp.hour * 3_600
        + timestamp.minute * 60
        + timestamp.second
        + timestamp.microsecond / 1_000_000
    )
    return min(int(seconds * phase_bins / 86_400), phase_bins - 1)


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _describe_values(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "standard_deviation": None,
            "minimum": None,
            "q25": None,
            "median": None,
            "q75": None,
            "q90": None,
            "maximum": None,
        }
    return {
        "count": len(values),
        "mean": fmean(values),
        "standard_deviation": pstdev(values),
        "minimum": min(values),
        "q25": _quantile(values, 0.25),
        "median": _quantile(values, 0.5),
        "q75": _quantile(values, 0.75),
        "q90": _quantile(values, 0.9),
        "maximum": max(values),
    }


def _bouts(
    series: TimeSeries, epoch_seconds: int
) -> dict[BehavioralState, list[float]]:
    result = {BehavioralState.WAKE: [], BehavioralState.SLEEP: []}
    current_state: BehavioralState | None = None
    current_epochs = 0
    previous_timestamp = None

    def finish() -> None:
        nonlocal current_state, current_epochs
        if current_state is not None and current_epochs:
            result[current_state].append(float(current_epochs * epoch_seconds))
        current_state = None
        current_epochs = 0

    for timestamp, state in zip(series.timestamps, series.states, strict=True):
        contiguous = (
            previous_timestamp is not None
            and (timestamp - previous_timestamp).total_seconds() == epoch_seconds
        )
        if state is None:
            finish()
            previous_timestamp = None
            continue
        if current_state is None or not contiguous:
            finish()
            current_state = state
            current_epochs = 1
        elif state is current_state:
            current_epochs += 1
        else:
            finish()
            current_state = state
            current_epochs = 1
        previous_timestamp = timestamp
    finish()
    return result


def _autocorrelation(
    series_collection: Iterable[TimeSeries], lag: int, epoch_seconds: int
) -> tuple[float | None, int]:
    count = 0
    sum_x = 0.0
    sum_y = 0.0
    sum_x2 = 0.0
    sum_y2 = 0.0
    sum_xy = 0.0
    expected_delta = lag * epoch_seconds

    for series in series_collection:
        for index in range(len(series.states) - lag):
            state_x = series.states[index]
            state_y = series.states[index + lag]
            if state_x is None or state_y is None:
                continue
            if (
                series.timestamps[index + lag] - series.timestamps[index]
            ).total_seconds() != expected_delta:
                continue
            x = 1.0 if state_x is BehavioralState.SLEEP else 0.0
            y = 1.0 if state_y is BehavioralState.SLEEP else 0.0
            count += 1
            sum_x += x
            sum_y += y
            sum_x2 += x * x
            sum_y2 += y * y
            sum_xy += x * y

    if count < 2:
        return None, count
    covariance = count * sum_xy - sum_x * sum_y
    variance_x = count * sum_x2 - sum_x * sum_x
    variance_y = count * sum_y2 - sum_y * sum_y
    denominator = sqrt(variance_x * variance_y)
    return (covariance / denominator if denominator else None), count


def describe_dataset(
    dataset: ObservationDataset,
    *,
    phase_bins: int = 24,
    autocorrelation_lags: Iterable[int] = (1, 6, 60, 360),
) -> dict[str, object]:
    """Compute transparent descriptive metrics for one source category."""

    if not dataset.series:
        raise DataError("Evaluation requires at least one time series.")
    if phase_bins < 1 or phase_bins > 1_440:
        raise DataError("Evaluation phase bins must be between 1 and 1440.")
    lags = tuple(autocorrelation_lags)
    if not lags or any(not isinstance(lag, int) or lag <= 0 for lag in lags):
        raise DataError("Autocorrelation lags must be positive integers.")
    if len(lags) != len(set(lags)):
        raise DataError("Autocorrelation lags must be unique.")

    sources = {series.source for series in dataset.series}
    if len(sources) != 1:
        raise DataError("Evaluation cannot combine recorded and synthetic series.")
    source = next(iter(sources))
    epoch_seconds = _epoch_seconds(dataset)

    state_counts = {BehavioralState.WAKE: 0, BehavioralState.SLEEP: 0}
    activity_values: list[float] = []
    wake_activity_values: list[float] = []
    positive_activity_values: list[float] = []
    missing_epochs = 0
    transition_exposure = 0
    transition_counts = {
        "wake_to_sleep": 0,
        "sleep_to_wake": 0,
        "wake_to_wake": 0,
        "sleep_to_sleep": 0,
    }
    all_bouts = {BehavioralState.WAKE: [], BehavioralState.SLEEP: []}
    phase_observations = [0] * phase_bins
    phase_sleep = [0] * phase_bins
    phase_activity = [0.0] * phase_bins
    subject_summaries: list[dict[str, object]] = []

    for series in dataset.series:
        subject_sleep = 0
        subject_valid = 0
        subject_activity = 0.0
        previous_state: BehavioralState | None = None
        previous_timestamp = None

        for timestamp, state, activity in zip(
            series.timestamps, series.states, series.activities, strict=True
        ):
            if state is None or activity is None:
                missing_epochs += 1
                previous_state = None
                previous_timestamp = None
                continue
            state_counts[state] += 1
            subject_valid += 1
            subject_sleep += state is BehavioralState.SLEEP
            subject_activity += activity
            activity_values.append(activity)
            if state is BehavioralState.WAKE:
                wake_activity_values.append(activity)
            if activity > 0:
                positive_activity_values.append(activity)
            phase = _phase(timestamp, phase_bins)
            phase_observations[phase] += 1
            phase_sleep[phase] += state is BehavioralState.SLEEP
            phase_activity[phase] += activity

            contiguous = (
                previous_timestamp is not None
                and (timestamp - previous_timestamp).total_seconds() == epoch_seconds
            )
            if previous_state is not None and contiguous:
                transition_exposure += 1
                key = f"{previous_state.value}_to_{state.value}"
                transition_counts[key] += 1
            previous_state = state
            previous_timestamp = timestamp

        bouts = _bouts(series, epoch_seconds)
        all_bouts[BehavioralState.WAKE].extend(bouts[BehavioralState.WAKE])
        all_bouts[BehavioralState.SLEEP].extend(bouts[BehavioralState.SLEEP])
        subject_summaries.append(
            {
                "subject_id": series.subject_id,
                "valid_epochs": subject_valid,
                "missing_epochs": len(series.states) - subject_valid,
                "sleep_fraction": subject_sleep / subject_valid
                if subject_valid
                else None,
                "mean_activity": subject_activity / subject_valid
                if subject_valid
                else None,
            }
        )

    observed_epochs = sum(state_counts.values())
    if not observed_epochs:
        raise DataError("Evaluation requires at least one present observation.")
    sleep_fraction = state_counts[BehavioralState.SLEEP] / observed_epochs
    state_changes = (
        transition_counts["wake_to_sleep"] + transition_counts["sleep_to_wake"]
    )
    subject_sleep_fractions = [
        summary["sleep_fraction"]
        for summary in subject_summaries
        if summary["sleep_fraction"] is not None
    ]

    phase_profile = []
    for phase in range(phase_bins):
        count = phase_observations[phase]
        phase_profile.append(
            {
                "phase": phase,
                "observations": count,
                "sleep_fraction": phase_sleep[phase] / count if count else None,
                "mean_activity": phase_activity[phase] / count if count else None,
            }
        )

    autocorrelation = []
    for lag in lags:
        value, pairs = _autocorrelation(dataset.series, lag, epoch_seconds)
        autocorrelation.append(
            {
                "lag_epochs": lag,
                "lag_seconds": lag * epoch_seconds,
                "pairs": pairs,
                "value": value,
            }
        )

    return {
        "schema_version": "umwelt.metrics.v1",
        "dataset_id": dataset.dataset_id,
        "source_category": source.value,
        "subject_count": len(dataset.series),
        "epoch_seconds": epoch_seconds,
        "observed_epochs": observed_epochs,
        "missing_epochs": missing_epochs,
        "state_occupancy": {
            "wake": state_counts[BehavioralState.WAKE],
            "sleep": state_counts[BehavioralState.SLEEP],
            "wake_fraction": 1.0 - sleep_fraction,
            "sleep_fraction": sleep_fraction,
        },
        "between_subject_sleep_fraction": _describe_values(subject_sleep_fractions),
        "activity": {
            "all_epochs": _describe_values(activity_values),
            "wake_epochs": _describe_values(wake_activity_values),
            "positive_epochs": _describe_values(positive_activity_values),
            "nonzero_fraction": len(positive_activity_values) / observed_epochs,
        },
        "transitions": {
            "exposure_pairs": transition_exposure,
            **transition_counts,
            "state_changes": state_changes,
            "state_changes_per_hour": (
                state_changes * 3_600 / (transition_exposure * epoch_seconds)
                if transition_exposure
                else None
            ),
            "wake_to_sleep_probability": (
                transition_counts["wake_to_sleep"]
                / (
                    transition_counts["wake_to_sleep"]
                    + transition_counts["wake_to_wake"]
                )
                if transition_counts["wake_to_sleep"]
                + transition_counts["wake_to_wake"]
                else None
            ),
            "sleep_to_wake_probability": (
                transition_counts["sleep_to_wake"]
                / (
                    transition_counts["sleep_to_wake"]
                    + transition_counts["sleep_to_sleep"]
                )
                if transition_counts["sleep_to_wake"]
                + transition_counts["sleep_to_sleep"]
                else None
            ),
        },
        "bout_durations_seconds": {
            "wake": _describe_values(all_bouts[BehavioralState.WAKE]),
            "sleep": _describe_values(all_bouts[BehavioralState.SLEEP]),
        },
        "phase_profile": phase_profile,
        "sleep_state_autocorrelation": autocorrelation,
        "subjects": subject_summaries,
    }


def _scalar_comparison(
    name: str, recorded: float | None, synthetic: float | None
) -> dict[str, object]:
    if recorded is None or synthetic is None:
        absolute = None
        relative = None
    else:
        absolute = abs(float(synthetic) - float(recorded))
        relative = absolute / abs(float(recorded)) if recorded != 0 else None
    return {
        "metric": name,
        "recorded": recorded,
        "synthetic": synthetic,
        "absolute_difference": absolute,
        "relative_difference": relative,
    }


def _profile_rmse(
    recorded: Sequence[dict[str, object]],
    synthetic: Sequence[dict[str, object]],
    key: str,
) -> float | None:
    differences = []
    for recorded_phase, synthetic_phase in zip(recorded, synthetic, strict=True):
        recorded_value = recorded_phase[key]
        synthetic_value = synthetic_phase[key]
        if recorded_value is None or synthetic_value is None:
            continue
        differences.append((float(synthetic_value) - float(recorded_value)) ** 2)
    return sqrt(fmean(differences)) if differences else None


def compare_metrics(
    recorded: dict[str, object], synthetic: dict[str, object]
) -> dict[str, object]:
    """Compare recorded and synthetic summaries without assigning pass/fail labels."""

    if recorded.get("source_category") != ObservationSource.RECORDED.value:
        raise DataError("The first comparison input must describe recorded data.")
    if synthetic.get("source_category") != ObservationSource.SYNTHETIC.value:
        raise DataError("The second comparison input must describe synthetic data.")
    if recorded.get("epoch_seconds") != synthetic.get("epoch_seconds"):
        raise DataError("Recorded and synthetic metrics use different epoch durations.")

    recorded_occupancy = recorded["state_occupancy"]
    synthetic_occupancy = synthetic["state_occupancy"]
    recorded_activity = recorded["activity"]
    synthetic_activity = synthetic["activity"]
    recorded_transitions = recorded["transitions"]
    synthetic_transitions = synthetic["transitions"]
    recorded_bouts = recorded["bout_durations_seconds"]
    synthetic_bouts = synthetic["bout_durations_seconds"]

    scalar_metrics = [
        _scalar_comparison(
            "sleep_fraction",
            recorded_occupancy["sleep_fraction"],
            synthetic_occupancy["sleep_fraction"],
        ),
        _scalar_comparison(
            "mean_activity",
            recorded_activity["all_epochs"]["mean"],
            synthetic_activity["all_epochs"]["mean"],
        ),
        _scalar_comparison(
            "nonzero_activity_fraction",
            recorded_activity["nonzero_fraction"],
            synthetic_activity["nonzero_fraction"],
        ),
        _scalar_comparison(
            "state_changes_per_hour",
            recorded_transitions["state_changes_per_hour"],
            synthetic_transitions["state_changes_per_hour"],
        ),
    ]
    for state in ("wake", "sleep"):
        for statistic in ("mean", "median", "q90"):
            scalar_metrics.append(
                _scalar_comparison(
                    f"{state}_bout_{statistic}_seconds",
                    recorded_bouts[state][statistic],
                    synthetic_bouts[state][statistic],
                )
            )

    recorded_autocorrelation = {
        item["lag_epochs"]: item for item in recorded["sleep_state_autocorrelation"]
    }
    synthetic_autocorrelation = {
        item["lag_epochs"]: item for item in synthetic["sleep_state_autocorrelation"]
    }
    common_lags = sorted(set(recorded_autocorrelation) & set(synthetic_autocorrelation))
    autocorrelation = [
        _scalar_comparison(
            f"sleep_autocorrelation_lag_{lag}",
            recorded_autocorrelation[lag]["value"],
            synthetic_autocorrelation[lag]["value"],
        )
        for lag in common_lags
    ]

    ranked = [
        item
        for item in (*scalar_metrics, *autocorrelation)
        if item["relative_difference"] is not None
    ]
    ranked.sort(key=lambda item: item["relative_difference"], reverse=True)

    return {
        "schema_version": "umwelt.comparison.v1",
        "recorded_dataset_id": recorded["dataset_id"],
        "synthetic_dataset_id": synthetic["dataset_id"],
        "interpretation": (
            "Differences are descriptive. Umwelt v0.1 does not assign a pass/fail "
            "threshold or treat correspondence as proof of biological mechanism."
        ),
        "scalar_metrics": scalar_metrics,
        "phase_profile": {
            "sleep_fraction_rmse": _profile_rmse(
                recorded["phase_profile"], synthetic["phase_profile"], "sleep_fraction"
            ),
            "mean_activity_rmse": _profile_rmse(
                recorded["phase_profile"], synthetic["phase_profile"], "mean_activity"
            ),
        },
        "autocorrelation": autocorrelation,
        "largest_relative_discrepancies": ranked[:5],
    }
