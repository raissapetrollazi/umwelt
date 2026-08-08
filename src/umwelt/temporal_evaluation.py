"""Replicate, distribution, and subject-level evaluation for the temporal lab."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import sqrt
from statistics import fmean

from umwelt.errors import DataError
from umwelt.evaluation import bout_durations, describe_dataset
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


def metric_snapshot(metrics: Mapping[str, object]) -> dict[str, object]:
    """Flatten important v0.1 metrics while retaining declared phase profiles."""

    occupancy = metrics["state_occupancy"]
    activity = metrics["activity"]
    transitions = metrics["transitions"]
    bouts = metrics["bout_durations_seconds"]
    autocorrelation = metrics["sleep_state_autocorrelation"]
    profile = metrics["phase_profile"]
    scalars: dict[str, float | int | None] = {
        "sleep_fraction": occupancy["sleep_fraction"],
        "wake_fraction": occupancy["wake_fraction"],
        "mean_activity": activity["all_epochs"]["mean"],
        "nonzero_activity_fraction": activity["nonzero_fraction"],
        "state_changes_per_hour": transitions["state_changes_per_hour"],
    }
    for state in ("wake", "sleep"):
        for statistic in ("mean", "median", "q90"):
            scalars[f"{state}_bout_{statistic}_seconds"] = bouts[state][statistic]
    for item in autocorrelation:
        scalars[f"sleep_autocorrelation_lag_{item['lag_epochs']}"] = item["value"]
    return {
        "scalars": scalars,
        "phase_profile": {
            "sleep_fraction": [item["sleep_fraction"] for item in profile],
            "mean_activity": [item["mean_activity"] for item in profile],
        },
    }


def _collect_bouts(
    series_collection: Iterable[TimeSeries], epoch_seconds: int
) -> dict[str, list[float]]:
    collected = {"wake": [], "sleep": []}
    for series in series_collection:
        values = bout_durations(series, epoch_seconds)
        collected["wake"].extend(values[BehavioralState.WAKE])
        collected["sleep"].extend(values[BehavioralState.SLEEP])
    return collected


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


def _singleton(dataset: ObservationDataset, series: TimeSeries) -> ObservationDataset:
    return ObservationDataset(
        dataset_id=f"{dataset.dataset_id}:{series.subject_id}",
        series=(series,),
        provenance=dataset.provenance,
    )


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
    lags = tuple(autocorrelation_lags)
    aggregate_metrics = describe_dataset(
        dataset, phase_bins=phase_bins, autocorrelation_lags=lags
    )
    epoch_seconds = int(aggregate_metrics["epoch_seconds"])
    subjects = {}
    for series in dataset.series:
        metrics = describe_dataset(
            _singleton(dataset, series),
            phase_bins=phase_bins,
            autocorrelation_lags=lags,
        )
        subjects[series.subject_id] = EvaluationUnit(
            snapshot=metric_snapshot(metrics),
            bouts=_collect_bouts((series,), epoch_seconds),
        )
    return RecordedReference(
        aggregate=EvaluationUnit(
            snapshot=metric_snapshot(aggregate_metrics),
            bouts=_collect_bouts(dataset.series, epoch_seconds),
        ),
        subjects=subjects,
        phase_bins=phase_bins,
        autocorrelation_lags=lags,
        epoch_seconds=epoch_seconds,
    )


def _compare_unit(
    recorded: EvaluationUnit,
    synthetic_metrics: Mapping[str, object],
    synthetic_bouts: Mapping[str, Sequence[float]],
) -> dict[str, object]:
    synthetic = metric_snapshot(synthetic_metrics)
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
            synthetic["phase_profile"][variable],
        )
    return {"metrics": synthetic, "discrepancies": discrepancies}


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
    arguments = {
        "phase_bins": recorded.phase_bins,
        "autocorrelation_lags": recorded.autocorrelation_lags,
    }
    aggregate_metrics = describe_dataset(dataset, **arguments)
    aggregate = _compare_unit(
        recorded.aggregate,
        aggregate_metrics,
        _collect_bouts(dataset.series, recorded.epoch_seconds),
    )
    subjects = {}
    for series in dataset.series:
        metrics = describe_dataset(_singleton(dataset, series), **arguments)
        subjects[series.subject_id] = _compare_unit(
            recorded.subjects[series.subject_id],
            metrics,
            _collect_bouts((series,), recorded.epoch_seconds),
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
