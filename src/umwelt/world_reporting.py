"""Compact, animal-level summaries for the first behavioral-world run."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite

from umwelt.errors import DataError
from umwelt.spatial_protocol import (
    ROCHE_CONTROL_DEVELOPMENT_SUBJECTS,
    ROCHE_CONTROL_FITTING_SUBJECTS,
)
from umwelt.world import V03_ARENA_SCALE_CONDITIONS, WorldZone
from umwelt.world_config import WorldLabConfig
from umwelt.world_evaluation import WorldComparison
from umwelt.world_execution import (
    INTERVENTION_VALUE_IDS,
    InterventionOutcome,
    WorldRunData,
)
from umwelt.world_protocol import (
    BOUNDARY_CONDITIONED_MODEL_ID,
    WORLD_MODEL_IDS,
    WORLD_PREDICTIVE_QUANTILES,
    WorldExperimentProtocol,
)


PREDICTIVE_METRIC_IDS = (
    "path_length_signed_relative_difference",
    "path_length_absolute_relative_difference",
    "adjacent_displacement_wasserstein_px",
    "absolute_turning_wasserstein_radians",
    "boundary_band_occupancy_absolute_difference",
    "boundary_band_positive_displacement_wasserstein_px",
    "interior_positive_displacement_wasserstein_px",
)
INTERVENTION_DELTA_IDS = tuple(
    metric_id
    for metric_id in INTERVENTION_VALUE_IDS
    if metric_id != "valid_transition_count"
)


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    weight = position - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def quantile_summary(values: Sequence[float | int | None]) -> dict[str, object]:
    """Retain missing counts and the three frozen replicate quantiles."""

    present = sorted(float(value) for value in values if value is not None)
    if any(not isfinite(value) for value in present):
        raise DataError("World summaries require finite values.")
    return {
        "present_count": len(present),
        "missing_count": len(values) - len(present),
        **{
            f"p{int(probability * 100):02d}": (
                _quantile(present, probability) if present else None
            )
            for probability in WORLD_PREDICTIVE_QUANTILES
        },
    }


def _comparison_metrics(comparison: WorldComparison) -> dict[str, float | None]:
    spatial = comparison.spatial
    zones = dict(comparison.zone_positive_displacement_wasserstein_px)
    return {
        "path_length_signed_relative_difference": (
            spatial.path_length_signed_relative_difference
        ),
        "path_length_absolute_relative_difference": (
            spatial.path_length_relative_difference
        ),
        "adjacent_displacement_wasserstein_px": (
            spatial.adjacent_displacement_wasserstein_px
        ),
        "absolute_turning_wasserstein_radians": (
            spatial.absolute_turning_wasserstein_radians
        ),
        "boundary_band_occupancy_absolute_difference": (
            comparison.boundary_band_occupancy_absolute_difference
        ),
        "boundary_band_positive_displacement_wasserstein_px": zones[
            WorldZone.BOUNDARY_BAND.value
        ],
        "interior_positive_displacement_wasserstein_px": zones[
            WorldZone.INTERIOR.value
        ],
    }


def _predictive_summary(comparisons: Sequence[WorldComparison]) -> dict[str, object]:
    metrics = tuple(_comparison_metrics(comparison) for comparison in comparisons)
    return {
        metric_id: quantile_summary(tuple(item[metric_id] for item in metrics))
        for metric_id in PREDICTIVE_METRIC_IDS
    }


def build_model_comparison(
    protocol: WorldExperimentProtocol,
    run: WorldRunData,
) -> dict[str, object]:
    """Report every registered discrepancy, including the legacy diagnostic."""

    cross_validation: list[dict[str, object]] = []
    for subject_id in ROCHE_CONTROL_FITTING_SUBJECTS:
        for model_id in WORLD_MODEL_IDS:
            rows = tuple(
                row
                for row in run.comparison_rows
                if row.subject_id == subject_id and row.model_id == model_id
            )
            if len(rows) != protocol.replicates:
                raise DataError("World predictive summary requires complete CV rows.")
            cross_validation.append(
                {
                    "subject_id": subject_id,
                    "model_id": model_id,
                    "replicate_count": len(rows),
                    "predictive": _predictive_summary(
                        tuple(row.comparison for row in rows)
                    ),
                }
            )
    legacy: list[dict[str, object]] = []
    for subject_id in ROCHE_CONTROL_DEVELOPMENT_SUBJECTS:
        for model_id in WORLD_MODEL_IDS:
            legacy_matches = tuple(
                row
                for row in run.legacy_rows
                if row.subject_id == subject_id and row.model_id == model_id
            )
            if len(legacy_matches) != protocol.replicates:
                raise DataError(
                    "World predictive summary requires complete legacy rows."
                )
            legacy.append(
                {
                    "subject_id": subject_id,
                    "model_id": model_id,
                    "replicate_count": len(legacy_matches),
                    "predictive": _predictive_summary(
                        tuple(row.comparison for row in legacy_matches)
                    ),
                    "excluded_from_preference": True,
                }
            )
    return {
        "decision": run.preference.to_dict(),
        "predictive_quantiles": list(WORLD_PREDICTIVE_QUANTILES),
        "predictive_quantile_method": "linear-interpolation-at-p-times-n-minus-one",
        "registered_predictive_metric_ids": list(PREDICTIVE_METRIC_IDS),
        "cross_validation": cross_validation,
        "legacy_diagnostic": legacy,
        "legacy_diagnostic_used_for_model_selection": False,
        "population_or_validation_claim": False,
    }


def _paired_outcomes(
    protocol: WorldExperimentProtocol,
    rows: Sequence[InterventionOutcome],
) -> dict[tuple[str, int, str], InterventionOutcome]:
    expected_conditions = tuple(
        condition.condition_id for condition in V03_ARENA_SCALE_CONDITIONS
    )
    expected = {
        (subject_id, replicate, condition_id)
        for subject_id in ROCHE_CONTROL_FITTING_SUBJECTS
        for replicate in range(protocol.replicates)
        for condition_id in expected_conditions
    }
    actual: dict[tuple[str, int, str], InterventionOutcome] = {}
    for row in rows:
        if row.subject_id not in ROCHE_CONTROL_FITTING_SUBJECTS:
            raise DataError("Intervention templates must use the declared six animals.")
        key = (row.subject_id, row.replicate, row.condition_id)
        if key in actual:
            raise DataError("Duplicate world intervention outcome.")
        if row.seed != protocol.seed(row.subject_id, row.replicate):
            raise DataError("Intervention outcomes must use paired protocol seeds.")
        actual[key] = row
    if set(actual) != expected:
        raise DataError("Intervention comparison requires every declared paired run.")
    for subject_id in ROCHE_CONTROL_FITTING_SUBJECTS:
        for replicate in range(protocol.replicates):
            exposures = {
                dict(actual[(subject_id, replicate, condition_id)].values)[
                    "valid_transition_count"
                ]
                for condition_id in expected_conditions
            }
            if len(exposures) != 1:
                raise DataError(
                    "Paired interventions require equal transition exposure."
                )
    return actual


def _paired_deltas(
    paired: Mapping[tuple[str, int, str], InterventionOutcome],
    subject_id: str,
    replicate: int,
    condition_id: str,
) -> dict[str, float | None]:
    anchor = dict(paired[(subject_id, replicate, "recorded")].values)
    changed = dict(paired[(subject_id, replicate, condition_id)].values)
    deltas: dict[str, float | None] = {}
    for metric_id in INTERVENTION_DELTA_IDS:
        changed_value = changed[metric_id]
        anchor_value = anchor[metric_id]
        deltas[metric_id] = (
            float(changed_value - anchor_value)
            if changed_value is not None and anchor_value is not None
            else None
        )
    return deltas


def build_intervention_comparison(
    protocol: WorldExperimentProtocol,
    outcomes: Sequence[InterventionOutcome],
) -> dict[str, object]:
    """Summarize paired synthetic changes by animal before aggregating animals."""

    paired = _paired_outcomes(protocol, outcomes)
    per_animal: list[dict[str, object]] = []
    aggregate: dict[str, object] = {}
    for condition_id in ("compact", "expanded"):
        animal_medians: dict[str, list[float | None]] = {
            metric_id: [] for metric_id in INTERVENTION_DELTA_IDS
        }
        for subject_id in ROCHE_CONTROL_FITTING_SUBJECTS:
            deltas = tuple(
                _paired_deltas(paired, subject_id, replicate, condition_id)
                for replicate in range(protocol.replicates)
            )
            summaries = {
                metric_id: quantile_summary(tuple(row[metric_id] for row in deltas))
                for metric_id in INTERVENTION_DELTA_IDS
            }
            for metric_id, summary in summaries.items():
                median_value = summary["p50"]
                if median_value is not None and not isinstance(
                    median_value, (int, float)
                ):
                    raise DataError("World predictive median must be numeric.")
                animal_medians[metric_id].append(
                    float(median_value) if median_value is not None else None
                )
            per_animal.append(
                {
                    "subject_id": subject_id,
                    "condition_id": condition_id,
                    "paired_replicates": protocol.replicates,
                    "delta_vs_recorded_condition": summaries,
                }
            )
        aggregate[condition_id] = {
            metric_id: quantile_summary(values)
            for metric_id, values in animal_medians.items()
        }
    return {
        "comparison_scope": "synthetic-model-arena-scale-sensitivity",
        "model_id": BOUNDARY_CONDITIONED_MODEL_ID,
        "recorded_animal_intervention_counterpart": False,
        "paired_anchor_condition_id": "recorded",
        "equal_valid_transition_exposure": True,
        "reflection_count_scope": "full-generated-trajectory-before-observation-mask",
        "delta_metric_ids": list(INTERVENTION_DELTA_IDS),
        "per_animal": per_animal,
        "aggregate_animal_medians": aggregate,
    }


def _aggregate_p50(
    intervention: Mapping[str, object], condition_id: str, metric_id: str
) -> float | None:
    aggregate = intervention.get("aggregate_animal_medians")
    if not isinstance(aggregate, dict):
        raise DataError("Intervention report lacks aggregate animal medians.")
    condition = aggregate.get(condition_id)
    if not isinstance(condition, dict):
        raise DataError("Intervention report lacks a declared condition.")
    summary = condition.get(metric_id)
    if not isinstance(summary, dict):
        raise DataError("Intervention report lacks a declared metric.")
    value = summary.get("p50")
    return float(value) if value is not None else None


def _format(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.6g}"


def render_world_report(
    *,
    config: WorldLabConfig,
    run: WorldRunData,
    intervention_comparison: Mapping[str, object],
) -> str:
    """Describe measured development results without biological overclaiming."""

    decision = run.preference
    lines = [
        f"# Experiment report: {config.experiment_id}",
        "",
        "## Interpretation boundary",
        "",
        (
            "This is a model-development comparison and a synthetic arena-size "
            "sensitivity analysis. Neither the model preference rule nor a synthetic "
            "intervention response validates a biological mechanism or predicts how "
            "a real animal would respond to an arena-size change."
        ),
        "",
        "## Protocol",
        "",
        "- Dataset: pinned Roche open-field Control dose 0 recordings",
        "- Cross-validation: six leave-one-animal-out folds, each fitted on five controls",
        "- Legacy diagnostic: two previously inspected controls, excluded from preference",
        f"- Master seed: {config.seed}; replicates per animal/model: {config.replicates}",
        "- Position: source `bodycentre`, no invented likelihood cutoff or smoothing",
        "- Units: image pixels and radians; no physical speed or arena dimension claims",
        (
            "- Recorded positions outside landmark-derived bounds remain in spatial "
            "metrics and quality control; starts outside have no zone assignment"
        ),
        "",
        "## Recorded-scale model preference",
        "",
        (
            "M1 eligible for later independent evaluation: **"
            + ("yes" if decision.eligible_for_independent_evaluation else "no")
            + "**. This is a pre-declared development rule, not a validation threshold."
        ),
        "",
        "| Criterion | Result |",
        "| --- | --- |",
    ]
    lines.extend(
        f"| `{name}` | {'met' if passed else 'not met'} |"
        for name, passed in decision.criteria
    )
    lines.extend(
        [
            "",
            (
                "Animals with lower absolute relative path discrepancy: "
                f"{decision.path_improved_animal_count} of 6."
            ),
            "",
            "| Animal | M0 absolute relative path error | M1 absolute relative path error |",
            "| --- | ---: | ---: |",
        ]
    )
    lines.extend(
        f"| `{subject_id}` | {baseline.absolute_relative_path_length:.6g} | "
        f"{candidate.absolute_relative_path_length:.6g} |"
        for subject_id, baseline, candidate in decision.subject_medians
    )
    lines.extend(
        [
            "",
            (
                "The machine-readable model comparison reports all registered "
                "metric distributions and 5th/50th/95th replicate percentiles, "
                "including the separate legacy diagnostic."
            ),
            "",
            "## Paired arena-size interventions",
            "",
            (
                "The final M1 model was fitted on the six development-pool animals. "
                "Their arenas and observation masks supplied templates for paired "
                "compact, recorded-scale, and expanded synthetic conditions."
            ),
            "",
            "| Condition vs recorded-scale simulation | Median animal path change (px) | "
            "Median animal boundary occupancy change | Median animal reflection change |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for condition_id in ("compact", "expanded"):
        lines.append(
            f"| `{condition_id}` | "
            f"{_format(_aggregate_p50(intervention_comparison, condition_id, 'path_length_px'))} | "
            f"{_format(_aggregate_p50(intervention_comparison, condition_id, 'boundary_band_occupancy_fraction'))} | "
            f"{_format(_aggregate_p50(intervention_comparison, condition_id, 'reflection_count'))} |"
        )
    lines.extend(
        [
            "",
            (
                "The intervention artifact also retains paired changes in valid "
                "zone transitions and positive-displacement summaries. Reflection "
                "counts are model/debug values from the full generated path before "
                "observation masking. No recorded arena-size intervention exists here."
            ),
            "",
            "## Reproducibility",
            "",
            (
                "Resolved configuration, frozen protocol, source checksums, fitted "
                "fold and final models, paired seeds, compact replicate metrics, "
                "software provenance, and artifact hashes accompany this report. "
                "Full synthetic trajectories were discarded after measurement."
            ),
            "",
        ]
    )
    return "\n".join(lines)
