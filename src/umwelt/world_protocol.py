"""Frozen comparison and intervention rules for the first v0.3 experiment."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from math import isclose, isfinite
from statistics import median

from umwelt.errors import DataError
from umwelt.spatial_protocol import (
    ROCHE_CONTROL_DEVELOPMENT_SUBJECTS,
    ROCHE_CONTROL_FITTING_SUBJECTS,
)
from umwelt.world import (
    V03_ARENA_SCALE_CONDITIONS,
    V03_BOUNDARY_BAND_FRACTION,
    WorldZone,
)
from umwelt.world_evaluation import WorldComparison
from umwelt.world_model import BOUNDARY_CONDITIONED_MODEL_ID


WORLD_PROTOCOL_ID = "umwelt.behavioral-world.v1"
BASELINE_MODEL_ID = "persistent-reflecting-random-walk-v1"
WORLD_MODEL_IDS = (BASELINE_MODEL_ID, BOUNDARY_CONDITIONED_MODEL_ID)
WORLD_RECORDED_CONDITION_ID = "recorded"
WORLD_SEED_NAMESPACE = "v0.3-behavioral-world"
WORLD_PREDICTIVE_QUANTILES = (0.05, 0.5, 0.95)
WORLD_COMPARISON_METRIC_IDS = (
    "path-length-signed-relative-difference",
    "adjacent-displacement-wasserstein-px",
    "absolute-turning-wasserstein-radians",
    "boundary-band-occupancy-absolute-difference",
    "zone-conditioned-positive-displacement-wasserstein-px",
)
WORLD_INTERVENTION_METRIC_IDS = (
    "path-length-px-with-valid-transition-count",
    "boundary-band-occupancy-fraction",
    "zone-transition-counts",
    "zone-positive-displacement-summaries",
    "reflection-counts",
)
WORLD_PREFERENCE_CRITERION_IDS = (
    "lower-median-absolute-relative-path-discrepancy",
    "lower-absolute-relative-path-discrepancy-in-at-least-four-animals",
    "no-higher-median-adjacent-displacement-distance",
    "no-higher-median-absolute-turning-distance",
    "lower-median-boundary-band-occupancy-error",
)


def _unsigned_seed(value: int) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and 0 <= value < 2**64


def derive_world_replicate_seed(
    master_seed: int, subject_id: str, replicate: int
) -> int:
    """Pair model and intervention runs by omitting their identities from the seed."""

    if not _unsigned_seed(master_seed):
        raise DataError("The v0.3 master seed must be an unsigned 64-bit integer.")
    if (
        subject_id
        not in ROCHE_CONTROL_FITTING_SUBJECTS + ROCHE_CONTROL_DEVELOPMENT_SUBJECTS
    ):
        raise DataError("The v0.3 seed requires a pinned control animal.")
    if isinstance(replicate, bool) or not isinstance(replicate, int) or replicate < 0:
        raise DataError("The v0.3 replicate index must be a non-negative integer.")
    payload = f"{master_seed}|{WORLD_SEED_NAMESPACE}|{subject_id}|{replicate}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


@dataclass(frozen=True, slots=True)
class WorldCrossValidationFold:
    """Five fitting animals and one evaluation animal from the development pool."""

    evaluation_subject_id: str
    fitting_subject_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        expected = tuple(
            subject_id
            for subject_id in ROCHE_CONTROL_FITTING_SUBJECTS
            if subject_id != self.evaluation_subject_id
        )
        if (
            self.evaluation_subject_id not in ROCHE_CONTROL_FITTING_SUBJECTS
            or self.fitting_subject_ids != expected
        ):
            raise DataError("World cross-validation folds must use the pinned split.")

    def to_dict(self) -> dict[str, object]:
        return {
            "fitting_subject_ids": list(self.fitting_subject_ids),
            "evaluation_subject_id": self.evaluation_subject_id,
        }


WORLD_CROSS_VALIDATION_FOLDS = tuple(
    WorldCrossValidationFold(
        evaluation_subject_id=subject_id,
        fitting_subject_ids=tuple(
            other for other in ROCHE_CONTROL_FITTING_SUBJECTS if other != subject_id
        ),
    )
    for subject_id in ROCHE_CONTROL_FITTING_SUBJECTS
)


@dataclass(frozen=True, slots=True)
class WorldExperimentProtocol:
    """Run choices and scientific constants fixed before v0.3 animal results."""

    master_seed: int
    replicates: int

    def __post_init__(self) -> None:
        if not _unsigned_seed(self.master_seed):
            raise DataError("The v0.3 master seed must be an unsigned 64-bit integer.")
        if (
            isinstance(self.replicates, bool)
            or not isinstance(self.replicates, int)
            or not 1 <= self.replicates <= 100
        ):
            raise DataError("The v0.3 replicate count must be between 1 and 100.")

    def seed(self, subject_id: str, replicate: int) -> int:
        if replicate >= self.replicates:
            raise DataError("Replicate index exceeds the v0.3 protocol count.")
        return derive_world_replicate_seed(self.master_seed, subject_id, replicate)

    def to_dict(self) -> dict[str, object]:
        """Serialize the frozen split, metrics, preference rule, and interventions."""

        return {
            "protocol_id": WORLD_PROTOCOL_ID,
            "development_pool_subject_ids": list(ROCHE_CONTROL_FITTING_SUBJECTS),
            "legacy_diagnostic_subject_ids": list(ROCHE_CONTROL_DEVELOPMENT_SUBJECTS),
            "cross_validation": {
                "method": "leave-one-animal-out",
                "folds": [fold.to_dict() for fold in WORLD_CROSS_VALIDATION_FOLDS],
                "model_ids": list(WORLD_MODEL_IDS),
                "condition_id": WORLD_RECORDED_CONDITION_ID,
                "registered_comparison_metric_ids": list(WORLD_COMPARISON_METRIC_IDS),
                "preference_criterion_ids": list(WORLD_PREFERENCE_CRITERION_IDS),
                "preference_unit": "animal-median-across-replicates",
                "legacy_diagnostic_excluded_from_preference": True,
                "retain_all_replicate_comparisons": True,
                "predictive_quantiles": list(WORLD_PREDICTIVE_QUANTILES),
                "predictive_quantile_method": "linear-interpolation-at-p-times-n-minus-one",
            },
            "simulation": {
                "master_seed": self.master_seed,
                "replicates": self.replicates,
                "seed_derivation": "sha256-first-eight-bytes-big-endian",
                "seed_namespace": WORLD_SEED_NAMESPACE,
                "seed_payload_template": (
                    "{master_seed}|{seed_namespace}|{subject_id}|{replicate}"
                ),
                "paired_across_models_and_conditions": True,
                "availability_mask": "recorded-bodycentre-availability-only",
            },
            "interventions": {
                "model_id": BOUNDARY_CONDITIONED_MODEL_ID,
                "template_subject_ids": list(ROCHE_CONTROL_FITTING_SUBJECTS),
                "conditions": [
                    condition.to_dict() for condition in V03_ARENA_SCALE_CONDITIONS
                ],
                "boundary_band_fraction": V03_BOUNDARY_BAND_FRACTION,
                "comparison_anchor": WORLD_RECORDED_CONDITION_ID,
                "recorded_coordinates_compared_only_at_recorded_scale": True,
                "registered_metric_ids": list(WORLD_INTERVENTION_METRIC_IDS),
                "reflection_count_scope": "full-generated-trajectory-before-observation-mask",
            },
        }


def _finite_nonnegative(value: float | None, label: str) -> float:
    if value is None or not isfinite(value) or value < 0:
        raise DataError(f"World preference requires finite non-negative {label}.")
    return value


@dataclass(frozen=True, slots=True)
class WorldReplicateComparison:
    """One recorded-scale comparison, with identity and a shared random stream."""

    subject_id: str
    model_id: str
    replicate: int
    seed: int
    condition_id: str
    comparison: WorldComparison

    def __post_init__(self) -> None:
        if self.subject_id not in ROCHE_CONTROL_FITTING_SUBJECTS:
            raise DataError(
                "Preference input must come from a cross-validation animal."
            )
        if self.model_id not in WORLD_MODEL_IDS:
            raise DataError("Preference input must use a declared v0.3 model.")
        if (
            isinstance(self.replicate, bool)
            or not isinstance(self.replicate, int)
            or self.replicate < 0
        ):
            raise DataError("Preference replicate index must be non-negative.")
        if not _unsigned_seed(self.seed):
            raise DataError("Preference seed must be an unsigned 64-bit integer.")
        if self.condition_id != WORLD_RECORDED_CONDITION_ID:
            raise DataError("Model preference uses only the recorded-scale condition.")
        if not isinstance(self.comparison, WorldComparison):
            raise DataError("Preference input requires a world comparison.")
        spatial = self.comparison.spatial
        signed_path = spatial.path_length_signed_relative_difference
        if signed_path is None or not isfinite(signed_path):
            raise DataError("Preference requires finite signed relative path length.")
        absolute_path = _finite_nonnegative(
            spatial.path_length_relative_difference, "absolute relative path length"
        )
        if not isclose(absolute_path, abs(signed_path), rel_tol=1e-12, abs_tol=1e-12):
            raise DataError("Absolute and signed relative path discrepancies disagree.")
        _finite_nonnegative(
            spatial.adjacent_displacement_wasserstein_px,
            "adjacent-displacement distance",
        )
        _finite_nonnegative(
            spatial.absolute_turning_wasserstein_radians,
            "absolute-turning distance",
        )
        occupancy = _finite_nonnegative(
            self.comparison.boundary_band_occupancy_absolute_difference,
            "boundary-band occupancy error",
        )
        if occupancy > 1:
            raise DataError("Boundary-band occupancy error cannot exceed one.")
        zones = self.comparison.zone_positive_displacement_wasserstein_px
        if tuple(zone for zone, _ in zones) != tuple(zone.value for zone in WorldZone):
            raise DataError("Zone distances must retain both declared zones in order.")
        for zone, value in zones:
            if value is not None:
                _finite_nonnegative(value, f"{zone} displacement distance")

    def to_dict(self) -> dict[str, object]:
        return {
            "subject_id": self.subject_id,
            "model_id": self.model_id,
            "replicate": self.replicate,
            "seed": self.seed,
            "condition_id": self.condition_id,
            "comparison": self.comparison.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ModelMetricMedians:
    """Per-animal medians for the four metrics used by the preference rule."""

    absolute_relative_path_length: float
    adjacent_displacement_wasserstein_px: float
    absolute_turning_wasserstein_radians: float
    boundary_band_occupancy_absolute_difference: float

    def to_dict(self) -> dict[str, float]:
        return {
            "absolute_relative_path_length": self.absolute_relative_path_length,
            "adjacent_displacement_wasserstein_px": self.adjacent_displacement_wasserstein_px,
            "absolute_turning_wasserstein_radians": self.absolute_turning_wasserstein_radians,
            "boundary_band_occupancy_absolute_difference": (
                self.boundary_band_occupancy_absolute_difference
            ),
        }


def _medians(rows: tuple[WorldReplicateComparison, ...]) -> ModelMetricMedians:
    spatial = tuple(row.comparison.spatial for row in rows)
    return ModelMetricMedians(
        absolute_relative_path_length=float(
            median(
                _finite_nonnegative(
                    item.path_length_relative_difference,
                    "absolute relative path length",
                )
                for item in spatial
            )
        ),
        adjacent_displacement_wasserstein_px=float(
            median(
                _finite_nonnegative(
                    item.adjacent_displacement_wasserstein_px,
                    "adjacent-displacement distance",
                )
                for item in spatial
            )
        ),
        absolute_turning_wasserstein_radians=float(
            median(
                _finite_nonnegative(
                    item.absolute_turning_wasserstein_radians,
                    "absolute-turning distance",
                )
                for item in spatial
            )
        ),
        boundary_band_occupancy_absolute_difference=float(
            median(
                _finite_nonnegative(
                    row.comparison.boundary_band_occupancy_absolute_difference,
                    "boundary-band occupancy error",
                )
                for row in rows
            )
        ),
    )


@dataclass(frozen=True, slots=True)
class WorldPreferenceDecision:
    """Descriptive development decision; never a validation or population test."""

    subject_medians: tuple[tuple[str, ModelMetricMedians, ModelMetricMedians], ...]
    aggregate_baseline: ModelMetricMedians
    aggregate_candidate: ModelMetricMedians
    path_improved_animal_count: int
    criteria: tuple[tuple[str, bool], ...]

    @property
    def eligible_for_independent_evaluation(self) -> bool:
        return all(passed for _, passed in self.criteria)

    def to_dict(self) -> dict[str, object]:
        return {
            "decision_scope": "development-only-eligibility-for-independent-evaluation",
            "subject_medians": [
                {
                    "subject_id": subject_id,
                    BASELINE_MODEL_ID: baseline.to_dict(),
                    BOUNDARY_CONDITIONED_MODEL_ID: candidate.to_dict(),
                }
                for subject_id, baseline, candidate in self.subject_medians
            ],
            "aggregate_animal_medians": {
                BASELINE_MODEL_ID: self.aggregate_baseline.to_dict(),
                BOUNDARY_CONDITIONED_MODEL_ID: self.aggregate_candidate.to_dict(),
            },
            "path_improved_animal_count": self.path_improved_animal_count,
            "criteria": dict(self.criteria),
            "eligible_for_independent_evaluation": (
                self.eligible_for_independent_evaluation
            ),
        }


def _across_animals(
    rows: tuple[tuple[str, ModelMetricMedians, ModelMetricMedians], ...],
    *,
    candidate: bool,
) -> ModelMetricMedians:
    metrics = tuple(row[2] if candidate else row[1] for row in rows)
    return ModelMetricMedians(
        absolute_relative_path_length=float(
            median(item.absolute_relative_path_length for item in metrics)
        ),
        adjacent_displacement_wasserstein_px=float(
            median(item.adjacent_displacement_wasserstein_px for item in metrics)
        ),
        absolute_turning_wasserstein_radians=float(
            median(item.absolute_turning_wasserstein_radians for item in metrics)
        ),
        boundary_band_occupancy_absolute_difference=float(
            median(item.boundary_band_occupancy_absolute_difference for item in metrics)
        ),
    )


def evaluate_world_model_preference(
    protocol: WorldExperimentProtocol,
    comparisons: tuple[WorldReplicateComparison, ...],
) -> WorldPreferenceDecision:
    """Apply the five pre-declared rules to complete, seed-paired CV results."""

    expected = {
        (subject_id, model_id, replicate)
        for subject_id in ROCHE_CONTROL_FITTING_SUBJECTS
        for model_id in WORLD_MODEL_IDS
        for replicate in range(protocol.replicates)
    }
    actual: dict[tuple[str, str, int], WorldReplicateComparison] = {}
    for row in comparisons:
        if not isinstance(row, WorldReplicateComparison):
            raise DataError("Preference input must contain replicate comparisons.")
        key = (row.subject_id, row.model_id, row.replicate)
        if key in actual:
            raise DataError("Duplicate v0.3 model comparison replicate.")
        if row.seed != protocol.seed(row.subject_id, row.replicate):
            raise DataError("Model comparisons must use the declared paired seed.")
        actual[key] = row
    if set(actual) != expected:
        raise DataError(
            "Preference requires every declared animal, model, and replicate."
        )

    subject_medians = tuple(
        (
            subject_id,
            _medians(
                tuple(
                    actual[(subject_id, BASELINE_MODEL_ID, replicate)]
                    for replicate in range(protocol.replicates)
                )
            ),
            _medians(
                tuple(
                    actual[(subject_id, BOUNDARY_CONDITIONED_MODEL_ID, replicate)]
                    for replicate in range(protocol.replicates)
                )
            ),
        )
        for subject_id in ROCHE_CONTROL_FITTING_SUBJECTS
    )
    baseline = _across_animals(subject_medians, candidate=False)
    candidate = _across_animals(subject_medians, candidate=True)
    path_improved = sum(
        candidate_metrics.absolute_relative_path_length
        < baseline_metrics.absolute_relative_path_length
        for _, baseline_metrics, candidate_metrics in subject_medians
    )
    criteria = tuple(
        zip(
            WORLD_PREFERENCE_CRITERION_IDS,
            (
                candidate.absolute_relative_path_length
                < baseline.absolute_relative_path_length,
                path_improved >= 4,
                candidate.adjacent_displacement_wasserstein_px
                <= baseline.adjacent_displacement_wasserstein_px,
                candidate.absolute_turning_wasserstein_radians
                <= baseline.absolute_turning_wasserstein_radians,
                candidate.boundary_band_occupancy_absolute_difference
                < baseline.boundary_band_occupancy_absolute_difference,
            ),
            strict=True,
        )
    )
    return WorldPreferenceDecision(
        subject_medians=subject_medians,
        aggregate_baseline=baseline,
        aggregate_candidate=candidate,
        path_improved_animal_count=path_improved,
        criteria=criteria,
    )
