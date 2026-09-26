"""Paired recorded-scale and intervention execution for Umwelt v0.3."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from statistics import fmean, median

from umwelt.errors import DataError
from umwelt.observations import ObservationSource
from umwelt.spatial import Keypoint2D, Pose2D, SpatialFrame
from umwelt.spatial_model import (
    SpatialMovementModel,
    fit_spatial_movement_model,
    generate_spatial_trajectory,
)
from umwelt.spatial_protocol import (
    ROCHE_CONTROL_DEVELOPMENT_SUBJECTS,
    ROCHE_CONTROL_FITTING_SUBJECTS,
    SpatialTrajectoryProtocol,
)
from umwelt.trajectory import PositionStatus
from umwelt.world import V03_ARENA_SCALE_CONDITIONS, OpenFieldWorld, WorldZone
from umwelt.world_evaluation import (
    WorldComparison,
    WorldMeasurements,
    compare_world_measurements,
    measure_world_steps,
)
from umwelt.world_model import (
    BoundaryConditionedMovementModel,
    fit_boundary_conditioned_movement_model,
    generate_boundary_conditioned_trajectory,
)
from umwelt.world_protocol import (
    BASELINE_MODEL_ID,
    BOUNDARY_CONDITIONED_MODEL_ID,
    WORLD_CROSS_VALIDATION_FOLDS,
    WORLD_RECORDED_CONDITION_ID,
    WorldExperimentProtocol,
    WorldPreferenceDecision,
    WorldReplicateComparison,
    evaluate_world_model_preference,
)
from umwelt.world_recordings import RecordedWorldSeries


INTERVENTION_VALUE_IDS = (
    "path_length_px",
    "valid_transition_count",
    "boundary_band_occupancy_fraction",
    "boundary_band_valid_transition_count",
    "interior_valid_transition_count",
    "boundary_band_positive_displacement_count",
    "boundary_band_positive_displacement_mean_px",
    "boundary_band_positive_displacement_median_px",
    "interior_positive_displacement_count",
    "interior_positive_displacement_mean_px",
    "interior_positive_displacement_median_px",
    "reflection_count",
)


@dataclass(frozen=True, slots=True)
class InterventionOutcome:
    """Compact model response for one subject, replicate, and world condition."""

    subject_id: str
    replicate: int
    seed: int
    condition_id: str
    values: tuple[tuple[str, float | int | None], ...]

    def __post_init__(self) -> None:
        if tuple(name for name, _ in self.values) != INTERVENTION_VALUE_IDS:
            raise DataError("Intervention outcomes must retain every declared value.")
        for name, value in self.values:
            if value is not None and not isfinite(value):
                raise DataError(f"Intervention value {name} must be finite.")

    def to_dict(self) -> dict[str, object]:
        return {
            "subject_id": self.subject_id,
            "replicate": self.replicate,
            "seed": self.seed,
            "condition_id": self.condition_id,
            "values": dict(self.values),
        }


@dataclass(frozen=True, slots=True)
class LegacyDiagnosticComparison:
    """Post-freeze reference result excluded from the model preference rule."""

    subject_id: str
    model_id: str
    replicate: int
    seed: int
    comparison: WorldComparison


@dataclass(frozen=True, slots=True)
class WorldRunData:
    """Compact experiment products; no full synthetic trajectory is retained."""

    worlds: dict[str, object]
    models: dict[str, object]
    records: tuple[dict[str, object], ...]
    comparison_rows: tuple[WorldReplicateComparison, ...]
    legacy_rows: tuple[LegacyDiagnosticComparison, ...]
    preference: WorldPreferenceDecision
    intervention_outcomes: tuple[InterventionOutcome, ...]
    generated_series_count: int


def _masked_frames(
    series: RecordedWorldSeries, generated: Sequence[SpatialFrame]
) -> Iterator[SpatialFrame]:
    """Copy source availability reasons without copying recorded coordinates."""

    if len(series.steps) != len(generated):
        raise DataError("Synthetic and recorded world frame grids differ in length.")
    for step, frame in zip(series.steps, generated, strict=True):
        if step.frame_index != frame.frame_index:
            raise DataError("Synthetic and recorded world frame indices must match.")
        if frame.context.source is not ObservationSource.SYNTHETIC:
            raise DataError("World availability masking requires synthetic frames.")
        if step.status is PositionStatus.ACCEPTED:
            yield frame
        elif step.status is PositionStatus.SOURCE_POSE_MISSING:
            yield SpatialFrame(frame.context, frame.frame_index, None)
        else:
            yield SpatialFrame(
                frame.context,
                frame.frame_index,
                Pose2D((Keypoint2D("bodycentre", None, None),)),
            )


def _synthetic_measurements(
    series: RecordedWorldSeries,
    world: OpenFieldWorld,
    frames: Sequence[SpatialFrame],
    synthetic_protocol: SpatialTrajectoryProtocol,
) -> WorldMeasurements:
    return measure_world_steps(
        synthetic_protocol.trajectory_steps(_masked_frames(series, frames)),
        world=world,
    )


def _fit_models(
    fitting_subject_ids: tuple[str, ...],
    series: Mapping[str, RecordedWorldSeries],
) -> tuple[SpatialMovementModel, BoundaryConditionedMovementModel]:
    trajectories = tuple(series[subject_id].steps for subject_id in fitting_subject_ids)
    worlds = tuple(
        series[subject_id].world(WORLD_RECORDED_CONDITION_ID)
        for subject_id in fitting_subject_ids
    )
    baseline = fit_spatial_movement_model(
        trajectories, tuple(world.arena for world in worlds)
    )
    candidate = fit_boundary_conditioned_movement_model(trajectories, worlds)
    if candidate.shared_baseline != baseline:
        raise DataError("World models must share identical non-zone fitted parameters.")
    return baseline, candidate


def _positive_summary(
    values: tuple[float, ...],
) -> tuple[int, float | None, float | None]:
    if not values:
        return 0, None, None
    return len(values), fmean(values), float(median(values))


def _intervention_outcome(
    *,
    subject_id: str,
    replicate: int,
    seed: int,
    measured: WorldMeasurements,
    reflection_count: int,
) -> InterventionOutcome:
    counts = dict(measured.zone_valid_transition_counts)
    boundary = _positive_summary(
        measured.positive_displacements(WorldZone.BOUNDARY_BAND)
    )
    interior = _positive_summary(measured.positive_displacements(WorldZone.INTERIOR))
    values: tuple[tuple[str, float | int | None], ...] = (
        ("path_length_px", measured.spatial.path_length_px),
        (
            "valid_transition_count",
            measured.spatial.quality_control.valid_transition_count,
        ),
        ("boundary_band_occupancy_fraction", measured.boundary_band_occupancy_fraction),
        ("boundary_band_valid_transition_count", counts[WorldZone.BOUNDARY_BAND.value]),
        ("interior_valid_transition_count", counts[WorldZone.INTERIOR.value]),
        ("boundary_band_positive_displacement_count", boundary[0]),
        ("boundary_band_positive_displacement_mean_px", boundary[1]),
        ("boundary_band_positive_displacement_median_px", boundary[2]),
        ("interior_positive_displacement_count", interior[0]),
        ("interior_positive_displacement_mean_px", interior[1]),
        ("interior_positive_displacement_median_px", interior[2]),
        ("reflection_count", reflection_count),
    )
    return InterventionOutcome(
        subject_id=subject_id,
        replicate=replicate,
        seed=seed,
        condition_id=measured.world.condition.condition_id,
        values=values,
    )


def execute_world_experiment(
    protocol: WorldExperimentProtocol,
    series: Mapping[str, RecordedWorldSeries],
) -> WorldRunData:
    """Run frozen CV first, then separate legacy and synthetic interventions."""

    selected = ROCHE_CONTROL_FITTING_SUBJECTS + ROCHE_CONTROL_DEVELOPMENT_SUBJECTS
    if set(series) != set(selected):
        raise DataError(
            "The v0.3 experiment requires exactly the eight pinned controls."
        )
    synthetic_protocol = SpatialTrajectoryProtocol(
        keypoint_name="bodycentre",
        minimum_likelihood=None,
        required_source=ObservationSource.SYNTHETIC,
        required_coordinate_frame=series[selected[0]].arena.coordinate_frame,
    )
    recorded_measurements = {
        subject_id: measure_world_steps(
            series[subject_id].steps,
            world=series[subject_id].world(WORLD_RECORDED_CONDITION_ID),
        )
        for subject_id in selected
    }
    worlds_payload: dict[str, object] = {
        "world_id": series[selected[0]].world(WORLD_RECORDED_CONDITION_ID).world_id,
        "recordings": [
            {
                "subject_id": subject_id,
                "recording_id": series[subject_id].recording_id,
                "frame_count": len(series[subject_id].steps),
                "landmark_quality": series[subject_id].landmark_quality,
                "conditions": [
                    series[subject_id].world(condition.condition_id).to_dict()
                    for condition in V03_ARENA_SCALE_CONDITIONS
                    if subject_id in ROCHE_CONTROL_FITTING_SUBJECTS
                    or condition.condition_id == WORLD_RECORDED_CONDITION_ID
                ],
                "recorded_measurements": recorded_measurements[
                    subject_id
                ].compact_dict(),
            }
            for subject_id in selected
        ],
    }
    records: list[dict[str, object]] = []
    comparison_rows: list[WorldReplicateComparison] = []
    fold_models: list[dict[str, object]] = []
    generated = 0

    for fold in WORLD_CROSS_VALIDATION_FOLDS:
        baseline, candidate = _fit_models(fold.fitting_subject_ids, series)
        fold_models.append(
            {
                "evaluation_subject_id": fold.evaluation_subject_id,
                "fitting_subject_ids": list(fold.fitting_subject_ids),
                BASELINE_MODEL_ID: baseline.to_dict(),
                BOUNDARY_CONDITIONED_MODEL_ID: candidate.to_dict(),
            }
        )
        subject_id = fold.evaluation_subject_id
        source = series[subject_id]
        world = source.world(WORLD_RECORDED_CONDITION_ID)
        recorded = recorded_measurements[subject_id]
        indices = source.frame_indices
        for replicate in range(protocol.replicates):
            seed = protocol.seed(subject_id, replicate)
            for model_id in (BASELINE_MODEL_ID, BOUNDARY_CONDITIONED_MODEL_ID):
                recording_id = f"synthetic:v0.3:{subject_id}:{model_id}:{replicate}"
                if model_id == BASELINE_MODEL_ID:
                    frames = generate_spatial_trajectory(
                        baseline,
                        arena=world.arena,
                        frame_indices=indices,
                        subject_id=subject_id,
                        recording_id=recording_id,
                        seed=seed,
                    )
                else:
                    frames = generate_boundary_conditioned_trajectory(
                        candidate,
                        world=world,
                        frame_indices=indices,
                        subject_id=subject_id,
                        recording_id=recording_id,
                        seed=seed,
                    ).frames
                synthetic = _synthetic_measurements(
                    source, world, frames, synthetic_protocol
                )
                comparison = compare_world_measurements(recorded, synthetic)
                row = WorldReplicateComparison(
                    subject_id=subject_id,
                    model_id=model_id,
                    replicate=replicate,
                    seed=seed,
                    condition_id=WORLD_RECORDED_CONDITION_ID,
                    comparison=comparison,
                )
                comparison_rows.append(row)
                records.append(
                    {
                        "phase": "cross-validation",
                        "fold": fold.to_dict(),
                        "recording_id": source.recording_id,
                        **row.to_dict(),
                        "synthetic": synthetic.compact_dict(),
                    }
                )
                generated += 1

    preference = evaluate_world_model_preference(protocol, tuple(comparison_rows))
    final_baseline, final_candidate = _fit_models(
        ROCHE_CONTROL_FITTING_SUBJECTS, series
    )
    models_payload: dict[str, object] = {
        "cross_validation_folds": fold_models,
        "final_fit_subject_ids": list(ROCHE_CONTROL_FITTING_SUBJECTS),
        "final_models": {
            BASELINE_MODEL_ID: final_baseline.to_dict(),
            BOUNDARY_CONDITIONED_MODEL_ID: final_candidate.to_dict(),
        },
    }

    legacy_rows: list[LegacyDiagnosticComparison] = []
    for subject_id in ROCHE_CONTROL_DEVELOPMENT_SUBJECTS:
        source = series[subject_id]
        world = source.world(WORLD_RECORDED_CONDITION_ID)
        recorded = recorded_measurements[subject_id]
        indices = source.frame_indices
        for replicate in range(protocol.replicates):
            seed = protocol.seed(subject_id, replicate)
            for model_id in (BASELINE_MODEL_ID, BOUNDARY_CONDITIONED_MODEL_ID):
                recording_id = (
                    f"synthetic:v0.3:legacy:{subject_id}:{model_id}:{replicate}"
                )
                if model_id == BASELINE_MODEL_ID:
                    frames = generate_spatial_trajectory(
                        final_baseline,
                        arena=world.arena,
                        frame_indices=indices,
                        subject_id=subject_id,
                        recording_id=recording_id,
                        seed=seed,
                    )
                else:
                    frames = generate_boundary_conditioned_trajectory(
                        final_candidate,
                        world=world,
                        frame_indices=indices,
                        subject_id=subject_id,
                        recording_id=recording_id,
                        seed=seed,
                    ).frames
                synthetic = _synthetic_measurements(
                    source, world, frames, synthetic_protocol
                )
                comparison = compare_world_measurements(recorded, synthetic)
                legacy_rows.append(
                    LegacyDiagnosticComparison(
                        subject_id, model_id, replicate, seed, comparison
                    )
                )
                records.append(
                    {
                        "phase": "legacy-diagnostic",
                        "subject_id": subject_id,
                        "recording_id": source.recording_id,
                        "model_id": model_id,
                        "replicate": replicate,
                        "seed": seed,
                        "condition_id": WORLD_RECORDED_CONDITION_ID,
                        "synthetic": synthetic.compact_dict(),
                        "comparison": comparison.to_dict(),
                        "excluded_from_preference": True,
                    }
                )
                generated += 1

    intervention_outcomes: list[InterventionOutcome] = []
    for subject_id in ROCHE_CONTROL_FITTING_SUBJECTS:
        source = series[subject_id]
        indices = source.frame_indices
        for replicate in range(protocol.replicates):
            seed = protocol.seed(subject_id, replicate)
            for condition in V03_ARENA_SCALE_CONDITIONS:
                world = source.world(condition.condition_id)
                generated_trajectory = generate_boundary_conditioned_trajectory(
                    final_candidate,
                    world=world,
                    frame_indices=indices,
                    subject_id=subject_id,
                    recording_id=(
                        f"synthetic:v0.3:intervention:{subject_id}:"
                        f"{condition.condition_id}:{replicate}"
                    ),
                    seed=seed,
                )
                synthetic = _synthetic_measurements(
                    source, world, generated_trajectory.frames, synthetic_protocol
                )
                outcome = _intervention_outcome(
                    subject_id=subject_id,
                    replicate=replicate,
                    seed=seed,
                    measured=synthetic,
                    reflection_count=generated_trajectory.reflection_count,
                )
                intervention_outcomes.append(outcome)
                records.append(
                    {
                        "phase": "intervention",
                        "recording_id": source.recording_id,
                        "model_id": BOUNDARY_CONDITIONED_MODEL_ID,
                        **outcome.to_dict(),
                        "source_category": ObservationSource.SYNTHETIC.value,
                        "synthetic": synthetic.compact_dict(),
                    }
                )
                generated += 1

    return WorldRunData(
        worlds=worlds_payload,
        models=models_payload,
        records=tuple(records),
        comparison_rows=tuple(comparison_rows),
        legacy_rows=tuple(legacy_rows),
        preference=preference,
        intervention_outcomes=tuple(intervention_outcomes),
        generated_series_count=generated,
    )
