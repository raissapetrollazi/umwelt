"""Boundary-conditioned movement model for the Umwelt v0.3 experiment."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, exp, floor, isfinite, log, pi, sin
from random import Random
from statistics import fmean, pstdev

from umwelt.errors import DataError
from umwelt.observations import ObservationSource
from umwelt.spatial import (
    AxisOrientation,
    Keypoint2D,
    Point2D,
    Pose2D,
    SpatialContext,
    SpatialFrame,
)
from umwelt.spatial_model import (
    SpatialMovementModel,
    _reflect,
    fit_spatial_movement_model,
)
from umwelt.trajectory import TrajectoryStep
from umwelt.world import (
    V03_BOUNDARY_BAND_FRACTION,
    ArenaScaleCondition,
    OpenFieldWorld,
    WorldZone,
)
from umwelt.world_evaluation import measure_world_steps


BOUNDARY_CONDITIONED_MODEL_ID = "boundary-conditioned-random-walk-v1"
MINIMUM_POSITIVE_DISPLACEMENTS_PER_ZONE = 2
RECORDED_WORLD_CONDITION = ArenaScaleCondition("recorded", 1.0)


@dataclass(frozen=True, slots=True)
class ZoneDisplacementParameters:
    """One fitted positive-displacement distribution for a geometric zone."""

    zone: WorldZone
    sample_count: int
    log_mean: float
    log_std: float

    def __post_init__(self) -> None:
        if not isinstance(self.zone, WorldZone):
            raise DataError("Zone displacement parameters require a declared zone.")
        if self.sample_count < MINIMUM_POSITIVE_DISPLACEMENTS_PER_ZONE:
            raise DataError(
                "Zone displacement fitting requires at least "
                f"{MINIMUM_POSITIVE_DISPLACEMENTS_PER_ZONE} positive samples."
            )
        if not isfinite(self.log_mean) or not isfinite(self.log_std):
            raise DataError("Zone displacement parameters must be finite.")
        if self.log_std < 0:
            raise DataError("Zone displacement dispersion must be non-negative.")

    def to_dict(self) -> dict[str, object]:
        return {
            "family": "lognormal",
            "sample_count": self.sample_count,
            "log_mean": self.log_mean,
            "log_std": self.log_std,
        }


@dataclass(frozen=True, slots=True)
class BoundaryConditionedMovementModel:
    """The v0.2 baseline with movement magnitude conditioned on world zone."""

    shared_baseline: SpatialMovementModel
    zone_displacements: tuple[ZoneDisplacementParameters, ...]
    excluded_outside_position_count: int = 0
    excluded_positive_transition_start_count: int = 0
    boundary_band_fraction: float = V03_BOUNDARY_BAND_FRACTION
    model_id: str = BOUNDARY_CONDITIONED_MODEL_ID

    def __post_init__(self) -> None:
        if self.model_id != BOUNDARY_CONDITIONED_MODEL_ID:
            raise DataError("Boundary-conditioned model identity is frozen.")
        if self.shared_baseline.model_id != "persistent-reflecting-random-walk-v1":
            raise DataError("Boundary-conditioned fitting requires the v0.2 baseline.")
        if self.boundary_band_fraction != V03_BOUNDARY_BAND_FRACTION:
            raise DataError("Boundary-conditioned zone fraction is frozen at 0.10.")
        if (
            self.excluded_outside_position_count < 0
            or self.excluded_positive_transition_start_count < 0
        ):
            raise DataError("Zone-fitting exclusion counts must be non-negative.")
        expected = tuple(WorldZone)
        if tuple(parameters.zone for parameters in self.zone_displacements) != expected:
            raise DataError(
                "Boundary-conditioned model must include every zone in protocol order."
            )

    def displacement_parameters(self, zone: WorldZone) -> ZoneDisplacementParameters:
        """Return the declared positive-displacement distribution for one zone."""

        if not isinstance(zone, WorldZone):
            raise DataError("Displacement selection requires a declared world zone.")
        for parameters in self.zone_displacements:
            if parameters.zone is zone:
                return parameters
        raise AssertionError("Validated model is missing a world zone.")

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "environmental_dependency": {
                "input": "transition-start-zone",
                "boundary_band_fraction": self.boundary_band_fraction,
                "affected_parameter": "positive-displacement",
            },
            "positive_displacement_by_zone": {
                parameters.zone.value: parameters.to_dict()
                for parameters in self.zone_displacements
            },
            "shared_with_baseline": {
                "model_id": self.shared_baseline.model_id,
                "stationary_probability": (self.shared_baseline.stationary_probability),
                "turning": {
                    "family": "zero-mean-normal-radians",
                    "std": self.shared_baseline.turning_std_radians,
                },
                "boundary_rule": "repeated-axis-reflection",
                "initial_position_fraction": [
                    self.shared_baseline.initial_x_fraction,
                    self.shared_baseline.initial_y_fraction,
                ],
            },
            "minimum_positive_displacements_per_zone": (
                MINIMUM_POSITIVE_DISPLACEMENTS_PER_ZONE
            ),
            "zone_fitting_quality_control": {
                "outside_recorded_position_count": (
                    self.excluded_outside_position_count
                ),
                "unclassified_positive_transition_start_count": (
                    self.excluded_positive_transition_start_count
                ),
                "outside_positions_retained_in_shared_baseline": True,
            },
        }


@dataclass(frozen=True, slots=True)
class GeneratedWorldTrajectory:
    """Synthetic frames bound to the world condition that generated them."""

    world: OpenFieldWorld
    frames: tuple[SpatialFrame, ...]
    seed: int
    reflection_count: int

    def __post_init__(self) -> None:
        if not self.frames:
            raise DataError("A generated world trajectory must contain frames.")
        if self.reflection_count < 0:
            raise DataError("Generated reflection count must be non-negative.")
        context = self.frames[0].context
        if context.source is not ObservationSource.SYNTHETIC:
            raise DataError("Generated world trajectories must be synthetic.")
        if context.coordinate_frame != self.world.arena.coordinate_frame:
            raise DataError("Generated trajectory and world must share a frame.")
        for frame in self.frames:
            if frame.context != context:
                raise DataError("Generated world trajectory cannot mix contexts.")
            if frame.pose is None:
                continue
            point = frame.pose.keypoint("bodycentre").point
            if point is not None and not self.world.arena.contains(point):
                raise DataError("Generated positions must remain inside their world.")

    @property
    def context(self) -> SpatialContext:
        return self.frames[0].context


def _axis_reflections(value: float, minimum: float, maximum: float) -> int:
    """Count crossed axis boundaries for one attempted movement endpoint."""

    if not isfinite(value):
        raise DataError("Synthetic attempted movement must remain finite.")
    width = maximum - minimum
    if value > maximum:
        return floor((value - maximum) / width) + 1
    if value < minimum:
        return floor((minimum - value) / width) + 1
    return 0


def _fit_zone_parameters(
    samples: dict[WorldZone, list[float]],
) -> tuple[ZoneDisplacementParameters, ...]:
    parameters: list[ZoneDisplacementParameters] = []
    for zone in WorldZone:
        values = samples[zone]
        if len(values) < MINIMUM_POSITIVE_DISPLACEMENTS_PER_ZONE:
            raise DataError(
                f"Zone {zone.value} requires at least "
                f"{MINIMUM_POSITIVE_DISPLACEMENTS_PER_ZONE} positive displacements."
            )
        logs = [log(value) for value in values]
        parameters.append(
            ZoneDisplacementParameters(
                zone=zone,
                sample_count=len(values),
                log_mean=fmean(logs),
                log_std=pstdev(logs),
            )
        )
    return tuple(parameters)


def fit_boundary_conditioned_movement_model(
    trajectories: tuple[tuple[TrajectoryStep, ...], ...],
    worlds: tuple[OpenFieldWorld, ...],
) -> BoundaryConditionedMovementModel:
    """Fit zone-specific movement magnitude from recorded development-pool data."""

    if not trajectories or len(trajectories) != len(worlds):
        raise DataError(
            "Boundary-conditioned fitting requires one world per trajectory."
        )
    samples: dict[WorldZone, list[float]] = {zone: [] for zone in WorldZone}
    outside_position_count = 0
    unclassified_positive_start_count = 0

    for steps, world in zip(trajectories, worlds, strict=True):
        if not steps:
            raise DataError("Boundary-conditioned trajectories must not be empty.")
        if steps[0].context.source is not ObservationSource.RECORDED:
            raise DataError("Boundary-conditioned fitting requires recorded data.")
        if world.condition != RECORDED_WORLD_CONDITION:
            raise DataError(
                "Model fitting requires the recorded-scale world condition."
            )
        if world.boundary_band_fraction != V03_BOUNDARY_BAND_FRACTION:
            raise DataError("Model fitting requires the frozen boundary-zone fraction.")
        measured = measure_world_steps(steps, world=world)
        outside_position_count += measured.outside_arena_position_count
        unclassified_positive_start_count += (
            measured.unclassified_positive_transition_start_count
        )
        for zone in WorldZone:
            samples[zone].extend(measured.positive_displacements(zone))

    baseline = fit_spatial_movement_model(
        trajectories,
        tuple(world.arena for world in worlds),
    )
    return BoundaryConditionedMovementModel(
        shared_baseline=baseline,
        zone_displacements=_fit_zone_parameters(samples),
        excluded_outside_position_count=outside_position_count,
        excluded_positive_transition_start_count=unclassified_positive_start_count,
    )


def generate_boundary_conditioned_trajectory(
    model: BoundaryConditionedMovementModel,
    *,
    world: OpenFieldWorld,
    frame_indices: tuple[int, ...],
    subject_id: str,
    recording_id: str,
    seed: int,
) -> GeneratedWorldTrajectory:
    """Generate a seeded synthetic trajectory bound to one world condition."""

    if not frame_indices:
        raise DataError("Synthetic world generation requires at least one frame index.")
    if tuple(sorted(set(frame_indices))) != frame_indices:
        raise DataError(
            "Synthetic frame indices must be strictly increasing and unique."
        )
    if world.boundary_band_fraction != model.boundary_band_fraction:
        raise DataError("Synthetic world and model must share one zone definition.")

    arena = world.arena
    baseline = model.shared_baseline
    rng = Random(seed)
    context = SpatialContext(
        subject_id,
        recording_id,
        ObservationSource.SYNTHETIC,
        arena.coordinate_frame,
    )
    point = Point2D(
        arena.minimum.x + baseline.initial_x_fraction * arena.width,
        arena.minimum.y + baseline.initial_y_fraction * arena.height,
    )
    heading = rng.uniform(-pi, pi)
    frames: list[SpatialFrame] = []
    previous_index: int | None = None
    reflection_count = 0

    for frame_index in frame_indices:
        if previous_index is not None and frame_index == previous_index + 1:
            if rng.random() >= baseline.stationary_probability:
                parameters = model.displacement_parameters(world.zone(point))
                displacement = exp(rng.gauss(parameters.log_mean, parameters.log_std))
                heading += rng.gauss(0.0, baseline.turning_std_radians)
                dx = displacement * cos(heading)
                dy_math = displacement * sin(heading)
                dy_coordinate = (
                    -dy_math
                    if arena.coordinate_frame.axis_orientation
                    is AxisOrientation.X_RIGHT_Y_DOWN
                    else dy_math
                )
                reflection_count += _axis_reflections(
                    point.x + dx, arena.minimum.x, arena.maximum.x
                )
                reflection_count += _axis_reflections(
                    point.y + dy_coordinate, arena.minimum.y, arena.maximum.y
                )
                x, x_direction = _reflect(
                    point.x + dx,
                    arena.minimum.x,
                    arena.maximum.x,
                )
                y, y_direction = _reflect(
                    point.y + dy_coordinate,
                    arena.minimum.y,
                    arena.maximum.y,
                )
                point = Point2D(x, y)
                heading = atan2(
                    sin(heading) * y_direction,
                    cos(heading) * x_direction,
                )
        pose = Pose2D((Keypoint2D("bodycentre", point, None),))
        frames.append(SpatialFrame(context, frame_index, pose))
        previous_index = frame_index

    return GeneratedWorldTrajectory(
        world=world,
        frames=tuple(frames),
        seed=seed,
        reflection_count=reflection_count,
    )
