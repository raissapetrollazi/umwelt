"""Compact seeded generative movement baseline for Umwelt v0.2."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, exp, isfinite, log, pi, sin, sqrt
from random import Random
from statistics import fmean, pstdev

from umwelt.arena import RectangularArena
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
from umwelt.trajectory import TrajectoryStep


@dataclass(frozen=True, slots=True)
class SpatialMovementModel:
    model_id: str
    stationary_probability: float
    log_displacement_mean: float
    log_displacement_std: float
    turning_std_radians: float
    initial_x_fraction: float
    initial_y_fraction: float

    def __post_init__(self) -> None:
        values = (
            self.stationary_probability,
            self.log_displacement_mean,
            self.log_displacement_std,
            self.turning_std_radians,
            self.initial_x_fraction,
            self.initial_y_fraction,
        )
        if not all(isfinite(value) for value in values):
            raise DataError("Spatial movement model parameters must be finite.")
        if not 0.0 <= self.stationary_probability <= 1.0:
            raise DataError("Stationary probability must be in [0, 1].")
        if self.log_displacement_std < 0 or self.turning_std_radians < 0:
            raise DataError("Spatial movement dispersions must be non-negative.")
        if (
            not 0.0 <= self.initial_x_fraction <= 1.0
            or not 0.0 <= self.initial_y_fraction <= 1.0
        ):
            raise DataError("Initial spatial fractions must be in [0, 1].")

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "stationary_probability": self.stationary_probability,
            "positive_displacement": {
                "family": "lognormal",
                "log_mean": self.log_displacement_mean,
                "log_std": self.log_displacement_std,
            },
            "turning": {
                "family": "zero-mean-normal-radians",
                "std": self.turning_std_radians,
            },
            "boundary_rule": "repeated-axis-reflection",
            "initial_position_fraction": [
                self.initial_x_fraction,
                self.initial_y_fraction,
            ],
        }


def fit_spatial_movement_model(
    trajectories: tuple[tuple[TrajectoryStep, ...], ...],
    arenas: tuple[RectangularArena, ...],
) -> SpatialMovementModel:
    """Fit pooled control-only movement statistics without development data."""

    if not trajectories or len(trajectories) != len(arenas):
        raise DataError(
            "Spatial fitting requires one arena for each non-empty trajectory."
        )
    displacements: list[float] = []
    positive: list[float] = []
    turns: list[float] = []
    x_fractions: list[float] = []
    y_fractions: list[float] = []

    for steps, arena in zip(trajectories, arenas, strict=True):
        if not steps:
            raise DataError("Spatial fitting trajectories must not be empty.")
        if steps[0].context.coordinate_frame != arena.coordinate_frame:
            raise DataError(
                "Spatial fitting trajectories and arenas must share a coordinate frame."
            )
        initial_position_recorded = False
        for step in steps:
            if step.displacement is not None:
                displacements.append(step.displacement)
                if step.displacement > 0:
                    positive.append(step.displacement)
            if step.turning_radians is not None:
                turns.append(step.turning_radians)
            if (
                not initial_position_recorded
                and step.point is not None
                and arena.contains(step.point)
            ):
                x_fractions.append((step.point.x - arena.minimum.x) / arena.width)
                y_fractions.append((step.point.y - arena.minimum.y) / arena.height)
                initial_position_recorded = True

    if not displacements or not positive or not x_fractions:
        raise DataError("Spatial fitting needs valid movements and in-arena positions.")
    logs = [log(value) for value in positive]
    stationary = sum(value == 0 for value in displacements) / len(displacements)
    turning_std = sqrt(fmean(value * value for value in turns)) if turns else 0.0
    return SpatialMovementModel(
        model_id="persistent-reflecting-random-walk-v1",
        stationary_probability=stationary,
        log_displacement_mean=fmean(logs),
        log_displacement_std=pstdev(logs) if len(logs) > 1 else 0.0,
        turning_std_radians=turning_std,
        initial_x_fraction=fmean(x_fractions),
        initial_y_fraction=fmean(y_fractions),
    )


def _reflect(value: float, minimum: float, maximum: float) -> tuple[float, int]:
    """Reflect any finite overshoot and return the resulting direction sign."""

    width = maximum - minimum
    phase = (value - minimum) % (2 * width)
    if phase <= width:
        return minimum + phase, 1
    return maximum - (phase - width), -1


def generate_spatial_trajectory(
    model: SpatialMovementModel,
    *,
    arena: RectangularArena,
    frame_indices: tuple[int, ...],
    subject_id: str,
    recording_id: str,
    seed: int,
) -> tuple[SpatialFrame, ...]:
    """Generate a new complete synthetic trajectory on an explicit frame grid."""

    if not frame_indices:
        raise DataError(
            "Synthetic spatial generation requires at least one frame index."
        )
    if tuple(sorted(set(frame_indices))) != frame_indices:
        raise DataError(
            "Synthetic frame indices must be strictly increasing and unique."
        )
    rng = Random(seed)
    context = SpatialContext(
        subject_id,
        recording_id,
        ObservationSource.SYNTHETIC,
        arena.coordinate_frame,
    )
    point = Point2D(
        arena.minimum.x + model.initial_x_fraction * arena.width,
        arena.minimum.y + model.initial_y_fraction * arena.height,
    )
    heading = rng.uniform(-pi, pi)
    frames: list[SpatialFrame] = []
    previous_index: int | None = None

    for frame_index in frame_indices:
        if previous_index is not None and frame_index == previous_index + 1:
            if rng.random() >= model.stationary_probability:
                displacement = exp(
                    rng.gauss(
                        model.log_displacement_mean,
                        model.log_displacement_std,
                    )
                )
                heading += rng.gauss(0.0, model.turning_std_radians)
                dx = displacement * cos(heading)
                dy_math = displacement * sin(heading)
                dy_coordinate = (
                    -dy_math
                    if arena.coordinate_frame.axis_orientation
                    is AxisOrientation.X_RIGHT_Y_DOWN
                    else dy_math
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
    return tuple(frames)
