"""Gap-aware derived trajectory measurements for Umwelt v0.2."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from math import atan2, isfinite, pi, sqrt

from umwelt.errors import DataError
from umwelt.spatial import Point2D, SpatialFrame


@dataclass(frozen=True, slots=True)
class PositionSample:
    """One frame-indexed representative position derived from a pose keypoint."""

    frame_index: int
    point: Point2D | None
    confidence: float | None

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise DataError("Position-sample frame indices must be non-negative.")
        if self.confidence is not None and (
            not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0
        ):
            raise DataError("Position-sample confidence must be between 0 and 1.")


@dataclass(frozen=True, slots=True)
class TrajectoryStep:
    """Derived pixel-space movement quantities for one position sample."""

    frame_index: int
    point: Point2D | None
    confidence: float | None
    displacement: float | None
    movement_heading_radians: float | None
    turning_radians: float | None

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise DataError("Trajectory-step frame indices must be non-negative.")
        for label, value in (
            ("displacement", self.displacement),
            ("movement heading", self.movement_heading_radians),
            ("turning angle", self.turning_radians),
        ):
            if value is not None and not isfinite(value):
                raise DataError(f"Trajectory {label} must be finite when present.")
        if self.displacement is not None and self.displacement < 0:
            raise DataError("Trajectory displacement must be non-negative.")


def keypoint_position_samples(
    frames: Iterable[SpatialFrame],
    *,
    keypoint_name: str,
    minimum_likelihood: float | None = None,
) -> Iterator[PositionSample]:
    """Extract one pose keypoint while keeping rejected/missing positions explicit.

    No confidence threshold is assumed by default. When a threshold is supplied
    explicitly, lower-confidence coordinates become derived gaps while their
    source confidence value is retained.
    """

    if not keypoint_name.strip():
        raise DataError("A trajectory keypoint name must be non-empty.")
    if minimum_likelihood is not None and (
        not isfinite(minimum_likelihood) or not 0.0 <= minimum_likelihood <= 1.0
    ):
        raise DataError("Minimum keypoint likelihood must be between 0 and 1.")

    for frame in frames:
        if frame.pose is None:
            yield PositionSample(frame.frame_index, None, None)
            continue
        keypoint = frame.pose.keypoint(keypoint_name)
        accepted = keypoint.point
        if (
            minimum_likelihood is not None
            and keypoint.confidence is not None
            and keypoint.confidence < minimum_likelihood
        ):
            accepted = None
        yield PositionSample(
            frame_index=frame.frame_index,
            point=accepted,
            confidence=keypoint.confidence,
        )


def bodycentre_samples(
    frames: Iterable[SpatialFrame],
    *,
    minimum_likelihood: float | None = None,
) -> Iterator[PositionSample]:
    """Use the recorded `bodycentre` DLC keypoint as representative position."""

    yield from keypoint_position_samples(
        frames,
        keypoint_name="bodycentre",
        minimum_likelihood=minimum_likelihood,
    )


def _wrapped_angle_difference(current: float, previous: float) -> float:
    difference = current - previous
    while difference <= -pi:
        difference += 2 * pi
    while difference > pi:
        difference -= 2 * pi
    return difference


def iter_trajectory_steps(samples: Iterable[PositionSample]) -> Iterator[TrajectoryStep]:
    """Derive displacement and movement direction without crossing gaps.

    Displacement remains in the position coordinate unit. This function does not
    derive speed because v0.2's selected source has no verified sampling rate.
    Movement heading is the direction of displacement, not anatomical body pose.
    """

    previous_index: int | None = None
    previous_point: Point2D | None = None
    previous_heading: float | None = None

    for sample in samples:
        if previous_index is not None and sample.frame_index <= previous_index:
            raise DataError("Trajectory sample frame indices must be strictly increasing.")

        contiguous = previous_index is not None and sample.frame_index == previous_index + 1
        displacement: float | None = None
        heading: float | None = None
        turning: float | None = None

        if contiguous and previous_point is not None and sample.point is not None:
            dx = sample.point.x - previous_point.x
            dy = sample.point.y - previous_point.y
            displacement = sqrt(dx * dx + dy * dy)
            if displacement > 0:
                heading = atan2(dy, dx)
                if previous_heading is not None:
                    turning = _wrapped_angle_difference(heading, previous_heading)

        yield TrajectoryStep(
            frame_index=sample.frame_index,
            point=sample.point,
            confidence=sample.confidence,
            displacement=displacement,
            movement_heading_radians=heading,
            turning_radians=turning,
        )

        previous_index = sample.frame_index
        previous_point = sample.point
        previous_heading = heading


def trajectory_path_length(steps: Iterable[TrajectoryStep]) -> float:
    """Sum valid adjacent-frame displacements in the coordinate frame's unit."""

    return sum(step.displacement for step in steps if step.displacement is not None)
