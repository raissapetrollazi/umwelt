"""Gap-aware derived trajectory measurements for Umwelt v0.2."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from math import atan2, isfinite, pi, sqrt

from umwelt.errors import DataError
from umwelt.spatial import AxisOrientation, Point2D, SpatialContext, SpatialFrame


@dataclass(frozen=True, slots=True)
class PositionRule:
    """Declared source identity and rule used to derive a position series."""

    context: SpatialContext
    keypoint_name: str
    minimum_likelihood: float | None = None

    def __post_init__(self) -> None:
        if not self.keypoint_name.strip():
            raise DataError("A trajectory keypoint name must be non-empty.")
        if self.minimum_likelihood is not None and (
            not isfinite(self.minimum_likelihood)
            or not 0.0 <= self.minimum_likelihood <= 1.0
        ):
            raise DataError("Minimum keypoint likelihood must be between 0 and 1.")


class PositionStatus(StrEnum):
    """Why a source coordinate is accepted or represented as a derived gap."""

    ACCEPTED = "accepted"
    SOURCE_POSE_MISSING = "source-pose-missing"
    SOURCE_COORDINATE_MISSING = "source-coordinate-missing"
    BELOW_LIKELIHOOD_THRESHOLD = "below-likelihood-threshold"
    LIKELIHOOD_MISSING = "likelihood-missing"


@dataclass(frozen=True, slots=True)
class PositionSample:
    """One derived position without discarding its source coordinate or rule."""

    rule: PositionRule
    frame_index: int
    source_point: Point2D | None
    confidence: float | None
    status: PositionStatus

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise DataError("Position-sample frame indices must be non-negative.")
        if self.confidence is not None and (
            not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0
        ):
            raise DataError("Position-sample confidence must be between 0 and 1.")
        if self.status is PositionStatus.ACCEPTED and self.source_point is None:
            raise DataError("An accepted position sample must have source coordinates.")
        if (
            self.status
            in (
                PositionStatus.BELOW_LIKELIHOOD_THRESHOLD,
                PositionStatus.LIKELIHOOD_MISSING,
            )
            and self.source_point is None
        ):
            raise DataError(
                "A likelihood-rejected sample must preserve source coordinates."
            )
        if self.status is PositionStatus.SOURCE_POSE_MISSING and (
            self.source_point is not None or self.confidence is not None
        ):
            raise DataError(
                "A missing source pose cannot contain keypoint measurements."
            )
        if (
            self.status is PositionStatus.SOURCE_COORDINATE_MISSING
            and self.source_point is not None
        ):
            raise DataError("A missing source coordinate cannot contain a point.")

    @property
    def point(self) -> Point2D | None:
        """Return accepted coordinates, or None for an explicit derived gap."""

        if self.status is PositionStatus.ACCEPTED:
            return self.source_point
        return None

    @property
    def context(self) -> SpatialContext:
        """Return the spatial identity and coordinate conventions of the sample."""

        return self.rule.context


@dataclass(frozen=True, slots=True)
class TrajectoryStep:
    """Derived movement quantities for one fully attributed position sample."""

    sample: PositionSample
    displacement: float | None
    movement_heading_radians: float | None
    turning_radians: float | None

    def __post_init__(self) -> None:
        for label, value in (
            ("displacement", self.displacement),
            ("movement heading", self.movement_heading_radians),
            ("turning angle", self.turning_radians),
        ):
            if value is not None and not isfinite(value):
                raise DataError(f"Trajectory {label} must be finite when present.")
        if self.displacement is not None and self.displacement < 0:
            raise DataError("Trajectory displacement must be non-negative.")

    @property
    def rule(self) -> PositionRule:
        return self.sample.rule

    @property
    def context(self) -> SpatialContext:
        return self.sample.context

    @property
    def frame_index(self) -> int:
        return self.sample.frame_index

    @property
    def source_point(self) -> Point2D | None:
        return self.sample.source_point

    @property
    def point(self) -> Point2D | None:
        return self.sample.point

    @property
    def confidence(self) -> float | None:
        return self.sample.confidence

    @property
    def status(self) -> PositionStatus:
        return self.sample.status


@dataclass(frozen=True, slots=True)
class TrajectoryPathLength:
    """Path length coupled to the source and derivation rule that define it."""

    rule: PositionRule
    value: float

    def __post_init__(self) -> None:
        if not isfinite(self.value) or self.value < 0:
            raise DataError("Trajectory path length must be finite and non-negative.")

    @property
    def context(self) -> SpatialContext:
        return self.rule.context

    @property
    def unit(self) -> str:
        """Return the coordinate unit used by the path-length value."""

        return self.context.coordinate_frame.unit


def _position_status(
    *,
    point: Point2D | None,
    confidence: float | None,
    minimum_likelihood: float | None,
) -> PositionStatus:
    if point is None:
        return PositionStatus.SOURCE_COORDINATE_MISSING
    if minimum_likelihood is None:
        return PositionStatus.ACCEPTED
    if confidence is None:
        return PositionStatus.LIKELIHOOD_MISSING
    if confidence < minimum_likelihood:
        return PositionStatus.BELOW_LIKELIHOOD_THRESHOLD
    return PositionStatus.ACCEPTED


def keypoint_position_samples(
    frames: Iterable[SpatialFrame],
    *,
    keypoint_name: str,
    minimum_likelihood: float | None = None,
) -> Iterator[PositionSample]:
    """Extract one keypoint while preserving source gaps and analytical rejection.

    No confidence threshold is assumed by default. A declared threshold rejects
    coordinates with a lower or missing likelihood, but the source coordinates,
    confidence, rejection reason, and complete derivation rule remain available.
    """

    rule: PositionRule | None = None
    for frame in frames:
        if rule is None:
            rule = PositionRule(
                context=frame.context,
                keypoint_name=keypoint_name,
                minimum_likelihood=minimum_likelihood,
            )
        elif frame.context != rule.context:
            raise DataError("Position extraction cannot mix spatial contexts.")

        if frame.pose is None:
            yield PositionSample(
                rule=rule,
                frame_index=frame.frame_index,
                source_point=None,
                confidence=None,
                status=PositionStatus.SOURCE_POSE_MISSING,
            )
            continue

        keypoint = frame.pose.keypoint(keypoint_name)
        yield PositionSample(
            rule=rule,
            frame_index=frame.frame_index,
            source_point=keypoint.point,
            confidence=keypoint.confidence,
            status=_position_status(
                point=keypoint.point,
                confidence=keypoint.confidence,
                minimum_likelihood=minimum_likelihood,
            ),
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


def _canonical_heading(dy: float, dx: float) -> float:
    heading = atan2(dy, dx)
    return pi if heading == -pi else heading


def iter_trajectory_steps(
    samples: Iterable[PositionSample],
) -> Iterator[TrajectoryStep]:
    """Derive displacement and canonical heading without crossing gaps.

    Displacement remains in the declared coordinate unit. Speed is not derived
    because v0.2's selected source has no verified sampling rate. Heading uses
    mathematical orientation: counterclockwise from +x in the interval
    ``(-pi, pi]``. It describes displacement, not anatomical body orientation.
    """

    previous_index: int | None = None
    previous_point: Point2D | None = None
    previous_heading: float | None = None
    rule: PositionRule | None = None

    for sample in samples:
        if rule is None:
            rule = sample.rule
        elif sample.rule != rule:
            raise DataError("A trajectory cannot mix position derivation rules.")
        if previous_index is not None and sample.frame_index <= previous_index:
            raise DataError(
                "Trajectory sample frame indices must be strictly increasing."
            )

        contiguous = (
            previous_index is not None and sample.frame_index == previous_index + 1
        )
        displacement: float | None = None
        heading: float | None = None
        turning: float | None = None

        if contiguous and previous_point is not None and sample.point is not None:
            dx = sample.point.x - previous_point.x
            dy = sample.point.y - previous_point.y
            displacement = sqrt(dx * dx + dy * dy)
            if displacement > 0:
                if (
                    sample.context.coordinate_frame.axis_orientation
                    is AxisOrientation.X_RIGHT_Y_DOWN
                ):
                    dy = -dy
                heading = _canonical_heading(dy, dx)
                if previous_heading is not None:
                    turning = _wrapped_angle_difference(heading, previous_heading)

        yield TrajectoryStep(
            sample=sample,
            displacement=displacement,
            movement_heading_radians=heading,
            turning_radians=turning,
        )

        previous_index = sample.frame_index
        previous_point = sample.point
        previous_heading = heading


def trajectory_path_length(steps: Iterable[TrajectoryStep]) -> TrajectoryPathLength:
    """Sum valid displacement without discarding source, rule, or units."""

    iterator = iter(steps)
    try:
        first = next(iterator)
    except StopIteration as error:
        raise DataError("Trajectory path length requires at least one step.") from error

    rule = first.rule
    total = first.displacement or 0.0
    for step in iterator:
        if step.rule != rule:
            raise DataError(
                "A trajectory path length cannot mix position derivation rules."
            )
        if step.displacement is not None:
            total += step.displacement
    return TrajectoryPathLength(rule=rule, value=total)
