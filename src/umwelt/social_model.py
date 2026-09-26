"""Frozen first independent and distance-directed dyad generators."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import hypot, isfinite, sqrt
from random import Random

from umwelt.calms21 import CALMS21_COORDINATE_FRAME
from umwelt.dyad import DyadFrame
from umwelt.errors import DataError
from umwelt.observations import ObservationSource
from umwelt.spatial import (
    Keypoint2D,
    Point2D,
    Pose2D,
    SpatialContext,
    SpatialFrame,
)


SOCIAL_MODEL_IDS = ("S0-independent", "S1-resident-directional-drift")
FITTING_STEP_LIMIT_PX = 50.0
IMAGE_WIDTH_PX = 1024.0
IMAGE_HEIGHT_PX = 570.0


@dataclass(slots=True)
class _RoleMoments:
    eligible_triplets: int = 0
    excluded_large_step_triplets: int = 0
    excluded_missing_or_gap_triplets: int = 0
    sum_previous_squared: float = 0.0
    sum_next_squared: float = 0.0
    sum_cross: float = 0.0

    def add(self, previous: Point2D, following: Point2D) -> None:
        self.eligible_triplets += 1
        self.sum_previous_squared += previous.x**2 + previous.y**2
        self.sum_next_squared += following.x**2 + following.y**2
        self.sum_cross += previous.x * following.x + previous.y * following.y

    def parameters(self) -> tuple[float, float]:
        if self.eligible_triplets == 0 or self.sum_previous_squared <= 0:
            raise DataError("Social movement fitting has no usable role triplets.")
        alpha = min(0.98, max(0.0, self.sum_cross / self.sum_previous_squared))
        squared_error = (
            self.sum_next_squared
            - 2 * alpha * self.sum_cross
            + alpha * alpha * self.sum_previous_squared
        )
        sigma = sqrt(max(0.0, squared_error) / (2 * self.eligible_triplets))
        return alpha, sigma


@dataclass(frozen=True, slots=True)
class SocialMovementFit:
    resident_alpha: float
    resident_sigma_px: float
    intruder_alpha: float
    intruder_sigma_px: float
    resident_directional_drift_px: float
    fitting_sequence_count: int
    resident_eligible_triplets: int
    intruder_eligible_triplets: int
    resident_excluded_large_step_triplets: int
    intruder_excluded_large_step_triplets: int
    resident_excluded_missing_or_gap_triplets: int
    intruder_excluded_missing_or_gap_triplets: int
    directional_drift_triplets: int

    def __post_init__(self) -> None:
        parameters = (
            self.resident_alpha,
            self.resident_sigma_px,
            self.intruder_alpha,
            self.intruder_sigma_px,
            self.resident_directional_drift_px,
        )
        if not all(isfinite(value) for value in parameters):
            raise DataError("Social model parameters must be finite.")
        if not (0 <= self.resident_alpha <= 0.98 and 0 <= self.intruder_alpha <= 0.98):
            raise DataError("Social movement persistence must be in [0, 0.98].")
        if self.resident_sigma_px < 0 or self.intruder_sigma_px < 0:
            raise DataError("Social innovation scales must be non-negative.")
        if self.fitting_sequence_count < 1:
            raise DataError("Social model fitting needs at least one sequence.")

    def to_dict(self) -> dict[str, object]:
        return {
            "model_ids": list(SOCIAL_MODEL_IDS),
            "resident_alpha": self.resident_alpha,
            "resident_sigma_px": self.resident_sigma_px,
            "intruder_alpha": self.intruder_alpha,
            "intruder_sigma_px": self.intruder_sigma_px,
            "resident_directional_drift_px": self.resident_directional_drift_px,
            "fitting_step_limit_px": FITTING_STEP_LIMIT_PX,
            "fitting_sequence_count": self.fitting_sequence_count,
            "resident_eligible_triplets": self.resident_eligible_triplets,
            "intruder_eligible_triplets": self.intruder_eligible_triplets,
            "resident_excluded_large_step_triplets": (
                self.resident_excluded_large_step_triplets
            ),
            "intruder_excluded_large_step_triplets": (
                self.intruder_excluded_large_step_triplets
            ),
            "resident_excluded_missing_or_gap_triplets": (
                self.resident_excluded_missing_or_gap_triplets
            ),
            "intruder_excluded_missing_or_gap_triplets": (
                self.intruder_excluded_missing_or_gap_triplets
            ),
            "directional_drift_triplets": self.directional_drift_triplets,
        }


def _difference(left: Point2D, right: Point2D) -> Point2D:
    return Point2D(right.x - left.x, right.y - left.y)


def fit_social_movement_model(
    sequences: Iterable[Iterable[DyadFrame]],
) -> SocialMovementFit:
    """Fit both model components on fitting animals only."""

    resident = _RoleMoments()
    intruder = _RoleMoments()
    social_projection_next = 0.0
    social_projection_previous = 0.0
    social_count = 0
    sequence_count = 0

    for sequence in sequences:
        history: list[DyadFrame] = []
        first: DyadFrame | None = None
        seen = False
        for frame in sequence:
            if frame.source is not ObservationSource.RECORDED:
                raise DataError("Social fitting requires recorded source frames.")
            if frame.coordinate_frame != CALMS21_COORDINATE_FRAME:
                raise DataError("Social fitting requires the CalMS21 image frame.")
            if first is None:
                first = frame
            elif (
                frame.pair_id != first.pair_id
                or frame.resident.context != first.resident.context
                or frame.intruder.context != first.intruder.context
            ):
                raise DataError("Social fitting cannot mix pair contexts.")
            if history and frame.frame_index <= history[-1].frame_index:
                raise DataError("Social fitting frame indices must increase.")
            seen = True
            history.append(frame)
            if len(history) < 3:
                continue
            if len(history) > 3:
                history.pop(0)
            previous, current, following = history
            adjacent = (
                current.frame_index == previous.frame_index + 1
                and following.frame_index == current.frame_index + 1
            )
            positions = [item.neck_positions() for item in history]
            for role_index, moments in ((0, resident), (1, intruder)):
                a, b, c = (pair[role_index] for pair in positions)
                if not adjacent or a is None or b is None or c is None:
                    moments.excluded_missing_or_gap_triplets += 1
                    continue
                prior_velocity = _difference(a, b)
                next_velocity = _difference(b, c)
                if (
                    hypot(prior_velocity.x, prior_velocity.y) > FITTING_STEP_LIMIT_PX
                    or hypot(next_velocity.x, next_velocity.y) > FITTING_STEP_LIMIT_PX
                ):
                    moments.excluded_large_step_triplets += 1
                    continue
                moments.add(prior_velocity, next_velocity)
                if role_index == 0:
                    other = positions[1][1]
                    if other is not None:
                        bearing = _difference(b, other)
                        distance = hypot(bearing.x, bearing.y)
                        if distance > 0:
                            unit_x, unit_y = bearing.x / distance, bearing.y / distance
                            social_projection_previous += (
                                prior_velocity.x * unit_x + prior_velocity.y * unit_y
                            )
                            social_projection_next += (
                                next_velocity.x * unit_x + next_velocity.y * unit_y
                            )
                            social_count += 1
        if not seen:
            raise DataError("Social fitting sequence cannot be empty.")
        sequence_count += 1

    resident_alpha, resident_sigma = resident.parameters()
    intruder_alpha, intruder_sigma = intruder.parameters()
    if social_count == 0:
        raise DataError("Social fitting has no usable directional observations.")
    beta = (
        social_projection_next - resident_alpha * social_projection_previous
    ) / social_count
    return SocialMovementFit(
        resident_alpha=resident_alpha,
        resident_sigma_px=resident_sigma,
        intruder_alpha=intruder_alpha,
        intruder_sigma_px=intruder_sigma,
        resident_directional_drift_px=beta,
        fitting_sequence_count=sequence_count,
        resident_eligible_triplets=resident.eligible_triplets,
        intruder_eligible_triplets=intruder.eligible_triplets,
        resident_excluded_large_step_triplets=(resident.excluded_large_step_triplets),
        intruder_excluded_large_step_triplets=(intruder.excluded_large_step_triplets),
        resident_excluded_missing_or_gap_triplets=(
            resident.excluded_missing_or_gap_triplets
        ),
        intruder_excluded_missing_or_gap_triplets=(
            intruder.excluded_missing_or_gap_triplets
        ),
        directional_drift_triplets=social_count,
    )


def social_role_seed(
    master_seed: int, sequence_id: str, replicate: int, role: str
) -> int:
    """Derive a stable seed without depending on Python's process hash."""

    if role not in ("resident", "intruder") or replicate < 0:
        raise DataError("Social seed requires a known role and replicate.")
    payload = f"umwelt-v0.4|{master_seed}|{sequence_id}|{replicate}|{role}"
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _reflect(value: float, maximum: float) -> tuple[float, int]:
    phase = value % (2 * maximum)
    if phase <= maximum:
        return phase, 1
    return 2 * maximum - phase, -1


def generate_social_dyad(
    fit: SocialMovementFit,
    template: Sequence[DyadFrame],
    *,
    model_id: str,
    master_seed: int,
    replicate: int,
) -> tuple[DyadFrame, ...]:
    """Generate one paired, mask-aligned trajectory from a first-frame state."""

    if model_id not in SOCIAL_MODEL_IDS:
        raise DataError("Unknown first social model identifier.")
    if not template:
        raise DataError("Social generation needs a non-empty template.")
    first = template[0]
    start = first.neck_positions()
    if start[0] is None or start[1] is None:
        raise DataError("Social generation requires a valid first pair position.")
    sequence_id = first.resident.context.recording_id
    if first.source is not ObservationSource.RECORDED:
        raise DataError("Social generation template must be recorded.")
    if first.coordinate_frame != CALMS21_COORDINATE_FRAME:
        raise DataError("Social generation requires CalMS21 image coordinates.")
    contexts = (
        SpatialContext(
            first.resident.context.subject_id,
            sequence_id,
            ObservationSource.SYNTHETIC,
            CALMS21_COORDINATE_FRAME,
        ),
        SpatialContext(
            first.intruder.context.subject_id,
            sequence_id,
            ObservationSource.SYNTHETIC,
            CALMS21_COORDINATE_FRAME,
        ),
    )
    generators = (
        Random(social_role_seed(master_seed, sequence_id, replicate, "resident")),
        Random(social_role_seed(master_seed, sequence_id, replicate, "intruder")),
    )
    points = [start[0], start[1]]
    velocities = [Point2D(0, 0), Point2D(0, 0)]
    generated: list[DyadFrame] = []

    for index, source_frame in enumerate(template):
        if source_frame.pair_id != first.pair_id or (
            source_frame.resident.context != first.resident.context
            or source_frame.intruder.context != first.intruder.context
        ):
            raise DataError("Social generation template contexts must agree.")
        if index and source_frame.frame_index != template[index - 1].frame_index + 1:
            raise DataError("Social generation requires adjacent frame indices.")
        if index:
            bearing = _difference(points[0], points[1])
            pair_distance = hypot(bearing.x, bearing.y)
            unit = (
                Point2D(bearing.x / pair_distance, bearing.y / pair_distance)
                if pair_distance > 0
                else Point2D(0, 0)
            )
            next_points: list[Point2D] = []
            next_velocities: list[Point2D] = []
            for role_index, rng in enumerate(generators):
                alpha = fit.resident_alpha if role_index == 0 else fit.intruder_alpha
                sigma = (
                    fit.resident_sigma_px if role_index == 0 else fit.intruder_sigma_px
                )
                beta = (
                    fit.resident_directional_drift_px
                    if role_index == 0 and model_id == SOCIAL_MODEL_IDS[1]
                    else 0.0
                )
                vx = alpha * velocities[role_index].x + rng.gauss(0, sigma)
                vy = alpha * velocities[role_index].y + rng.gauss(0, sigma)
                if role_index == 0:
                    vx += beta * unit.x
                    vy += beta * unit.y
                x, x_sign = _reflect(points[role_index].x + vx, IMAGE_WIDTH_PX)
                y, y_sign = _reflect(points[role_index].y + vy, IMAGE_HEIGHT_PX)
                next_points.append(Point2D(x, y))
                next_velocities.append(Point2D(vx * x_sign, vy * y_sign))
            points = next_points
            velocities = next_velocities

        masks = source_frame.neck_positions()
        members = tuple(
            SpatialFrame(
                contexts[role_index],
                source_frame.frame_index,
                Pose2D(
                    (
                        Keypoint2D(
                            "neck",
                            points[role_index]
                            if masks[role_index] is not None
                            else None,
                        ),
                    )
                ),
            )
            for role_index in (0, 1)
        )
        generated.append(DyadFrame(source_frame.pair_id, members[0], members[1]))
    return tuple(generated)
