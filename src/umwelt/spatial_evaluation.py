"""Pre-declared spatial measurements and recorded/synthetic comparison for Umwelt v0.2."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import isfinite
from statistics import fmean, median

from umwelt.arena import RectangularArena
from umwelt.errors import DataError
from umwelt.spatial import Point2D, SpatialContext, SpatialFrame
from umwelt.spatial_protocol import SpatialTrajectoryProtocol


@dataclass(frozen=True, slots=True)
class SpatialEvaluationProtocol:
    """Geometric evaluation choices frozen before inspecting baseline results."""

    center_margin_fraction: float = 0.25
    occupancy_grid_size: int = 4

    def __post_init__(self) -> None:
        if (
            not isfinite(self.center_margin_fraction)
            or not 0.0 <= self.center_margin_fraction < 0.5
        ):
            raise DataError("Spatial center margin fraction must be in [0, 0.5).")
        if not 2 <= self.occupancy_grid_size <= 32:
            raise DataError("Spatial occupancy grid size must be between 2 and 32.")

    def to_dict(self) -> dict[str, object]:
        return {
            "arena_geometry": "median-axis-aligned-roche-landmarks",
            "center_margin_fraction": self.center_margin_fraction,
            "occupancy_grid_size": self.occupancy_grid_size,
            "metrics": [
                "path_length",
                "displacement_wasserstein",
                "turning_wasserstein",
                "boundary_distance_wasserstein",
                "center_fraction_absolute_difference",
                "occupancy_total_variation",
                "outside_fraction_absolute_difference",
            ],
            "speed_included": False,
            "movement_autocorrelation_included": False,
        }


ROCHE_SPATIAL_EVALUATION_PROTOCOL = SpatialEvaluationProtocol()


@dataclass(frozen=True, slots=True)
class SpatialMeasurements:
    """Measurements for one trajectory, retaining raw 1D samples only in memory."""

    context: SpatialContext
    total_samples: int
    accepted_positions: int
    in_arena_positions: int
    outside_positions: int
    path_length: float
    displacements: tuple[float, ...]
    turnings_radians: tuple[float, ...]
    boundary_distances: tuple[float, ...]
    center_fraction: float | None
    occupancy: tuple[float, ...]

    @property
    def accepted_fraction(self) -> float:
        return self.accepted_positions / self.total_samples if self.total_samples else 0.0

    @property
    def outside_fraction(self) -> float:
        return (
            self.outside_positions / self.accepted_positions
            if self.accepted_positions
            else 0.0
        )

    def compact_dict(self) -> dict[str, object]:
        return {
            "subject_id": self.context.subject_id,
            "recording_id": self.context.recording_id,
            "source_category": self.context.source.value,
            "coordinate_unit": self.context.coordinate_frame.unit,
            "total_samples": self.total_samples,
            "accepted_positions": self.accepted_positions,
            "accepted_fraction": self.accepted_fraction,
            "in_arena_positions": self.in_arena_positions,
            "outside_positions": self.outside_positions,
            "outside_fraction": self.outside_fraction,
            "path_length": self.path_length,
            "valid_displacements": len(self.displacements),
            "mean_displacement": fmean(self.displacements) if self.displacements else None,
            "valid_turnings": len(self.turnings_radians),
            "mean_absolute_turning": (
                fmean(abs(value) for value in self.turnings_radians)
                if self.turnings_radians
                else None
            ),
            "boundary_samples": len(self.boundary_distances),
            "mean_boundary_distance": (
                fmean(self.boundary_distances) if self.boundary_distances else None
            ),
            "center_fraction": self.center_fraction,
            "occupancy": list(self.occupancy),
        }


def _median(values: Sequence[float], label: str) -> float:
    if not values:
        raise DataError(f"Cannot derive Roche arena geometry without {label} landmarks.")
    return float(median(values))


def derive_roche_recording_arena(
    frames: Iterable[SpatialFrame], *, arena_id: str | None = None
) -> RectangularArena:
    """Derive one stable image-space rectangle from the four source landmarks."""

    context: SpatialContext | None = None
    left_x: list[float] = []
    right_x: list[float] = []
    top_y: list[float] = []
    bottom_y: list[float] = []

    for frame in frames:
        if context is None:
            context = frame.context
        elif frame.context != context:
            raise DataError("Arena derivation cannot mix spatial contexts.")
        if frame.landmarks is None:
            continue
        for name in ("tl", "bl"):
            point = frame.landmarks.landmark(name).point
            if point is not None:
                left_x.append(point.x)
        for name in ("tr", "br"):
            point = frame.landmarks.landmark(name).point
            if point is not None:
                right_x.append(point.x)
        for name in ("tl", "tr"):
            point = frame.landmarks.landmark(name).point
            if point is not None:
                top_y.append(point.y)
        for name in ("bl", "br"):
            point = frame.landmarks.landmark(name).point
            if point is not None:
                bottom_y.append(point.y)

    if context is None:
        raise DataError("Arena derivation requires at least one spatial frame.")

    return RectangularArena(
        arena_id=arena_id or f"{context.recording_id}:arena",
        coordinate_frame=context.coordinate_frame,
        minimum=Point2D(_median(left_x, "left"), _median(top_y, "top")),
        maximum=Point2D(_median(right_x, "right"), _median(bottom_y, "bottom")),
    )


def _grid_index(point: Point2D, arena: RectangularArena, grid_size: int) -> int:
    x_fraction = (point.x - arena.minimum.x) / arena.width
    y_fraction = (point.y - arena.minimum.y) / arena.height
    x_index = min(grid_size - 1, max(0, int(x_fraction * grid_size)))
    y_index = min(grid_size - 1, max(0, int(y_fraction * grid_size)))
    return y_index * grid_size + x_index


def measure_spatial_trajectory(
    frames: Iterable[SpatialFrame],
    *,
    arena: RectangularArena,
    trajectory_protocol: SpatialTrajectoryProtocol,
    evaluation_protocol: SpatialEvaluationProtocol = ROCHE_SPATIAL_EVALUATION_PROTOCOL,
) -> SpatialMeasurements:
    """Measure one trajectory without interpolating gaps or fabricating time units."""

    steps = tuple(trajectory_protocol.trajectory_steps(frames))
    if not steps:
        raise DataError("Spatial evaluation requires at least one trajectory sample.")
    context = steps[0].context
    if arena.coordinate_frame != context.coordinate_frame:
        raise DataError(
            "Spatial evaluation arena and trajectory must share a coordinate frame."
        )

    center = arena.center_region(
        margin_fraction=evaluation_protocol.center_margin_fraction
    )
    occupancy_counts = [0] * (evaluation_protocol.occupancy_grid_size**2)
    accepted = in_arena = outside = center_count = 0
    boundary_distances: list[float] = []
    displacements: list[float] = []
    turnings: list[float] = []
    path_length = 0.0

    for step in steps:
        if step.context != context:
            raise DataError("Spatial evaluation cannot mix trajectory contexts.")
        if step.displacement is not None:
            displacements.append(step.displacement)
            path_length += step.displacement
        if step.turning_radians is not None:
            turnings.append(step.turning_radians)
        if step.point is None:
            continue
        accepted += 1
        if not arena.contains(step.point):
            outside += 1
            continue
        in_arena += 1
        boundary_distances.append(arena.distance_to_boundary(step.point))
        if center.contains(step.point):
            center_count += 1
        occupancy_counts[
            _grid_index(
                step.point, arena, evaluation_protocol.occupancy_grid_size
            )
        ] += 1

    occupancy = (
        tuple(count / in_arena for count in occupancy_counts)
        if in_arena
        else tuple(0.0 for _ in occupancy_counts)
    )
    return SpatialMeasurements(
        context=context,
        total_samples=len(steps),
        accepted_positions=accepted,
        in_arena_positions=in_arena,
        outside_positions=outside,
        path_length=path_length,
        displacements=tuple(displacements),
        turnings_radians=tuple(turnings),
        boundary_distances=tuple(boundary_distances),
        center_fraction=(center_count / in_arena if in_arena else None),
        occupancy=occupancy,
    )


def empirical_wasserstein(
    left: Sequence[float], right: Sequence[float]
) -> float | None:
    """Return exact empirical 1-Wasserstein distance in one dimension."""

    if not left or not right:
        return None
    a = sorted(left)
    b = sorted(right)
    points = sorted(set(a) | set(b))
    ai = bi = 0
    distance = 0.0
    for index, point in enumerate(points[:-1]):
        while ai < len(a) and a[ai] <= point:
            ai += 1
        while bi < len(b) and b[bi] <= point:
            bi += 1
        distance += (
            abs(ai / len(a) - bi / len(b)) * (points[index + 1] - point)
        )
    return distance


def occupancy_total_variation(
    left: Sequence[float], right: Sequence[float]
) -> float:
    if len(left) != len(right):
        raise DataError("Occupancy comparisons require equal grid shapes.")
    return 0.5 * sum(
        abs(a - b) for a, b in zip(left, right, strict=True)
    )


def compare_spatial_measurements(
    recorded: SpatialMeasurements, synthetic: SpatialMeasurements
) -> dict[str, float | None]:
    """Compare separately measured recorded and synthetic trajectories descriptively."""

    if recorded.context.coordinate_frame != synthetic.context.coordinate_frame:
        raise DataError(
            "Recorded and synthetic spatial comparisons require one coordinate frame."
        )
    relative_path = (
        abs(synthetic.path_length - recorded.path_length) / recorded.path_length
        if recorded.path_length > 0
        else None
    )
    center_difference = (
        abs(synthetic.center_fraction - recorded.center_fraction)
        if recorded.center_fraction is not None
        and synthetic.center_fraction is not None
        else None
    )
    return {
        "path_length_absolute_difference": abs(
            synthetic.path_length - recorded.path_length
        ),
        "path_length_relative_difference": relative_path,
        "displacement_wasserstein": empirical_wasserstein(
            recorded.displacements, synthetic.displacements
        ),
        "turning_wasserstein": empirical_wasserstein(
            recorded.turnings_radians, synthetic.turnings_radians
        ),
        "boundary_distance_wasserstein": empirical_wasserstein(
            recorded.boundary_distances, synthetic.boundary_distances
        ),
        "center_fraction_absolute_difference": center_difference,
        "occupancy_total_variation": occupancy_total_variation(
            recorded.occupancy, synthetic.occupancy
        ),
        "outside_fraction_absolute_difference": abs(
            synthetic.outside_fraction - recorded.outside_fraction
        ),
    }
