"""Environment-linked measurements for the Umwelt v0.3 open-field world."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite

from umwelt.errors import DataError
from umwelt.observations import ObservationSource
from umwelt.spatial import SpatialContext, SpatialFrame
from umwelt.spatial_evaluation import (
    SpatialComparison,
    SpatialMeasurements,
    compare_spatial_measurements,
    distribution_summary,
    empirical_wasserstein,
    measure_trajectory_steps,
)
from umwelt.spatial_protocol import SpatialTrajectoryProtocol
from umwelt.trajectory import TrajectoryStep
from umwelt.world import OpenFieldWorld, WorldZone


WORLD_REGISTERED_METRIC_IDS = (
    "boundary-band-occupancy-fraction",
    "zone-conditioned-positive-displacement-px-distribution",
)
WORLD_QUALITY_CONTROL_IDS = (
    "in-arena-position-count",
    "outside-arena-position-count",
    "zone-position-counts",
    "zone-valid-transition-counts",
    "unclassified-transition-start-count",
    "unclassified-positive-transition-start-count",
)


@dataclass(frozen=True, slots=True)
class WorldMeasurements:
    """Spatial measurements augmented by explicit world-zone derivations."""

    world: OpenFieldWorld
    spatial: SpatialMeasurements
    zone_position_counts: tuple[tuple[str, int], ...]
    zone_valid_transition_counts: tuple[tuple[str, int], ...]
    outside_arena_position_count: int
    unclassified_transition_start_count: int
    unclassified_positive_transition_start_count: int
    zone_positive_displacements_px: tuple[tuple[str, tuple[float, ...]], ...]

    def __post_init__(self) -> None:
        if self.spatial.context.coordinate_frame != self.world.arena.coordinate_frame:
            raise DataError("World measurements require one coordinate frame.")
        expected_zones = tuple(zone.value for zone in WorldZone)
        if tuple(name for name, _ in self.zone_position_counts) != expected_zones:
            raise DataError("World position counts must include every declared zone.")
        if (
            tuple(name for name, _ in self.zone_valid_transition_counts)
            != expected_zones
        ):
            raise DataError("World transition counts must include every declared zone.")
        if (
            tuple(name for name, _ in self.zone_positive_displacements_px)
            != expected_zones
        ):
            raise DataError(
                "World displacement samples must include every declared zone."
            )
        counts = (
            tuple(count for _, count in self.zone_position_counts)
            + tuple(count for _, count in self.zone_valid_transition_counts)
            + (
                self.outside_arena_position_count,
                self.unclassified_transition_start_count,
                self.unclassified_positive_transition_start_count,
            )
        )
        if any(count < 0 for count in counts):
            raise DataError("World measurement counts must be non-negative.")
        if (
            self.in_arena_position_count + self.outside_arena_position_count
            != self.spatial.quality_control.accepted_positions
        ):
            raise DataError(
                "World position counts must cover every accepted spatial position."
            )
        if (
            sum(count for _, count in self.zone_valid_transition_counts)
            + self.unclassified_transition_start_count
            != self.spatial.quality_control.valid_transition_count
        ):
            raise DataError(
                "World transition counts must cover every valid spatial transition."
            )
        if (
            self.unclassified_positive_transition_start_count
            > self.unclassified_transition_start_count
        ):
            raise DataError("Unclassified positive transitions exceed all transitions.")
        classified_displacements = tuple(
            value
            for _, values in self.zone_positive_displacements_px
            for value in values
        )
        if any(not isfinite(value) or value <= 0 for value in classified_displacements):
            raise DataError(
                "Zone-conditioned displacement samples must be finite and positive."
            )
        positive_displacement_count = sum(
            value > 0 for value in self.spatial.displacements_px
        )
        if (
            len(classified_displacements)
            + self.unclassified_positive_transition_start_count
            != positive_displacement_count
        ):
            raise DataError(
                "World displacement counts must cover every positive spatial movement."
            )

    @property
    def context(self) -> SpatialContext:
        return self.spatial.context

    @property
    def in_arena_position_count(self) -> int:
        return sum(count for _, count in self.zone_position_counts)

    @property
    def boundary_band_occupancy_fraction(self) -> float | None:
        if not self.in_arena_position_count:
            return None
        counts = dict(self.zone_position_counts)
        return counts[WorldZone.BOUNDARY_BAND.value] / self.in_arena_position_count

    def positive_displacements(self, zone: WorldZone) -> tuple[float, ...]:
        """Return positive displacements classified by transition-start zone."""

        if not isinstance(zone, WorldZone):
            raise DataError("World displacement lookup requires a declared zone.")
        return dict(self.zone_positive_displacements_px)[zone.value]

    def compact_dict(self) -> dict[str, object]:
        """Return environment-linked metrics with world and source provenance."""

        displacements = dict(self.zone_positive_displacements_px)
        return {
            "subject_id": self.context.subject_id,
            "recording_id": self.context.recording_id,
            "source_category": self.context.source.value,
            "world": {
                "world_id": self.world.world_id,
                "world_instance_id": (
                    f"{self.world.world_id}:{self.world.condition.condition_id}"
                ),
                "source_arena_id": self.world.source_arena.arena_id,
                "condition_id": self.world.condition.condition_id,
                "boundary_band_fraction": self.world.boundary_band_fraction,
            },
            "spatial_measurements": self.spatial.compact_dict(),
            "quality_control": {
                "in_arena_position_count": self.in_arena_position_count,
                "outside_arena_position_count": self.outside_arena_position_count,
                "zone_position_counts": dict(self.zone_position_counts),
                "zone_valid_transition_counts": dict(self.zone_valid_transition_counts),
                "unclassified_transition_start_count": (
                    self.unclassified_transition_start_count
                ),
                "unclassified_positive_transition_start_count": (
                    self.unclassified_positive_transition_start_count
                ),
            },
            "registered_metrics": {
                "boundary-band-occupancy-fraction": {
                    "value": self.boundary_band_occupancy_fraction,
                    "in-arena-position-count": self.in_arena_position_count,
                },
                "zone-conditioned-positive-displacement-px-distribution": {
                    zone.value: distribution_summary(displacements[zone.value])
                    for zone in WorldZone
                },
            },
        }


@dataclass(frozen=True, slots=True)
class WorldComparison:
    """Recorded-versus-synthetic differences for one declared world."""

    spatial: SpatialComparison
    boundary_band_occupancy_absolute_difference: float | None
    zone_positive_displacement_wasserstein_px: tuple[tuple[str, float | None], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "spatial": self.spatial.to_dict(),
            "boundary_band_occupancy_absolute_difference": (
                self.boundary_band_occupancy_absolute_difference
            ),
            "zone_positive_displacement_wasserstein_px": dict(
                self.zone_positive_displacement_wasserstein_px
            ),
        }


def measure_world_steps(
    steps: Iterable[TrajectoryStep], *, world: OpenFieldWorld
) -> WorldMeasurements:
    """Measure zones without altering the frozen v0.2 spatial measurements.

    Recorded accepted positions outside the arena remain explicit quality
    control and are excluded from zone occupancy. Synthetic positions outside
    their declared world are invariant violations. Positive movement is assigned
    by its transition-start zone; a recorded transition starting outside the
    arena is retained in the base spatial metrics but unclassified by zone.
    """

    materialized = tuple(steps)
    spatial = measure_trajectory_steps(materialized)
    if spatial.context.coordinate_frame != world.arena.coordinate_frame:
        raise DataError("World and trajectory must share a coordinate frame.")

    zone_counts: Counter[WorldZone] = Counter()
    zone_transitions: Counter[WorldZone] = Counter()
    zone_displacements: dict[WorldZone, list[float]] = {zone: [] for zone in WorldZone}
    outside_positions = 0
    unclassified_transition_starts = 0
    unclassified_positive_transition_starts = 0
    previous_step: TrajectoryStep | None = None
    previous_zone: WorldZone | None = None

    for step in materialized:
        if step.context != spatial.context:
            raise DataError("World evaluation cannot mix trajectory contexts.")
        point = step.point
        position_zone: WorldZone | None = None
        if point is not None:
            if world.arena.contains(point):
                position_zone = world.zone(point)
                zone_counts[position_zone] += 1
            else:
                outside_positions += 1
                if step.context.source is ObservationSource.SYNTHETIC:
                    raise DataError(
                        "Synthetic positions must remain inside their declared world."
                    )

        if step.displacement is not None:
            start = previous_step.point if previous_step is not None else None
            if start is None:
                raise DataError(
                    "A valid displacement must retain its transition-start position."
                )
            if previous_zone is not None:
                zone_transitions[previous_zone] += 1
                if step.displacement > 0:
                    zone_displacements[previous_zone].append(step.displacement)
            else:
                unclassified_transition_starts += 1
                if step.displacement > 0:
                    unclassified_positive_transition_starts += 1
        previous_step = step
        previous_zone = position_zone

    ordered_counts = tuple((zone.value, zone_counts[zone]) for zone in WorldZone)
    ordered_transitions = tuple(
        (zone.value, zone_transitions[zone]) for zone in WorldZone
    )
    ordered_displacements = tuple(
        (zone.value, tuple(zone_displacements[zone])) for zone in WorldZone
    )
    return WorldMeasurements(
        world=world,
        spatial=spatial,
        zone_position_counts=ordered_counts,
        zone_valid_transition_counts=ordered_transitions,
        outside_arena_position_count=outside_positions,
        unclassified_transition_start_count=unclassified_transition_starts,
        unclassified_positive_transition_start_count=(
            unclassified_positive_transition_starts
        ),
        zone_positive_displacements_px=ordered_displacements,
    )


def measure_world_trajectory(
    frames: Iterable[SpatialFrame],
    *,
    trajectory_protocol: SpatialTrajectoryProtocol,
    world: OpenFieldWorld,
) -> WorldMeasurements:
    """Apply recorded trajectory and environmental measurement contracts."""

    return measure_world_steps(
        trajectory_protocol.trajectory_steps(frames),
        world=world,
    )


def compare_world_measurements(
    recorded: WorldMeasurements, synthetic: WorldMeasurements
) -> WorldComparison:
    """Compare one recorded and synthetic trajectory under the same world."""

    if recorded.world != synthetic.world:
        raise DataError("World comparison requires the same declared world.")
    spatial = compare_spatial_measurements(recorded.spatial, synthetic.spatial)
    recorded_occupancy = recorded.boundary_band_occupancy_fraction
    synthetic_occupancy = synthetic.boundary_band_occupancy_fraction
    occupancy_difference = (
        abs(synthetic_occupancy - recorded_occupancy)
        if recorded_occupancy is not None and synthetic_occupancy is not None
        else None
    )
    zone_distances = tuple(
        (
            zone.value,
            empirical_wasserstein(
                recorded.positive_displacements(zone),
                synthetic.positive_displacements(zone),
            ),
        )
        for zone in WorldZone
    )
    return WorldComparison(
        spatial=spatial,
        boundary_band_occupancy_absolute_difference=occupancy_difference,
        zone_positive_displacement_wasserstein_px=zone_distances,
    )
