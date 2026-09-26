"""Two-animal observation and gap-aware geometry for Umwelt v0.4."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import hypot, isfinite

from umwelt.errors import DataError
from umwelt.observations import ObservationSource
from umwelt.spatial import CoordinateFrame, Point2D, SpatialFrame
from umwelt.spatial_evaluation import distribution_summary


@dataclass(frozen=True, slots=True)
class DyadFrame:
    """Two separately identified poses observed at the same frame index."""

    pair_id: str
    resident: SpatialFrame
    intruder: SpatialFrame

    def __post_init__(self) -> None:
        if not self.pair_id.strip():
            raise DataError("A dyad frame requires a pair identifier.")
        resident = self.resident.context
        intruder = self.intruder.context
        if resident.subject_id == intruder.subject_id:
            raise DataError("Dyad members must have distinct subject identifiers.")
        if self.resident.frame_index != self.intruder.frame_index:
            raise DataError("Dyad poses must share one frame index.")
        if resident.recording_id != intruder.recording_id:
            raise DataError("Dyad poses must share one recording identifier.")
        if resident.source != intruder.source:
            raise DataError("Dyad poses must share one observation source.")
        if resident.coordinate_frame != intruder.coordinate_frame:
            raise DataError("Dyad poses must share one coordinate frame.")

    @property
    def frame_index(self) -> int:
        return self.resident.frame_index

    @property
    def source(self) -> ObservationSource:
        return self.resident.context.source

    @property
    def coordinate_frame(self) -> CoordinateFrame:
        return self.resident.context.coordinate_frame

    def neck_positions(self) -> tuple[Point2D | None, Point2D | None]:
        """Select observed neck coordinates without inventing a body centre."""

        positions: list[Point2D | None] = []
        for member in (self.resident, self.intruder):
            if member.pose is None:
                positions.append(None)
            else:
                positions.append(member.pose.keypoint("neck").point)
        return positions[0], positions[1]


@dataclass(frozen=True, slots=True)
class DyadMeasurements:
    """Pair-distance samples with frame and transition exposure."""

    pair_id: str
    recording_id: str
    resident_id: str
    intruder_id: str
    source: ObservationSource
    coordinate_frame: CoordinateFrame
    frame_count: int
    resident_valid_position_count: int
    intruder_valid_position_count: int
    valid_pair_position_count: int
    adjacent_frame_opportunity_count: int
    valid_pair_transition_count: int
    pair_distances_px: tuple[float, ...]
    signed_distance_changes_px: tuple[float, ...]

    def __post_init__(self) -> None:
        if not all(
            value.strip()
            for value in (
                self.pair_id,
                self.recording_id,
                self.resident_id,
                self.intruder_id,
            )
        ):
            raise DataError("Dyad measurements require non-empty identifiers.")
        if self.resident_id == self.intruder_id:
            raise DataError("Dyad members must have distinct subject identifiers.")
        if self.coordinate_frame.unit != "px":
            raise DataError("First dyad measurements require image-pixel coordinates.")
        if (
            self.frame_count < 1
            or not all(
                0 <= count <= self.frame_count
                for count in (
                    self.resident_valid_position_count,
                    self.intruder_valid_position_count,
                )
            )
            or not (
                0
                <= self.valid_pair_position_count
                <= min(
                    self.resident_valid_position_count,
                    self.intruder_valid_position_count,
                )
            )
        ):
            raise DataError("Dyad position exposure counts are inconsistent.")
        if not (
            0
            <= self.valid_pair_transition_count
            <= self.adjacent_frame_opportunity_count
            <= max(self.frame_count - 1, 0)
        ):
            raise DataError("Dyad transition exposure counts are inconsistent.")
        if len(self.pair_distances_px) != self.valid_pair_position_count:
            raise DataError("Pair distances must match valid pair positions.")
        if len(self.signed_distance_changes_px) != self.valid_pair_transition_count:
            raise DataError("Distance changes must match valid pair transitions.")
        if any(not isfinite(value) or value < 0 for value in self.pair_distances_px):
            raise DataError("Pair distances must be finite and non-negative.")
        if any(not isfinite(value) for value in self.signed_distance_changes_px):
            raise DataError("Pair distance changes must be finite.")

    def compact_dict(self) -> dict[str, object]:
        """Summarize pixel geometry without retaining full trajectories."""

        return {
            "pair_id": self.pair_id,
            "recording_id": self.recording_id,
            "resident_id": self.resident_id,
            "intruder_id": self.intruder_id,
            "source_category": self.source.value,
            "coordinate_frame": {
                "frame_id": self.coordinate_frame.frame_id,
                "unit": self.coordinate_frame.unit,
                "axis_orientation": self.coordinate_frame.axis_orientation.value,
            },
            "quality_control": {
                "frame_count": self.frame_count,
                "resident_valid_position_count": self.resident_valid_position_count,
                "intruder_valid_position_count": self.intruder_valid_position_count,
                "valid_pair_position_count": self.valid_pair_position_count,
                "adjacent_frame_opportunity_count": (
                    self.adjacent_frame_opportunity_count
                ),
                "valid_pair_transition_count": self.valid_pair_transition_count,
            },
            "pair_distance_px": distribution_summary(self.pair_distances_px),
            "signed_pair_distance_change_px": distribution_summary(
                self.signed_distance_changes_px
            ),
        }


def measure_dyad_frames(frames: Iterable[DyadFrame]) -> DyadMeasurements:
    """Measure neck-to-neck geometry without bridging gaps or missing poses."""

    first: DyadFrame | None = None
    previous_index: int | None = None
    previous_distance: float | None = None
    frame_count = 0
    resident_count = 0
    intruder_count = 0
    adjacent_opportunities = 0
    distances: list[float] = []
    changes: list[float] = []

    for frame in frames:
        if first is None:
            first = frame
        elif (
            frame.pair_id != first.pair_id
            or frame.resident.context != first.resident.context
            or frame.intruder.context != first.intruder.context
        ):
            raise DataError("Dyad measurement cannot mix pair or source contexts.")
        if previous_index is not None and frame.frame_index <= previous_index:
            raise DataError("Dyad frame indices must be strictly increasing.")

        resident, intruder = frame.neck_positions()
        if resident is not None:
            resident_count += 1
        if intruder is not None:
            intruder_count += 1
        distance = (
            hypot(intruder.x - resident.x, intruder.y - resident.y)
            if resident is not None and intruder is not None
            else None
        )
        if distance is not None:
            distances.append(distance)
        if previous_index is not None and frame.frame_index == previous_index + 1:
            adjacent_opportunities += 1
            if previous_distance is not None and distance is not None:
                changes.append(distance - previous_distance)
        previous_index = frame.frame_index
        previous_distance = distance
        frame_count += 1

    if first is None:
        raise DataError("Dyad measurement requires at least one frame.")
    return DyadMeasurements(
        pair_id=first.pair_id,
        recording_id=first.resident.context.recording_id,
        resident_id=first.resident.context.subject_id,
        intruder_id=first.intruder.context.subject_id,
        source=first.source,
        coordinate_frame=first.coordinate_frame,
        frame_count=frame_count,
        resident_valid_position_count=resident_count,
        intruder_valid_position_count=intruder_count,
        valid_pair_position_count=len(distances),
        adjacent_frame_opportunity_count=adjacent_opportunities,
        valid_pair_transition_count=len(changes),
        pair_distances_px=tuple(distances),
        signed_distance_changes_px=tuple(changes),
    )
