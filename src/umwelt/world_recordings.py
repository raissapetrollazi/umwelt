"""Compact, validated Roche inputs for the first behavioral-world experiment."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from statistics import median

from umwelt.arena import RectangularArena
from umwelt.datasets.roche_open_field import RocheRecording
from umwelt.errors import DataError
from umwelt.spatial import Pose2D, SpatialFrame
from umwelt.spatial_evaluation import derive_roche_recording_arena, distribution_summary
from umwelt.spatial_protocol import ROCHE_RECORDED_TRAJECTORY_PROTOCOL
from umwelt.trajectory import PositionStatus, TrajectoryStep
from umwelt.world import (
    V03_ARENA_SCALE_CONDITIONS,
    V03_WORLD_ID,
    OpenFieldWorld,
)


@dataclass(frozen=True, slots=True)
class RecordedWorldSeries:
    """One source animal, with full pose rows discarded after trajectory extraction."""

    subject_id: str
    recording_id: str
    arena: RectangularArena
    steps: tuple[TrajectoryStep, ...]
    landmark_quality: dict[str, object]

    @property
    def frame_indices(self) -> tuple[int, ...]:
        return tuple(step.frame_index for step in self.steps)

    @property
    def availability_statuses(self) -> tuple[PositionStatus, ...]:
        return tuple(step.status for step in self.steps)

    def world(self, condition_id: str) -> OpenFieldWorld:
        for condition in V03_ARENA_SCALE_CONDITIONS:
            if condition.condition_id == condition_id:
                return OpenFieldWorld(V03_WORLD_ID, self.arena, condition)
        raise DataError(f"Unknown v0.3 world condition: {condition_id}")


def load_recorded_world_series(recording: RocheRecording) -> RecordedWorldSeries:
    """Read source rows once, retain bodycentre only, and validate arena corners."""

    if recording.metadata.group != "Control" or recording.metadata.dosage != "0":
        raise DataError("The v0.3 world experiment requires Control dose 0 data.")
    body_frames: list[SpatialFrame] = []
    corners: dict[str, dict[str, list[float]]] = {
        name: {"x": [], "y": []} for name in ("tl", "tr", "bl", "br")
    }

    def project() -> Iterator[SpatialFrame]:
        for frame in recording.frames():
            pose = (
                Pose2D((frame.pose.keypoint("bodycentre"),))
                if frame.pose is not None
                else None
            )
            body_frames.append(SpatialFrame(frame.context, frame.frame_index, pose))
            if frame.landmarks is not None:
                for name in corners:
                    point = frame.landmarks.landmark(name).point
                    if point is not None:
                        corners[name]["x"].append(point.x)
                        corners[name]["y"].append(point.y)
            yield frame

    arena = derive_roche_recording_arena(project())
    if any(not coordinates["x"] for coordinates in corners.values()):
        raise DataError("World arena requires every source corner landmark.")
    positions = {
        name: (median(coordinates["x"]), median(coordinates["y"]))
        for name, coordinates in corners.items()
    }
    if not (
        positions["tl"][0] < positions["tr"][0]
        and positions["bl"][0] < positions["br"][0]
        and positions["tl"][1] < positions["bl"][1]
        and positions["tr"][1] < positions["br"][1]
    ):
        raise DataError("Source corner medians describe inconsistent arena bounds.")

    steps = tuple(ROCHE_RECORDED_TRAJECTORY_PROTOCOL.trajectory_steps(body_frames))
    if not steps or steps[0].context.subject_id != recording.subject_id:
        raise DataError("Recorded world trajectory identity is inconsistent.")
    return RecordedWorldSeries(
        subject_id=recording.subject_id,
        recording_id=recording.recording_id,
        arena=arena,
        steps=steps,
        landmark_quality={
            name: {
                axis: distribution_summary(values)
                for axis, values in coordinates.items()
            }
            for name, coordinates in corners.items()
        },
    )
