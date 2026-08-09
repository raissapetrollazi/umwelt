"""Explicit two-dimensional spatial observations for Umwelt v0.2."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from umwelt.errors import DataError
from umwelt.observations import ObservationSource


@dataclass(frozen=True, slots=True)
class CoordinateFrame:
    """Named coordinate frame with an explicit unit and no implied calibration."""

    frame_id: str
    unit: str

    def __post_init__(self) -> None:
        if not self.frame_id.strip():
            raise DataError("A coordinate frame must have a non-empty identifier.")
        if not self.unit.strip():
            raise DataError("A coordinate frame must declare its unit.")


@dataclass(frozen=True, slots=True)
class Point2D:
    """One finite point in an explicitly declared coordinate frame."""

    x: float
    y: float

    def __post_init__(self) -> None:
        if not isfinite(self.x) or not isfinite(self.y):
            raise DataError("Spatial coordinates must be finite.")

    def squared_distance_to(self, other: Point2D) -> float:
        """Return squared Euclidean distance without changing coordinate units."""

        dx = other.x - self.x
        dy = other.y - self.y
        return dx * dx + dy * dy


@dataclass(frozen=True, slots=True)
class Keypoint2D:
    """One named pose-estimation keypoint and its optional source confidence."""

    name: str
    point: Point2D | None
    confidence: float | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise DataError("A spatial keypoint must have a non-empty name.")
        if self.confidence is not None and (
            not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0
        ):
            raise DataError("Spatial keypoint confidence must be between 0 and 1.")


@dataclass(frozen=True, slots=True)
class Pose2D:
    """A named set of two-dimensional keypoints for one sampled frame."""

    keypoints: tuple[Keypoint2D, ...]

    def __post_init__(self) -> None:
        names = [keypoint.name for keypoint in self.keypoints]
        if len(names) != len(set(names)):
            raise DataError("Pose keypoint names must be unique within one frame.")

    def keypoint(self, name: str) -> Keypoint2D:
        """Return one keypoint by name."""

        for keypoint in self.keypoints:
            if keypoint.name == name:
                return keypoint
        raise DataError(f"Unknown pose keypoint: {name}")

    @property
    def observed_keypoint_count(self) -> int:
        """Return the number of keypoints with accepted coordinates."""

        return sum(keypoint.point is not None for keypoint in self.keypoints)


@dataclass(frozen=True, slots=True)
class LandmarkSet2D:
    """A named set of environmental landmarks kept separate from body pose."""

    landmarks: tuple[Keypoint2D, ...]

    def __post_init__(self) -> None:
        names = [landmark.name for landmark in self.landmarks]
        if len(names) != len(set(names)):
            raise DataError("Environmental landmark names must be unique within one frame.")

    def landmark(self, name: str) -> Keypoint2D:
        """Return one environmental landmark by name."""

        for landmark in self.landmarks:
            if landmark.name == name:
                return landmark
        raise DataError(f"Unknown environmental landmark: {name}")

    @property
    def observed_landmark_count(self) -> int:
        """Return the number of landmarks with accepted coordinates."""

        return sum(landmark.point is not None for landmark in self.landmarks)


@dataclass(frozen=True, slots=True)
class SpatialFrame:
    """One frame-indexed pose sample with optional environmental landmarks."""

    frame_index: int
    pose: Pose2D | None
    landmarks: LandmarkSet2D | None = None

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise DataError("Spatial frame indices must be non-negative.")


@dataclass(frozen=True, slots=True)
class SpatialSeries:
    """One spatial recording whose missing frames and source remain explicit."""

    subject_id: str
    recording_id: str
    source: ObservationSource
    coordinate_frame: CoordinateFrame
    frames: tuple[SpatialFrame, ...]
    sampling_rate_hz: float | None = None

    def __post_init__(self) -> None:
        if not self.subject_id.strip():
            raise DataError("A spatial series must have a subject identifier.")
        if not self.recording_id.strip():
            raise DataError("A spatial series must have a recording identifier.")
        if self.sampling_rate_hz is not None and (
            not isfinite(self.sampling_rate_hz) or self.sampling_rate_hz <= 0
        ):
            raise DataError("Spatial sampling rate must be finite and positive.")

        previous_index: int | None = None
        for frame in self.frames:
            if previous_index is not None and frame.frame_index <= previous_index:
                raise DataError("Spatial frame indices must be strictly increasing.")
            previous_index = frame.frame_index

    @property
    def valid_pose_count(self) -> int:
        """Return the number of frames that contain a pose object."""

        return sum(frame.pose is not None for frame in self.frames)

    @property
    def missing_pose_count(self) -> int:
        """Return the number of explicit whole-pose gaps."""

        return len(self.frames) - self.valid_pose_count

    def elapsed_seconds(self, frame_index: int) -> float:
        """Convert a frame index to elapsed seconds when calibration is known."""

        if self.sampling_rate_hz is None:
            raise DataError("Spatial series has no validated sampling rate.")
        if frame_index < 0:
            raise DataError("Spatial frame indices must be non-negative.")
        return frame_index / self.sampling_rate_hz

    def keypoint_names(self) -> tuple[str, ...]:
        """Return sorted keypoint names observed in any non-missing pose."""

        names: set[str] = set()
        for frame in self.frames:
            if frame.pose is not None:
                names.update(keypoint.name for keypoint in frame.pose.keypoints)
        return tuple(sorted(names))


@dataclass(frozen=True, slots=True)
class SpatialDataset:
    """A source-consistent collection of spatial recordings with provenance."""

    dataset_id: str
    series: tuple[SpatialSeries, ...]
    provenance: dict[str, object]

    def __post_init__(self) -> None:
        if not self.dataset_id.strip():
            raise DataError("A spatial dataset must have an identifier.")

        recording_ids = [item.recording_id for item in self.series]
        if len(recording_ids) != len(set(recording_ids)):
            raise DataError("Spatial dataset recording identifiers must be unique.")

        sources = {item.source for item in self.series}
        if len(sources) > 1:
            raise DataError("A spatial dataset cannot silently mix observation sources.")

    def recording(self, recording_id: str) -> SpatialSeries:
        """Return one spatial series by recording identifier."""

        for item in self.series:
            if item.recording_id == recording_id:
                return item
        raise DataError(f"Unknown spatial recording identifier: {recording_id}")

    def subject_series(self, subject_id: str) -> tuple[SpatialSeries, ...]:
        """Return every recording associated with one subject identifier."""

        matches = tuple(item for item in self.series if item.subject_id == subject_id)
        if not matches:
            raise DataError(f"Unknown spatial subject identifier: {subject_id}")
        return matches
