"""Optional, streaming adapter for CalMS21 Task 1 pose JSON."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import cast
from zipfile import ZipFile

from umwelt.dyad import DyadFrame
from umwelt.errors import DataError
from umwelt.observations import ObservationSource
from umwelt.spatial import (
    AxisOrientation,
    CoordinateFrame,
    Keypoint2D,
    Point2D,
    Pose2D,
    SpatialContext,
    SpatialFrame,
)


CALMS21_KEYPOINT_NAMES = (
    "nose",
    "left_ear",
    "right_ear",
    "neck",
    "left_hip",
    "right_hip",
    "tail_base",
)
CALMS21_COORDINATE_FRAME = CoordinateFrame(
    "calms21-image-1024x570", "px", AxisOrientation.X_RIGHT_Y_DOWN
)
CALMS21_ARCHIVE_MD5 = "8a02654fddae28614ee24a6a082261b8"


def _array(value: object, length: int, label: str) -> Sequence[object]:
    if not isinstance(value, list) or len(value) != length:
        raise DataError(f"CalMS21 {label} must have length {length}.")
    return cast(Sequence[object], value)


def _number(value: object, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataError(f"CalMS21 {label} must be numeric or null.")
    result = float(value)
    return result if isfinite(result) else None


@dataclass(frozen=True, slots=True)
class Calms21Task1Sequence:
    """One source recording, with animal identities local to that recording."""

    sequence_id: str
    keypoints: Sequence[object]
    scores: Sequence[object]

    def __post_init__(self) -> None:
        if not self.sequence_id.startswith("task1/"):
            raise DataError("CalMS21 Task 1 sequence identifier is invalid.")
        if (
            not isinstance(self.keypoints, list)
            or not isinstance(self.scores, list)
            or not self.keypoints
            or len(self.keypoints) != len(self.scores)
        ):
            raise DataError("CalMS21 keypoints and scores need equal nonzero frames.")

    @property
    def frame_count(self) -> int:
        return len(self.keypoints)

    def frames(self) -> Iterator[DyadFrame]:
        """Preserve all seven confidence values per mouse and explicit gaps."""

        contexts = (
            SpatialContext(
                f"{self.sequence_id}:resident",
                self.sequence_id,
                ObservationSource.RECORDED,
                CALMS21_COORDINATE_FRAME,
            ),
            SpatialContext(
                f"{self.sequence_id}:intruder",
                self.sequence_id,
                ObservationSource.RECORDED,
                CALMS21_COORDINATE_FRAME,
            ),
        )
        for index, (positions_raw, scores_raw) in enumerate(
            zip(self.keypoints, self.scores, strict=True)
        ):
            positions = _array(positions_raw, 2, "mouse positions")
            confidences = _array(scores_raw, 2, "mouse scores")
            members: list[SpatialFrame] = []
            for mouse_index in (0, 1):
                xy = _array(positions[mouse_index], 2, "coordinate axes")
                x_coordinates = _array(xy[0], 7, "x keypoints")
                y_coordinates = _array(xy[1], 7, "y keypoints")
                scores = _array(confidences[mouse_index], 7, "keypoint scores")
                keypoints = []
                for part_index, part_name in enumerate(CALMS21_KEYPOINT_NAMES):
                    x = _number(x_coordinates[part_index], "x coordinate")
                    y = _number(y_coordinates[part_index], "y coordinate")
                    score = _number(scores[part_index], "confidence")
                    point = Point2D(x, y) if x is not None and y is not None else None
                    keypoints.append(Keypoint2D(part_name, point, score))
                members.append(
                    SpatialFrame(contexts[mouse_index], index, Pose2D(tuple(keypoints)))
                )
            yield DyadFrame(self.sequence_id, members[0], members[1])


def iter_calms21_task1_sequences(
    archive_path: str | Path, *, split: str = "train"
) -> Iterator[Calms21Task1Sequence]:
    """Read one named split, materializing at most one sequence at a time."""

    if split not in ("train", "test"):
        raise DataError("CalMS21 Task 1 split must be train or test.")
    try:
        import ijson  # type: ignore[import-not-found]
    except ImportError as error:
        raise DataError(
            "Install Umwelt with the calms21 extra to read poses."
        ) from error

    expected_name = f"calms21_task1_{split}.json"
    with ZipFile(archive_path) as archive:
        matches = [
            name for name in archive.namelist() if Path(name).name == expected_name
        ]
        if len(matches) != 1:
            raise DataError(f"Expected one {expected_name} in the CalMS21 archive.")
        sequence_count = 0
        with archive.open(matches[0]) as source:
            for sequence_id, payload in ijson.kvitems(
                source, "annotator-id_0", use_float=True
            ):
                if not isinstance(payload, Mapping):
                    raise DataError("CalMS21 sequence payload must be an object.")
                if not str(sequence_id).startswith(f"task1/{split}/"):
                    raise DataError("CalMS21 sequence does not match selected split.")
                sequence = Calms21Task1Sequence(
                    str(sequence_id),
                    cast(Sequence[object], payload.get("keypoints")),
                    cast(Sequence[object], payload.get("scores")),
                )
                sequence_count += 1
                yield sequence
        if not sequence_count:
            raise DataError("No CalMS21 Task 1 sequences found in selected split.")
