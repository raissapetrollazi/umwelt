"""Tests for compact Roche world inputs and arena-quality rejection."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from umwelt.datasets.roche_open_field import ROCHE_COORDINATE_FRAME
from umwelt.errors import DataError
from umwelt.observations import ObservationSource
from umwelt.spatial import (
    Keypoint2D,
    LandmarkSet2D,
    Point2D,
    Pose2D,
    SpatialContext,
    SpatialFrame,
)
from umwelt.trajectory import PositionStatus
from umwelt.world_recordings import load_recorded_world_series


def _recording(*, inverted: bool = False, group: str = "Control") -> object:
    context = SpatialContext(
        "16459-67049", "source.csv", ObservationSource.RECORDED, ROCHE_COORDINATE_FRAME
    )
    landmarks = LandmarkSet2D(
        (
            Keypoint2D("tl", Point2D(90.0 if inverted else 0.0, 0.0)),
            Keypoint2D("tr", Point2D(100.0 if inverted else 10.0, 0.0)),
            Keypoint2D("bl", Point2D(90.0 if inverted else 0.0, 10.0)),
            Keypoint2D("br", Point2D(100.0 if inverted else 10.0, 10.0)),
        )
    )
    frames = (
        SpatialFrame(
            context,
            0,
            Pose2D((Keypoint2D("bodycentre", Point2D(1, 1), 0.9),)),
            landmarks,
        ),
        SpatialFrame(
            context,
            1,
            Pose2D((Keypoint2D("bodycentre", Point2D(2, 1), 0.8),)),
            landmarks,
        ),
        SpatialFrame(
            context, 2, Pose2D((Keypoint2D("bodycentre", None, 0.7),)), landmarks
        ),
        SpatialFrame(context, 3, None, landmarks),
    )
    return SimpleNamespace(
        metadata=SimpleNamespace(group=group, dosage="0"),
        subject_id=context.subject_id,
        recording_id=context.recording_id,
        frames=lambda: iter(frames),
    )


class WorldRecordingTests(unittest.TestCase):
    def test_loader_retains_bodycentre_provenance_and_gap_mask(self) -> None:
        series = load_recorded_world_series(_recording())  # type: ignore[arg-type]
        self.assertEqual(series.arena.minimum, Point2D(0.0, 0.0))
        self.assertEqual(series.arena.maximum, Point2D(10.0, 10.0))
        self.assertEqual(series.frame_indices, (0, 1, 2, 3))
        self.assertEqual(
            series.availability_statuses,
            (
                PositionStatus.ACCEPTED,
                PositionStatus.ACCEPTED,
                PositionStatus.SOURCE_COORDINATE_MISSING,
                PositionStatus.SOURCE_POSE_MISSING,
            ),
        )
        self.assertEqual(series.steps[0].confidence, 0.9)
        self.assertEqual(series.landmark_quality["tl"]["x"]["count"], 4)

    def test_loader_rejects_inconsistent_corners_and_treatment(self) -> None:
        with self.assertRaisesRegex(DataError, "Control dose 0"):
            load_recorded_world_series(_recording(group="Yohimbine"))  # type: ignore[arg-type]
        # Each row has finite landmarks, but their corner ordering conflicts
        # with the pooled rectangle used by the scientific protocol.
        bad = _recording(inverted=True)
        frames = tuple(bad.frames())
        swapped = LandmarkSet2D(
            (
                Keypoint2D("tl", Point2D(90.0, 0.0)),
                Keypoint2D("tr", Point2D(10.0, 0.0)),
                Keypoint2D("bl", Point2D(0.0, 10.0)),
                Keypoint2D("br", Point2D(100.0, 10.0)),
            )
        )
        bad.frames = lambda: (
            SpatialFrame(f.context, f.frame_index, f.pose, swapped) for f in frames
        )
        with self.assertRaisesRegex(DataError, "inconsistent arena bounds"):
            load_recorded_world_series(bad)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
