"""Tests for explicit v0.2 spatial observations and arena geometry."""

from __future__ import annotations

import unittest

from umwelt.arena import RectangularArena
from umwelt.errors import DataError
from umwelt.observations import ObservationSource
from umwelt.spatial import (
    CoordinateFrame,
    Keypoint2D,
    Point2D,
    Pose2D,
    SpatialDataset,
    SpatialFrame,
    SpatialSeries,
)


class SpatialObservationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = CoordinateFrame("camera-pixels", "px")
        self.pose = Pose2D(
            (
                Keypoint2D("nose", Point2D(10.0, 20.0), 0.99),
                Keypoint2D("center", Point2D(12.0, 22.0), 0.95),
                Keypoint2D("tail_base", None, 0.20),
            )
        )

    def _series(
        self,
        recording_id: str = "recording-1",
        source: ObservationSource = ObservationSource.RECORDED,
    ) -> SpatialSeries:
        return SpatialSeries(
            subject_id="mouse-1",
            recording_id=recording_id,
            source=source,
            coordinate_frame=self.frame,
            frames=(
                SpatialFrame(0, self.pose),
                SpatialFrame(1, None),
                SpatialFrame(2, self.pose),
            ),
            sampling_rate_hz=30.0,
        )

    def test_pose_preserves_keypoint_confidence_and_missingness(self) -> None:
        self.assertEqual(self.pose.observed_keypoint_count, 2)
        self.assertEqual(self.pose.keypoint("nose").confidence, 0.99)
        self.assertIsNone(self.pose.keypoint("tail_base").point)

    def test_series_preserves_source_gaps_and_sampling_calibration(self) -> None:
        series = self._series()

        self.assertEqual(series.source, ObservationSource.RECORDED)
        self.assertEqual(series.valid_pose_count, 2)
        self.assertEqual(series.missing_pose_count, 1)
        self.assertAlmostEqual(series.elapsed_seconds(15), 0.5)
        self.assertEqual(series.keypoint_names(), ("center", "nose", "tail_base"))

    def test_series_without_sampling_rate_rejects_time_conversion(self) -> None:
        series = SpatialSeries(
            subject_id="mouse-1",
            recording_id="recording-1",
            source=ObservationSource.RECORDED,
            coordinate_frame=self.frame,
            frames=(SpatialFrame(0, self.pose),),
        )
        with self.assertRaisesRegex(DataError, "no validated sampling rate"):
            series.elapsed_seconds(1)

    def test_spatial_validation_rejects_invalid_values(self) -> None:
        with self.assertRaisesRegex(DataError, "coordinates must be finite"):
            Point2D(float("nan"), 1.0)
        with self.assertRaisesRegex(DataError, "confidence must be between"):
            Keypoint2D("nose", Point2D(0.0, 0.0), 1.1)
        with self.assertRaisesRegex(DataError, "must be unique"):
            Pose2D(
                (
                    Keypoint2D("nose", Point2D(0.0, 0.0)),
                    Keypoint2D("nose", Point2D(1.0, 1.0)),
                )
            )
        with self.assertRaisesRegex(DataError, "strictly increasing"):
            SpatialSeries(
                subject_id="mouse-1",
                recording_id="recording-1",
                source=ObservationSource.RECORDED,
                coordinate_frame=self.frame,
                frames=(SpatialFrame(2, self.pose), SpatialFrame(1, self.pose)),
            )

    def test_dataset_rejects_source_mixing_and_duplicate_recordings(self) -> None:
        recorded = self._series("recording-1")
        duplicate = self._series("recording-1")
        synthetic = self._series("recording-2", ObservationSource.SYNTHETIC)

        with self.assertRaisesRegex(DataError, "recording identifiers must be unique"):
            SpatialDataset("fixture", (recorded, duplicate), {})
        with self.assertRaisesRegex(DataError, "cannot silently mix"):
            SpatialDataset("fixture", (recorded, synthetic), {})


class ArenaTests(unittest.TestCase):
    def setUp(self) -> None:
        frame = CoordinateFrame("camera-pixels", "px")
        self.arena = RectangularArena(
            arena_id="open-field",
            coordinate_frame=frame,
            minimum=Point2D(10.0, 20.0),
            maximum=Point2D(110.0, 220.0),
        )

    def test_rectangle_exposes_geometry_without_implied_physical_units(self) -> None:
        self.assertEqual(self.arena.coordinate_frame.unit, "px")
        self.assertEqual(self.arena.width, 100.0)
        self.assertEqual(self.arena.height, 200.0)
        self.assertTrue(self.arena.contains(Point2D(10.0, 20.0)))
        self.assertFalse(
            self.arena.contains(Point2D(10.0, 20.0), include_boundary=False)
        )
        self.assertFalse(self.arena.contains(Point2D(9.0, 20.0)))

    def test_boundary_distance_requires_in_arena_point(self) -> None:
        self.assertEqual(self.arena.distance_to_boundary(Point2D(25.0, 60.0)), 15.0)
        with self.assertRaisesRegex(DataError, "inside the arena"):
            self.arena.distance_to_boundary(Point2D(0.0, 0.0))

    def test_center_region_is_explicit_geometric_partition(self) -> None:
        center = self.arena.center_region(margin_fraction=0.25)

        self.assertEqual(center.minimum, Point2D(35.0, 70.0))
        self.assertEqual(center.maximum, Point2D(85.0, 170.0))
        self.assertEqual(center.coordinate_frame, self.arena.coordinate_frame)
        with self.assertRaisesRegex(DataError, "margin fraction"):
            self.arena.center_region(margin_fraction=0.5)

    def test_arena_rejects_degenerate_bounds(self) -> None:
        with self.assertRaisesRegex(DataError, "maximum bounds must exceed"):
            RectangularArena(
                arena_id="bad",
                coordinate_frame=self.arena.coordinate_frame,
                minimum=Point2D(1.0, 1.0),
                maximum=Point2D(1.0, 2.0),
            )


if __name__ == "__main__":
    unittest.main()
