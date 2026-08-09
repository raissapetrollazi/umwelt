"""Tests for gap-aware derived v0.2 trajectory measurements."""

from __future__ import annotations

import math
import unittest

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
from umwelt.trajectory import (
    PositionSample,
    bodycentre_samples,
    iter_trajectory_steps,
    keypoint_position_samples,
    trajectory_path_length,
)


CONTEXT = SpatialContext(
    subject_id="mouse-1",
    recording_id="recording-1",
    source=ObservationSource.RECORDED,
    coordinate_frame=CoordinateFrame(
        "arena-cartesian", "cm", AxisOrientation.X_RIGHT_Y_UP
    ),
)


def _pose(x: float, y: float, confidence: float = 0.95) -> Pose2D:
    return Pose2D(
        (
            Keypoint2D("bodycentre", Point2D(x, y), confidence),
            Keypoint2D("nose", Point2D(x + 1.0, y), 0.9),
        )
    )


class TrajectoryTests(unittest.TestCase):
    def test_bodycentre_is_used_without_inventing_a_confidence_threshold(self) -> None:
        frames = (
            SpatialFrame(CONTEXT, 0, _pose(1.0, 2.0, 0.2)),
            SpatialFrame(CONTEXT, 1, None),
        )
        samples = tuple(bodycentre_samples(frames))

        self.assertEqual(samples[0].point, Point2D(1.0, 2.0))
        self.assertEqual(samples[0].confidence, 0.2)
        self.assertIsNone(samples[1].point)
        self.assertIsNone(samples[1].confidence)

    def test_explicit_likelihood_threshold_turns_rejected_points_into_gaps(self) -> None:
        frames = (
            SpatialFrame(CONTEXT, 0, _pose(1.0, 2.0, 0.2)),
            SpatialFrame(CONTEXT, 1, _pose(2.0, 2.0, 0.95)),
        )
        samples = tuple(bodycentre_samples(frames, minimum_likelihood=0.8))

        self.assertIsNone(samples[0].point)
        self.assertEqual(samples[0].confidence, 0.2)
        self.assertEqual(samples[1].point, Point2D(2.0, 2.0))

    def test_displacement_and_path_length_do_not_cross_missing_frames(self) -> None:
        frames = (
            SpatialFrame(CONTEXT, 0, _pose(0.0, 0.0)),
            SpatialFrame(CONTEXT, 1, _pose(3.0, 4.0)),
            SpatialFrame(CONTEXT, 2, None),
            SpatialFrame(CONTEXT, 3, _pose(6.0, 8.0)),
            SpatialFrame(CONTEXT, 4, _pose(9.0, 12.0)),
        )
        steps = tuple(iter_trajectory_steps(bodycentre_samples(frames)))

        self.assertIsNone(steps[0].displacement)
        self.assertEqual(steps[1].displacement, 5.0)
        self.assertIsNone(steps[2].displacement)
        self.assertIsNone(steps[3].displacement)
        self.assertEqual(steps[4].displacement, 5.0)
        self.assertEqual(trajectory_path_length(steps), 10.0)

    def test_skipped_frame_index_is_treated_as_a_gap(self) -> None:
        samples = (
            PositionSample(0, Point2D(0.0, 0.0), 1.0),
            PositionSample(2, Point2D(3.0, 4.0), 1.0),
        )
        steps = tuple(iter_trajectory_steps(samples))

        self.assertIsNone(steps[1].displacement)
        self.assertIsNone(steps[1].movement_heading_radians)

    def test_movement_heading_and_turning_are_geometric_not_body_orientation(self) -> None:
        samples = (
            PositionSample(0, Point2D(0.0, 0.0), 1.0),
            PositionSample(1, Point2D(1.0, 0.0), 1.0),
            PositionSample(2, Point2D(1.0, 1.0), 1.0),
        )
        steps = tuple(iter_trajectory_steps(samples))

        self.assertAlmostEqual(steps[1].movement_heading_radians, 0.0)
        self.assertAlmostEqual(steps[2].movement_heading_radians, math.pi / 2)
        self.assertAlmostEqual(steps[2].turning_radians, math.pi / 2)

    def test_stationary_step_has_no_defined_movement_heading(self) -> None:
        samples = (
            PositionSample(0, Point2D(1.0, 1.0), 1.0),
            PositionSample(1, Point2D(1.0, 1.0), 1.0),
            PositionSample(2, Point2D(2.0, 1.0), 1.0),
        )
        steps = tuple(iter_trajectory_steps(samples))

        self.assertEqual(steps[1].displacement, 0.0)
        self.assertIsNone(steps[1].movement_heading_radians)
        self.assertIsNone(steps[2].turning_radians)

    def test_position_extraction_supports_other_recorded_keypoints(self) -> None:
        frame = SpatialFrame(CONTEXT, 0, _pose(5.0, 7.0))
        sample = next(keypoint_position_samples((frame,), keypoint_name="nose"))

        self.assertEqual(sample.point, Point2D(6.0, 7.0))
        with self.assertRaisesRegex(DataError, "Minimum keypoint likelihood"):
            tuple(bodycentre_samples((frame,), minimum_likelihood=1.1))

    def test_steps_reject_non_increasing_frame_indices(self) -> None:
        samples = (
            PositionSample(1, Point2D(0.0, 0.0), 1.0),
            PositionSample(1, Point2D(1.0, 0.0), 1.0),
        )
        with self.assertRaisesRegex(DataError, "strictly increasing"):
            tuple(iter_trajectory_steps(samples))


if __name__ == "__main__":
    unittest.main()
