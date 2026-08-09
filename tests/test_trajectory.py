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
    PositionRule,
    PositionSample,
    PositionStatus,
    TrajectoryPathLength,
    bodycentre_samples,
    iter_trajectory_steps,
    keypoint_position_samples,
    trajectory_path_length,
)


CONTEXT_Y_UP = SpatialContext(
    subject_id="mouse-1",
    recording_id="recording-1",
    source=ObservationSource.RECORDED,
    coordinate_frame=CoordinateFrame(
        "arena-cartesian", "cm", AxisOrientation.X_RIGHT_Y_UP
    ),
)
CONTEXT_Y_DOWN = SpatialContext(
    subject_id="mouse-1",
    recording_id="recording-1",
    source=ObservationSource.RECORDED,
    coordinate_frame=CoordinateFrame(
        "camera-image", "px", AxisOrientation.X_RIGHT_Y_DOWN
    ),
)


def _pose(
    x: float,
    y: float,
    confidence: float | None = 0.95,
) -> Pose2D:
    return Pose2D(
        (
            Keypoint2D("bodycentre", Point2D(x, y), confidence),
            Keypoint2D("nose", Point2D(x + 1.0, y), 0.9),
        )
    )


def _frame(
    index: int,
    pose: Pose2D | None,
    *,
    context: SpatialContext = CONTEXT_Y_UP,
) -> SpatialFrame:
    return SpatialFrame(context=context, frame_index=index, pose=pose)


def _sample(
    index: int,
    point: Point2D,
    *,
    context: SpatialContext = CONTEXT_Y_UP,
    keypoint_name: str = "bodycentre",
    minimum_likelihood: float | None = None,
) -> PositionSample:
    return PositionSample(
        rule=PositionRule(context, keypoint_name, minimum_likelihood),
        frame_index=index,
        source_point=point,
        confidence=1.0,
        status=PositionStatus.ACCEPTED,
    )


class TrajectoryTests(unittest.TestCase):
    def test_bodycentre_preserves_rule_and_source_gap_without_default_threshold(
        self,
    ) -> None:
        samples = tuple(
            bodycentre_samples((_frame(0, _pose(1.0, 2.0, 0.2)), _frame(1, None)))
        )

        self.assertEqual(samples[0].point, Point2D(1.0, 2.0))
        self.assertEqual(samples[0].source_point, Point2D(1.0, 2.0))
        self.assertEqual(samples[0].confidence, 0.2)
        self.assertEqual(samples[0].status, PositionStatus.ACCEPTED)
        self.assertEqual(samples[0].rule.keypoint_name, "bodycentre")
        self.assertIsNone(samples[0].rule.minimum_likelihood)
        self.assertIsNone(samples[1].point)
        self.assertEqual(samples[1].status, PositionStatus.SOURCE_POSE_MISSING)

    def test_threshold_rejection_preserves_source_measurement_and_reason(self) -> None:
        samples = tuple(
            bodycentre_samples(
                (_frame(0, _pose(1.0, 2.0, 0.2)), _frame(1, _pose(2.0, 2.0))),
                minimum_likelihood=0.8,
            )
        )

        self.assertIsNone(samples[0].point)
        self.assertEqual(samples[0].source_point, Point2D(1.0, 2.0))
        self.assertEqual(samples[0].confidence, 0.2)
        self.assertEqual(samples[0].status, PositionStatus.BELOW_LIKELIHOOD_THRESHOLD)
        self.assertEqual(samples[0].rule.minimum_likelihood, 0.8)
        self.assertEqual(samples[1].point, Point2D(2.0, 2.0))

    def test_displacement_and_path_length_do_not_cross_missing_frames(self) -> None:
        frames = (
            _frame(0, _pose(0.0, 0.0)),
            _frame(1, _pose(3.0, 4.0)),
            _frame(2, None),
            _frame(3, _pose(6.0, 8.0)),
            _frame(4, _pose(9.0, 12.0)),
        )
        steps = tuple(iter_trajectory_steps(bodycentre_samples(frames)))

        self.assertIsNone(steps[0].displacement)
        self.assertEqual(steps[1].displacement, 5.0)
        self.assertIsNone(steps[2].displacement)
        self.assertIsNone(steps[3].displacement)
        self.assertEqual(steps[4].displacement, 5.0)
        path_length = trajectory_path_length(steps)
        self.assertIsInstance(path_length, TrajectoryPathLength)
        self.assertEqual(path_length.value, 10.0)
        self.assertEqual(path_length.unit, "cm")
        self.assertEqual(path_length.context, CONTEXT_Y_UP)
        self.assertEqual(path_length.rule.keypoint_name, "bodycentre")

    def test_skipped_frame_index_is_treated_as_a_gap(self) -> None:
        steps = tuple(
            iter_trajectory_steps(
                (_sample(0, Point2D(0.0, 0.0)), _sample(2, Point2D(3.0, 4.0)))
            )
        )

        self.assertIsNone(steps[1].displacement)
        self.assertIsNone(steps[1].movement_heading_radians)

    def test_movement_heading_and_turning_are_geometric_not_body_orientation(
        self,
    ) -> None:
        steps = tuple(
            iter_trajectory_steps(
                (
                    _sample(0, Point2D(0.0, 0.0)),
                    _sample(1, Point2D(1.0, 0.0)),
                    _sample(2, Point2D(1.0, 1.0)),
                )
            )
        )

        self.assertAlmostEqual(steps[1].movement_heading_radians, 0.0)
        self.assertAlmostEqual(steps[2].movement_heading_radians, math.pi / 2)
        self.assertAlmostEqual(steps[2].turning_radians, math.pi / 2)

    def test_stationary_step_has_no_defined_movement_heading(self) -> None:
        steps = tuple(
            iter_trajectory_steps(
                (
                    _sample(0, Point2D(1.0, 1.0)),
                    _sample(1, Point2D(1.0, 1.0)),
                    _sample(2, Point2D(2.0, 1.0)),
                )
            )
        )

        self.assertEqual(steps[1].displacement, 0.0)
        self.assertIsNone(steps[1].movement_heading_radians)
        self.assertIsNone(steps[2].turning_radians)

    def test_position_extraction_supports_other_recorded_keypoints(self) -> None:
        frame = _frame(0, _pose(5.0, 7.0))
        sample = next(keypoint_position_samples((frame,), keypoint_name="nose"))

        self.assertEqual(sample.point, Point2D(6.0, 7.0))
        self.assertEqual(sample.rule.keypoint_name, "nose")
        with self.assertRaisesRegex(DataError, "Minimum keypoint likelihood"):
            tuple(bodycentre_samples((frame,), minimum_likelihood=1.1))

    def test_steps_reject_non_increasing_frame_indices(self) -> None:
        samples = (
            _sample(1, Point2D(0.0, 0.0)),
            _sample(1, Point2D(1.0, 0.0)),
        )
        with self.assertRaisesRegex(DataError, "strictly increasing"):
            tuple(iter_trajectory_steps(samples))

    def test_threshold_rejects_but_preserves_point_without_likelihood(self) -> None:
        sample = next(
            bodycentre_samples(
                (_frame(0, _pose(1.0, 2.0, None)),),
                minimum_likelihood=0.8,
            )
        )

        self.assertIsNone(sample.point)
        self.assertEqual(sample.source_point, Point2D(1.0, 2.0))
        self.assertIsNone(sample.confidence)
        self.assertEqual(sample.status, PositionStatus.LIKELIHOOD_MISSING)

    def test_source_coordinate_gap_is_distinct_from_missing_pose(self) -> None:
        missing_coordinate = Pose2D((Keypoint2D("bodycentre", None, 0.1),))
        samples = tuple(
            bodycentre_samples((_frame(0, missing_coordinate), _frame(1, None)))
        )

        self.assertEqual(samples[0].status, PositionStatus.SOURCE_COORDINATE_MISSING)
        self.assertEqual(samples[0].confidence, 0.1)
        self.assertEqual(samples[1].status, PositionStatus.SOURCE_POSE_MISSING)

    def test_image_y_down_is_normalized_to_counterclockwise_heading(self) -> None:
        steps = tuple(
            iter_trajectory_steps(
                (
                    _sample(0, Point2D(0.0, 1.0), context=CONTEXT_Y_DOWN),
                    _sample(1, Point2D(0.0, 0.0), context=CONTEXT_Y_DOWN),
                )
            )
        )

        self.assertAlmostEqual(steps[1].movement_heading_radians, math.pi / 2)

    def test_y_down_turning_is_normalized_and_resets_after_rejected_gap(self) -> None:
        frames = (
            _frame(0, _pose(0.0, 1.0), context=CONTEXT_Y_DOWN),
            _frame(1, _pose(1.0, 1.0), context=CONTEXT_Y_DOWN),
            _frame(2, _pose(1.0, 0.0), context=CONTEXT_Y_DOWN),
            _frame(3, _pose(1.0, -1.0, 0.1), context=CONTEXT_Y_DOWN),
            _frame(4, _pose(2.0, -1.0), context=CONTEXT_Y_DOWN),
            _frame(5, _pose(3.0, -1.0), context=CONTEXT_Y_DOWN),
        )
        steps = tuple(
            iter_trajectory_steps(bodycentre_samples(frames, minimum_likelihood=0.8))
        )

        self.assertAlmostEqual(steps[2].turning_radians, math.pi / 2)
        self.assertEqual(steps[3].status, PositionStatus.BELOW_LIKELIHOOD_THRESHOLD)
        self.assertIsNone(steps[4].displacement)
        self.assertIsNone(steps[5].turning_radians)

    def test_west_heading_is_canonical_across_axis_orientations(self) -> None:
        y_up = tuple(
            iter_trajectory_steps(
                (_sample(0, Point2D(1.0, 0.0)), _sample(1, Point2D(0.0, 0.0)))
            )
        )
        y_down = tuple(
            iter_trajectory_steps(
                (
                    _sample(0, Point2D(1.0, 0.0), context=CONTEXT_Y_DOWN),
                    _sample(1, Point2D(0.0, 0.0), context=CONTEXT_Y_DOWN),
                )
            )
        )

        self.assertEqual(y_up[1].movement_heading_radians, math.pi)
        self.assertEqual(y_down[1].movement_heading_radians, math.pi)

    def test_position_extraction_rejects_context_mixing(self) -> None:
        frames = (
            _frame(0, _pose(0.0, 0.0)),
            _frame(1, _pose(1.0, 0.0), context=CONTEXT_Y_DOWN),
        )

        with self.assertRaisesRegex(DataError, "cannot mix spatial contexts"):
            tuple(bodycentre_samples(frames))

    def test_trajectory_rejects_context_keypoint_or_threshold_mixing(self) -> None:
        other_context = SpatialContext(
            subject_id="mouse-2",
            recording_id="recording-2",
            source=ObservationSource.RECORDED,
            coordinate_frame=CONTEXT_Y_UP.coordinate_frame,
        )
        variants = (
            _sample(1, Point2D(1.0, 0.0), context=other_context),
            _sample(1, Point2D(1.0, 0.0), keypoint_name="nose"),
            _sample(1, Point2D(1.0, 0.0), minimum_likelihood=0.8),
        )
        for variant in variants:
            with self.subTest(rule=variant.rule):
                with self.assertRaisesRegex(DataError, "derivation rules"):
                    tuple(
                        iter_trajectory_steps((_sample(0, Point2D(0.0, 0.0)), variant))
                    )

    def test_path_length_rejects_empty_or_mixed_rules(self) -> None:
        with self.assertRaisesRegex(DataError, "at least one step"):
            trajectory_path_length(())

        first = next(iter_trajectory_steps((_sample(0, Point2D(0.0, 0.0)),)))
        second = next(
            iter_trajectory_steps(
                (_sample(0, Point2D(0.0, 0.0), keypoint_name="nose"),)
            )
        )
        with self.assertRaisesRegex(DataError, "derivation rules"):
            trajectory_path_length((first, second))


if __name__ == "__main__":
    unittest.main()
