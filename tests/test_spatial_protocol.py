"""Tests for the pre-result v0.2 spatial protocol facts."""

from __future__ import annotations

import unittest
from math import pi

from umwelt.datasets.roche_open_field import ROCHE_COORDINATE_FRAME
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
from umwelt.spatial_protocol import (
    ROCHE_CONTROL_DEVELOPMENT_SUBJECTS,
    ROCHE_CONTROL_FITTING_SUBJECTS,
    ROCHE_CONTROL_SPLIT_HASH_PREFIX,
    ROCHE_CONTROL_SUBJECT_IDS,
    ROCHE_RECORDED_TRAJECTORY_PROTOCOL,
    derive_roche_control_split,
)
from umwelt.trajectory import PositionStatus


CONTEXT = SpatialContext(
    subject_id="mouse-1",
    recording_id="recording-1",
    source=ObservationSource.RECORDED,
    coordinate_frame=ROCHE_COORDINATE_FRAME,
)


def _frame(
    index: int,
    point: Point2D | None,
    confidence: float | None = 0.95,
) -> SpatialFrame:
    pose = Pose2D((Keypoint2D("bodycentre", point, confidence),))
    return SpatialFrame(context=CONTEXT, frame_index=index, pose=pose)


class SpatialProtocolTests(unittest.TestCase):
    def test_outcome_blind_split_is_exact_and_animal_disjoint(self) -> None:
        fitting, development = derive_roche_control_split()

        self.assertEqual(
            ROCHE_CONTROL_SPLIT_HASH_PREFIX,
            "umwelt-v0.2-control-split-v1|096e21e4d319130370aa4bb244b670f8",
        )
        self.assertEqual(
            fitting,
            (
                "16459-67049",
                "16459-67053",
                "16459-67060",
                "16459-67067",
                "16459-67071",
                "16459-67074",
            ),
        )
        self.assertEqual(development, ("16459-67064", "16459-67078"))
        self.assertEqual(fitting, ROCHE_CONTROL_FITTING_SUBJECTS)
        self.assertEqual(development, ROCHE_CONTROL_DEVELOPMENT_SUBJECTS)
        self.assertFalse(set(fitting) & set(development))
        self.assertEqual(
            set(fitting) | set(development), set(ROCHE_CONTROL_SUBJECT_IDS)
        )

    def test_recorded_trajectory_policy_is_explicit_and_provenance_ready(self) -> None:
        protocol = ROCHE_RECORDED_TRAJECTORY_PROTOCOL

        self.assertEqual(protocol.position_rule(CONTEXT).context, CONTEXT)
        self.assertEqual(
            protocol.to_dict(),
            {
                "input": {
                    "source_category": "recorded",
                    "coordinate_frame": {
                        "frame_id": "roche-open-field-image",
                        "unit": "px",
                        "axis_orientation": "x-right-y-down",
                    },
                },
                "position": {
                    "keypoint_name": "bodycentre",
                    "minimum_likelihood": None,
                },
                "continuity": {
                    "interpolation_applied_by_protocol": False,
                    "smoothing_applied_by_protocol": False,
                    "valid_displacement": ("adjacent-frames-with-accepted-endpoints"),
                    "turning": ("two-consecutive-nonstationary-valid-movements"),
                    "zero_displacement": "retain-and-break-turning",
                },
            },
        )

    def test_recorded_policy_drives_confidence_gap_and_turning_behavior(self) -> None:
        frames = (
            _frame(0, Point2D(0.0, 0.0), 0.01),
            _frame(1, Point2D(1.0, 0.0), None),
            _frame(2, None),
            _frame(3, Point2D(3.0, 0.0)),
            SpatialFrame(context=CONTEXT, frame_index=4, pose=None),
            _frame(5, Point2D(5.0, 0.0)),
            _frame(6, Point2D(5.0, 0.0)),
            _frame(7, Point2D(5.0, -1.0)),
            _frame(8, Point2D(6.0, -1.0)),
        )

        steps = tuple(ROCHE_RECORDED_TRAJECTORY_PROTOCOL.trajectory_steps(frames))

        self.assertEqual(
            [step.status for step in steps],
            [
                PositionStatus.ACCEPTED,
                PositionStatus.ACCEPTED,
                PositionStatus.SOURCE_COORDINATE_MISSING,
                PositionStatus.ACCEPTED,
                PositionStatus.SOURCE_POSE_MISSING,
                PositionStatus.ACCEPTED,
                PositionStatus.ACCEPTED,
                PositionStatus.ACCEPTED,
                PositionStatus.ACCEPTED,
            ],
        )
        self.assertEqual(
            steps[0].rule,
            ROCHE_RECORDED_TRAJECTORY_PROTOCOL.position_rule(CONTEXT),
        )
        self.assertEqual(steps[0].confidence, 0.01)
        self.assertIsNone(steps[1].confidence)
        self.assertEqual(
            [step.displacement for step in steps],
            [None, 1.0, None, None, None, None, 0.0, 1.0, 1.0],
        )
        self.assertTrue(all(step.turning_radians is None for step in steps[:8]))
        self.assertAlmostEqual(steps[8].turning_radians, -pi / 2)

    def test_recorded_policy_rejects_synthetic_or_different_coordinate_input(
        self,
    ) -> None:
        synthetic_context = SpatialContext(
            subject_id="synthetic-1",
            recording_id="synthetic-recording-1",
            source=ObservationSource.SYNTHETIC,
            coordinate_frame=ROCHE_COORDINATE_FRAME,
        )
        different_frame_context = SpatialContext(
            subject_id="mouse-1",
            recording_id="recording-1",
            source=ObservationSource.RECORDED,
            coordinate_frame=CoordinateFrame(
                "other-image", "px", AxisOrientation.X_RIGHT_Y_DOWN
            ),
        )

        with self.assertRaisesRegex(DataError, "observation source"):
            ROCHE_RECORDED_TRAJECTORY_PROTOCOL.position_rule(synthetic_context)
        with self.assertRaisesRegex(DataError, "coordinate frame"):
            ROCHE_RECORDED_TRAJECTORY_PROTOCOL.position_rule(different_frame_context)


if __name__ == "__main__":
    unittest.main()
