"""Tests for dyad identity and gap-aware pair geometry."""

from __future__ import annotations

import unittest

from umwelt.dyad import DyadFrame, measure_dyad_frames
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


class DyadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.coordinates = CoordinateFrame(
            "camera-pixels", "px", AxisOrientation.X_RIGHT_Y_DOWN
        )
        self.resident = SpatialContext(
            "resident", "session-1", ObservationSource.RECORDED, self.coordinates
        )
        self.intruder = SpatialContext(
            "intruder", "session-1", ObservationSource.RECORDED, self.coordinates
        )

    def frame(
        self,
        index: int,
        resident: Point2D | None,
        intruder: Point2D | None,
    ) -> DyadFrame:
        def sample(context: SpatialContext, point: Point2D | None) -> SpatialFrame:
            pose = Pose2D((Keypoint2D("neck", point, 0.4),))
            return SpatialFrame(context, index, pose)

        return DyadFrame(
            "pair-1", sample(self.resident, resident), sample(self.intruder, intruder)
        )

    def test_signed_change_uses_adjacent_valid_pair_frames_only(self) -> None:
        measurement = measure_dyad_frames(
            (
                self.frame(0, Point2D(0, 0), Point2D(3, 4)),
                self.frame(1, Point2D(0, 0), Point2D(0, 3)),
                self.frame(2, Point2D(0, 0), None),
                self.frame(3, Point2D(0, 0), Point2D(0, 8)),
                self.frame(5, Point2D(0, 0), Point2D(0, 10)),
            )
        )

        self.assertEqual(measurement.pair_distances_px, (5.0, 3.0, 8.0, 10.0))
        self.assertEqual(measurement.signed_distance_changes_px, (-2.0,))
        self.assertEqual(measurement.resident_valid_position_count, 5)
        self.assertEqual(measurement.intruder_valid_position_count, 4)
        self.assertEqual(measurement.adjacent_frame_opportunity_count, 3)
        self.assertEqual(measurement.valid_pair_transition_count, 1)
        self.assertEqual(measurement.compact_dict()["source_category"], "recorded")

    def test_pair_identity_and_context_cannot_silently_change(self) -> None:
        initial = self.frame(0, Point2D(0, 0), Point2D(1, 0))
        swapped = DyadFrame(
            "pair-1",
            SpatialFrame(self.intruder, 1, initial.intruder.pose),
            SpatialFrame(self.resident, 1, initial.resident.pose),
        )
        with self.assertRaisesRegex(DataError, "cannot mix pair or source contexts"):
            measure_dyad_frames((initial, swapped))

        with self.assertRaisesRegex(DataError, "strictly increasing"):
            measure_dyad_frames((initial, initial))

        with self.assertRaisesRegex(DataError, "share one frame index"):
            DyadFrame(
                "pair-1",
                initial.resident,
                SpatialFrame(self.intruder, 1, initial.intruder.pose),
            )

    def test_empty_frames_are_rejected(self) -> None:
        with self.assertRaisesRegex(DataError, "at least one frame"):
            measure_dyad_frames(())


if __name__ == "__main__":
    unittest.main()
