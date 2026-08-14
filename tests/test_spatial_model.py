"""Tests for the v0.2 spatial generative baseline."""

from __future__ import annotations

import unittest

from umwelt.arena import RectangularArena
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
from umwelt.spatial_model import fit_spatial_movement_model, generate_spatial_trajectory
from umwelt.spatial_protocol import SpatialTrajectoryProtocol

FRAME = CoordinateFrame("test-image", "px", AxisOrientation.X_RIGHT_Y_DOWN)
CONTEXT = SpatialContext("mouse", "recording", ObservationSource.RECORDED, FRAME)
ARENA = RectangularArena("arena", FRAME, Point2D(0, 0), Point2D(10, 10))
PROTOCOL = SpatialTrajectoryProtocol("bodycentre", None, ObservationSource.RECORDED, FRAME)


def frame(index: int, x: float, y: float) -> SpatialFrame:
    return SpatialFrame(
        CONTEXT,
        index,
        Pose2D((Keypoint2D("bodycentre", Point2D(x, y), 0.9),)),
    )


class SpatialModelTests(unittest.TestCase):
    def test_fit_and_generation_are_seeded_and_bounded(self) -> None:
        frames = (
            frame(0, 2, 2),
            frame(1, 3, 2),
            frame(2, 4, 2),
            frame(3, 4, 2),
            frame(4, 4, 3),
        )
        steps = tuple(PROTOCOL.trajectory_steps(frames))
        model = fit_spatial_movement_model((steps,), (ARENA,))
        first = generate_spatial_trajectory(
            model,
            arena=ARENA,
            frame_indices=(0, 1, 2, 3, 4, 5),
            subject_id="synthetic",
            recording_id="s1",
            seed=1729,
        )
        second = generate_spatial_trajectory(
            model,
            arena=ARENA,
            frame_indices=(0, 1, 2, 3, 4, 5),
            subject_id="synthetic",
            recording_id="s1",
            seed=1729,
        )
        self.assertEqual(first, second)
        self.assertTrue(
            all(ARENA.contains(item.pose.keypoint("bodycentre").point) for item in first)
        )
        self.assertEqual(model.model_id, "persistent-reflecting-random-walk-v1")


if __name__ == "__main__":
    unittest.main()
