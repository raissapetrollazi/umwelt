"""Tests for the Umwelt v0.3 boundary-conditioned movement model."""

from __future__ import annotations

import unittest

from umwelt.arena import RectangularArena
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
from umwelt.spatial_protocol import SpatialTrajectoryProtocol
from umwelt.world import (
    V03_ARENA_SCALE_CONDITIONS,
    ArenaScaleCondition,
    OpenFieldWorld,
    WorldZone,
)
from umwelt.world_model import (
    BOUNDARY_CONDITIONED_MODEL_ID,
    MINIMUM_POSITIVE_DISPLACEMENTS_PER_ZONE,
    _axis_reflections,
    fit_boundary_conditioned_movement_model,
    generate_boundary_conditioned_trajectory,
)


FRAME = CoordinateFrame("test-image", "px", AxisOrientation.X_RIGHT_Y_DOWN)
RECORDED_CONTEXT = SpatialContext(
    "mouse", "recording", ObservationSource.RECORDED, FRAME
)
SYNTHETIC_CONTEXT = SpatialContext(
    "mouse", "recording", ObservationSource.SYNTHETIC, FRAME
)
RECORDED_PROTOCOL = SpatialTrajectoryProtocol(
    "bodycentre", None, ObservationSource.RECORDED, FRAME
)
SYNTHETIC_PROTOCOL = SpatialTrajectoryProtocol(
    "bodycentre", None, ObservationSource.SYNTHETIC, FRAME
)
ARENA = RectangularArena("arena", FRAME, Point2D(0, 0), Point2D(10, 10))
RECORDED_WORLD = OpenFieldWorld(
    "open-field-v1", ARENA, ArenaScaleCondition("recorded", 1.0)
)


def frame(context: SpatialContext, index: int, x: float, y: float) -> SpatialFrame:
    return SpatialFrame(
        context,
        index,
        Pose2D((Keypoint2D("bodycentre", Point2D(x, y), 0.9),)),
    )


def fitting_steps() -> tuple:
    frames = (
        frame(RECORDED_CONTEXT, 0, 0.5, 5.0),
        frame(RECORDED_CONTEXT, 1, 1.0, 5.0),
        frame(RECORDED_CONTEXT, 2, 2.0, 5.0),
        frame(RECORDED_CONTEXT, 3, 3.0, 5.0),
        frame(RECORDED_CONTEXT, 4, 5.0, 5.0),
        frame(RECORDED_CONTEXT, 5, 5.0, 5.0),
        frame(RECORDED_CONTEXT, 6, 4.0, 5.0),
    )
    return tuple(RECORDED_PROTOCOL.trajectory_steps(frames))


class BoundaryConditionedMovementModelTests(unittest.TestCase):
    def test_reflection_count_tracks_repeated_axis_crossings(self) -> None:
        self.assertEqual(_axis_reflections(5.0, 0.0, 10.0), 0)
        self.assertEqual(_axis_reflections(11.0, 0.0, 10.0), 1)
        self.assertEqual(_axis_reflections(21.0, 0.0, 10.0), 2)
        self.assertEqual(_axis_reflections(-11.0, 0.0, 10.0), 2)

    def test_fit_isolates_zone_displacement_and_shares_baseline_parameters(
        self,
    ) -> None:
        model = fit_boundary_conditioned_movement_model(
            (fitting_steps(),),
            (RECORDED_WORLD,),
        )

        self.assertEqual(model.model_id, BOUNDARY_CONDITIONED_MODEL_ID)
        self.assertEqual(
            model.displacement_parameters(WorldZone.BOUNDARY_BAND).sample_count,
            2,
        )
        self.assertEqual(
            model.displacement_parameters(WorldZone.INTERIOR).sample_count,
            3,
        )
        self.assertEqual(model.shared_baseline.stationary_probability, 1 / 6)
        serialized = model.to_dict()
        self.assertEqual(
            serialized["environmental_dependency"],
            {
                "input": "transition-start-zone",
                "boundary_band_fraction": 0.10,
                "affected_parameter": "positive-displacement",
            },
        )
        self.assertEqual(
            serialized["minimum_positive_displacements_per_zone"],
            MINIMUM_POSITIVE_DISPLACEMENTS_PER_ZONE,
        )

    def test_fit_rejects_insufficient_zone_samples_without_fallback(self) -> None:
        interior_only = tuple(
            RECORDED_PROTOCOL.trajectory_steps(
                (
                    frame(RECORDED_CONTEXT, 0, 3.0, 3.0),
                    frame(RECORDED_CONTEXT, 1, 4.0, 3.0),
                    frame(RECORDED_CONTEXT, 2, 5.0, 3.0),
                )
            )
        )

        with self.assertRaisesRegex(DataError, "boundary-band requires at least"):
            fit_boundary_conditioned_movement_model(
                (interior_only,),
                (RECORDED_WORLD,),
            )

    def test_fit_rejects_synthetic_or_non_recorded_world_input(self) -> None:
        synthetic_steps = tuple(
            SYNTHETIC_PROTOCOL.trajectory_steps(
                (
                    frame(SYNTHETIC_CONTEXT, 0, 0.5, 5.0),
                    frame(SYNTHETIC_CONTEXT, 1, 1.0, 5.0),
                    frame(SYNTHETIC_CONTEXT, 2, 2.0, 5.0),
                )
            )
        )
        with self.assertRaisesRegex(DataError, "requires recorded data"):
            fit_boundary_conditioned_movement_model(
                (synthetic_steps,),
                (RECORDED_WORLD,),
            )

        compact_world = OpenFieldWorld(
            "open-field-v1", ARENA, ArenaScaleCondition("compact", 0.75)
        )
        with self.assertRaisesRegex(DataError, "recorded-scale"):
            fit_boundary_conditioned_movement_model(
                (fitting_steps(),),
                (compact_world,),
            )

    def test_fit_retains_outside_positions_in_baseline_and_reports_zone_exclusions(
        self,
    ) -> None:
        outside_steps = tuple(
            RECORDED_PROTOCOL.trajectory_steps(
                (
                    frame(RECORDED_CONTEXT, 0, 11.0, 5.0),
                    frame(RECORDED_CONTEXT, 1, 9.0, 5.0),
                    frame(RECORDED_CONTEXT, 2, 8.0, 5.0),
                )
            )
        )

        fitted = fit_boundary_conditioned_movement_model(
            (fitting_steps(), outside_steps),
            (RECORDED_WORLD, RECORDED_WORLD),
        )
        self.assertEqual(fitted.excluded_outside_position_count, 1)
        self.assertEqual(fitted.excluded_positive_transition_start_count, 1)
        self.assertEqual(
            fitted.to_dict()["zone_fitting_quality_control"][
                "outside_positions_retained_in_shared_baseline"
            ],
            True,
        )

    def test_generation_is_seeded_bounded_and_bound_to_each_condition(self) -> None:
        model = fit_boundary_conditioned_movement_model(
            (fitting_steps(),),
            (RECORDED_WORLD,),
        )

        for condition in V03_ARENA_SCALE_CONDITIONS:
            with self.subTest(condition=condition.condition_id):
                world = OpenFieldWorld("open-field-v1", ARENA, condition)
                first = generate_boundary_conditioned_trajectory(
                    model,
                    world=world,
                    frame_indices=tuple(range(100)),
                    subject_id="synthetic-mouse",
                    recording_id=f"synthetic-{condition.condition_id}",
                    seed=1729,
                )
                second = generate_boundary_conditioned_trajectory(
                    model,
                    world=world,
                    frame_indices=tuple(range(100)),
                    subject_id="synthetic-mouse",
                    recording_id=f"synthetic-{condition.condition_id}",
                    seed=1729,
                )

                self.assertEqual(first, second)
                self.assertEqual(first.world.condition, condition)
                self.assertGreaterEqual(first.reflection_count, 0)
                self.assertIs(first.context.source, ObservationSource.SYNTHETIC)
                self.assertTrue(
                    all(
                        world.arena.contains(item.pose.keypoint("bodycentre").point)
                        for item in first.frames
                    )
                )


if __name__ == "__main__":
    unittest.main()
