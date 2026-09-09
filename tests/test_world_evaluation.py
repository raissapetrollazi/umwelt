"""Tests for v0.3 environment-linked trajectory measurements."""

from __future__ import annotations

import unittest
from dataclasses import replace

from umwelt.arena import RectangularArena
from umwelt.datasets.roche_open_field import ROCHE_COORDINATE_FRAME
from umwelt.errors import DataError
from umwelt.observations import ObservationSource
from umwelt.spatial import Keypoint2D, Point2D, Pose2D, SpatialContext, SpatialFrame
from umwelt.spatial_protocol import (
    ROCHE_RECORDED_TRAJECTORY_PROTOCOL,
    SpatialTrajectoryProtocol,
)
from umwelt.world import ArenaScaleCondition, OpenFieldWorld, WorldZone
from umwelt.world_evaluation import (
    compare_world_measurements,
    measure_world_trajectory,
)


RECORDED_CONTEXT = SpatialContext(
    "mouse-1",
    "recorded-1",
    ObservationSource.RECORDED,
    ROCHE_COORDINATE_FRAME,
)
SYNTHETIC_CONTEXT = SpatialContext(
    "mouse-1",
    "synthetic-1",
    ObservationSource.SYNTHETIC,
    ROCHE_COORDINATE_FRAME,
)
SYNTHETIC_PROTOCOL = SpatialTrajectoryProtocol(
    "bodycentre",
    None,
    ObservationSource.SYNTHETIC,
    ROCHE_COORDINATE_FRAME,
)


def _frame(context: SpatialContext, index: int, point: Point2D | None) -> SpatialFrame:
    return SpatialFrame(
        context,
        index,
        Pose2D((Keypoint2D("bodycentre", point, None),)),
    )


class WorldEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        arena = RectangularArena(
            "recording-1:arena",
            ROCHE_COORDINATE_FRAME,
            Point2D(0.0, 0.0),
            Point2D(100.0, 100.0),
        )
        self.world = OpenFieldWorld(
            "roche-open-field-v1",
            arena,
            ArenaScaleCondition("recorded", 1.0),
        )

    def test_measurement_assigns_positions_and_movements_by_start_zone(self) -> None:
        measured = measure_world_trajectory(
            (
                _frame(RECORDED_CONTEXT, 0, Point2D(5.0, 50.0)),
                _frame(RECORDED_CONTEXT, 1, Point2D(15.0, 50.0)),
                _frame(RECORDED_CONTEXT, 2, Point2D(15.0, 50.0)),
                _frame(RECORDED_CONTEXT, 3, None),
                _frame(RECORDED_CONTEXT, 4, Point2D(50.0, 50.0)),
                _frame(RECORDED_CONTEXT, 5, Point2D(50.0, 40.0)),
            ),
            trajectory_protocol=ROCHE_RECORDED_TRAJECTORY_PROTOCOL,
            world=self.world,
        )

        self.assertEqual(
            dict(measured.zone_position_counts),
            {"boundary-band": 1, "interior": 4},
        )
        self.assertEqual(measured.boundary_band_occupancy_fraction, 0.2)
        self.assertEqual(
            measured.positive_displacements(WorldZone.BOUNDARY_BAND), (10.0,)
        )
        self.assertEqual(measured.positive_displacements(WorldZone.INTERIOR), (10.0,))
        self.assertEqual(measured.spatial.displacements_px, (10.0, 0.0, 10.0))

    def test_recorded_outside_positions_remain_quality_control(self) -> None:
        measured = measure_world_trajectory(
            (
                _frame(RECORDED_CONTEXT, 0, Point2D(105.0, 50.0)),
                _frame(RECORDED_CONTEXT, 1, Point2D(95.0, 50.0)),
                _frame(RECORDED_CONTEXT, 2, Point2D(85.0, 50.0)),
            ),
            trajectory_protocol=ROCHE_RECORDED_TRAJECTORY_PROTOCOL,
            world=self.world,
        )

        self.assertEqual(measured.outside_arena_position_count, 1)
        self.assertEqual(measured.in_arena_position_count, 2)
        self.assertEqual(measured.unclassified_positive_transition_start_count, 1)
        self.assertEqual(
            measured.positive_displacements(WorldZone.BOUNDARY_BAND), (10.0,)
        )
        self.assertEqual(measured.spatial.path_length_px, 20.0)

    def test_synthetic_outside_position_is_an_invariant_violation(self) -> None:
        with self.assertRaisesRegex(DataError, "inside their declared world"):
            measure_world_trajectory(
                (_frame(SYNTHETIC_CONTEXT, 0, Point2D(101.0, 50.0)),),
                trajectory_protocol=SYNTHETIC_PROTOCOL,
                world=self.world,
            )

    def test_measurement_validation_rejects_incoherent_zone_counts(self) -> None:
        measured = measure_world_trajectory(
            (_frame(RECORDED_CONTEXT, 0, Point2D(50.0, 50.0)),),
            trajectory_protocol=ROCHE_RECORDED_TRAJECTORY_PROTOCOL,
            world=self.world,
        )

        with self.assertRaisesRegex(DataError, "every accepted spatial position"):
            replace(
                measured,
                zone_position_counts=(("boundary-band", 0), ("interior", 0)),
            )

    def test_compact_measurement_preserves_world_and_metric_exposure(self) -> None:
        measured = measure_world_trajectory(
            (
                _frame(RECORDED_CONTEXT, 0, Point2D(5.0, 50.0)),
                _frame(RECORDED_CONTEXT, 1, Point2D(15.0, 50.0)),
            ),
            trajectory_protocol=ROCHE_RECORDED_TRAJECTORY_PROTOCOL,
            world=self.world,
        ).compact_dict()

        self.assertEqual(measured["source_category"], "recorded")
        self.assertEqual(measured["world"]["condition_id"], "recorded")
        occupancy = measured["registered_metrics"]["boundary-band-occupancy-fraction"]
        self.assertEqual(occupancy["value"], 0.5)
        self.assertEqual(occupancy["in-arena-position-count"], 2)
        by_zone = measured["registered_metrics"][
            "zone-conditioned-positive-displacement-px-distribution"
        ]
        self.assertEqual(by_zone["boundary-band"]["count"], 1)
        self.assertEqual(by_zone["interior"]["count"], 0)

    def test_identical_recorded_and_synthetic_geometry_has_zero_difference(
        self,
    ) -> None:
        recorded = measure_world_trajectory(
            (
                _frame(RECORDED_CONTEXT, 0, Point2D(5.0, 50.0)),
                _frame(RECORDED_CONTEXT, 1, Point2D(15.0, 50.0)),
                _frame(RECORDED_CONTEXT, 2, Point2D(25.0, 50.0)),
            ),
            trajectory_protocol=ROCHE_RECORDED_TRAJECTORY_PROTOCOL,
            world=self.world,
        )
        synthetic = measure_world_trajectory(
            (
                _frame(SYNTHETIC_CONTEXT, 0, Point2D(5.0, 50.0)),
                _frame(SYNTHETIC_CONTEXT, 1, Point2D(15.0, 50.0)),
                _frame(SYNTHETIC_CONTEXT, 2, Point2D(25.0, 50.0)),
            ),
            trajectory_protocol=SYNTHETIC_PROTOCOL,
            world=self.world,
        )

        comparison = compare_world_measurements(recorded, synthetic)

        self.assertEqual(comparison.spatial.path_length_absolute_difference_px, 0.0)
        self.assertEqual(comparison.boundary_band_occupancy_absolute_difference, 0.0)
        self.assertEqual(
            dict(comparison.zone_positive_displacement_wasserstein_px),
            {"boundary-band": 0.0, "interior": 0.0},
        )

    def test_comparison_rejects_different_world_conditions(self) -> None:
        recorded = measure_world_trajectory(
            (_frame(RECORDED_CONTEXT, 0, Point2D(50.0, 50.0)),),
            trajectory_protocol=ROCHE_RECORDED_TRAJECTORY_PROTOCOL,
            world=self.world,
        )
        other_world = OpenFieldWorld(
            "roche-open-field-v1",
            self.world.source_arena,
            ArenaScaleCondition("expanded", 1.25),
        )
        synthetic = measure_world_trajectory(
            (_frame(SYNTHETIC_CONTEXT, 0, Point2D(50.0, 50.0)),),
            trajectory_protocol=SYNTHETIC_PROTOCOL,
            world=other_world,
        )

        with self.assertRaisesRegex(DataError, "same declared world"):
            compare_world_measurements(recorded, synthetic)


if __name__ == "__main__":
    unittest.main()
