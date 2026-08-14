"""Tests for the frozen v0.2 spatial evaluation contract."""

from __future__ import annotations

import unittest
from dataclasses import replace
from math import pi

from umwelt.datasets.roche_open_field import ROCHE_COORDINATE_FRAME
from umwelt.errors import DataError
from umwelt.observations import ObservationSource
from umwelt.spatial import Keypoint2D, Point2D, Pose2D, SpatialContext, SpatialFrame
from umwelt.spatial_evaluation import (
    DISTRIBUTION_SUMMARY_FIELDS,
    REGISTERED_SPATIAL_METRIC_IDS,
    ROCHE_SPATIAL_EVALUATION_PROTOCOL,
    SPATIAL_QUALITY_CONTROL_IDS,
    compare_spatial_measurements,
    distribution_summary,
    measure_spatial_trajectory,
)
from umwelt.spatial_protocol import (
    ROCHE_RECORDED_TRAJECTORY_PROTOCOL,
    SpatialTrajectoryProtocol,
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


def _frame(
    context: SpatialContext,
    index: int,
    point: Point2D | None,
    confidence: float | None = 0.9,
) -> SpatialFrame:
    return SpatialFrame(
        context,
        index,
        Pose2D((Keypoint2D("bodycentre", point, confidence),)),
    )


class SpatialEvaluationProtocolTests(unittest.TestCase):
    def test_registered_metrics_and_quality_control_are_exact(self) -> None:
        protocol = ROCHE_SPATIAL_EVALUATION_PROTOCOL

        self.assertEqual(protocol.registered_metric_ids, REGISTERED_SPATIAL_METRIC_IDS)
        self.assertEqual(protocol.quality_control_ids, SPATIAL_QUALITY_CONTROL_IDS)
        self.assertEqual(
            protocol.distribution_summary_fields,
            DISTRIBUTION_SUMMARY_FIELDS,
        )

    def test_protocol_serializes_definitions_units_and_exclusions(self) -> None:
        serialized = ROCHE_SPATIAL_EVALUATION_PROTOCOL.to_dict()

        self.assertEqual(serialized["unit_of_description"], "animal-recording")
        self.assertEqual(
            [item["id"] for item in serialized["registered_metrics"]],
            list(REGISTERED_SPATIAL_METRIC_IDS),
        )
        self.assertEqual(
            [item["id"] for item in serialized["quality_control"]],
            list(SPATIAL_QUALITY_CONTROL_IDS),
        )
        self.assertIn(
            "boundary-distance",
            serialized["deferred_pending_validated_arena_bounds"],
        )
        self.assertIn(
            "spatial-occupancy",
            serialized["deferred_pending_validated_arena_bounds"],
        )
        self.assertIn("speed", serialized["excluded_due_to_missing_source_support"])
        self.assertIsNone(serialized["acceptance_threshold"])

    def test_registered_contract_cannot_be_changed_silently(self) -> None:
        with self.assertRaisesRegex(DataError, "frozen protocol"):
            replace(
                ROCHE_SPATIAL_EVALUATION_PROTOCOL,
                registered_metric_ids=("speed",),
            )
        with self.assertRaisesRegex(DataError, "quality control"):
            replace(
                ROCHE_SPATIAL_EVALUATION_PROTOCOL,
                quality_control_ids=("accepted-fraction",),
            )

    def test_measurement_preserves_qc_exposure_and_zero_displacement(self) -> None:
        frames = (
            _frame(RECORDED_CONTEXT, 0, Point2D(0, 0), 0.1),
            _frame(RECORDED_CONTEXT, 1, Point2D(0, 0), None),
            _frame(RECORDED_CONTEXT, 2, None, 0.2),
            SpatialFrame(RECORDED_CONTEXT, 3, None),
            _frame(RECORDED_CONTEXT, 4, Point2D(3, 4), 0.9),
            _frame(RECORDED_CONTEXT, 5, Point2D(6, 8), 0.8),
        )

        measured = measure_spatial_trajectory(
            frames,
            trajectory_protocol=ROCHE_RECORDED_TRAJECTORY_PROTOCOL,
        )
        quality = measured.quality_control

        self.assertEqual(measured.displacements_px, (0.0, 5.0))
        self.assertEqual(measured.path_length_px, 5.0)
        self.assertEqual(quality.adjacent_transition_opportunities, 5)
        self.assertEqual(quality.valid_transition_count, 2)
        self.assertEqual(quality.zero_transition_count, 1)
        self.assertEqual(quality.valid_turn_count, 0)
        self.assertEqual(quality.source_likelihood_missing_count, 2)
        self.assertEqual(
            dict(quality.position_status_counts),
            {
                "accepted": 4,
                "source-pose-missing": 1,
                "source-coordinate-missing": 1,
                "below-likelihood-threshold": 0,
                "likelihood-missing": 0,
            },
        )

    def test_comparison_uses_absolute_turning_and_equal_exposure(self) -> None:
        recorded = measure_spatial_trajectory(
            (
                _frame(RECORDED_CONTEXT, 0, Point2D(0, 0)),
                _frame(RECORDED_CONTEXT, 1, Point2D(1, 0)),
                _frame(RECORDED_CONTEXT, 2, Point2D(1, -1)),
            ),
            trajectory_protocol=ROCHE_RECORDED_TRAJECTORY_PROTOCOL,
        )
        synthetic = measure_spatial_trajectory(
            (
                _frame(SYNTHETIC_CONTEXT, 0, Point2D(0, 0), None),
                _frame(SYNTHETIC_CONTEXT, 1, Point2D(1, 0), None),
                _frame(SYNTHETIC_CONTEXT, 2, Point2D(1, 1), None),
            ),
            trajectory_protocol=SYNTHETIC_PROTOCOL,
        )

        comparison = compare_spatial_measurements(recorded, synthetic)

        self.assertEqual(recorded.absolute_turnings_radians, (pi / 2,))
        self.assertEqual(synthetic.absolute_turnings_radians, (pi / 2,))
        self.assertEqual(comparison.path_length_absolute_difference_px, 0.0)
        self.assertEqual(comparison.adjacent_displacement_wasserstein_px, 0.0)
        self.assertEqual(comparison.absolute_turning_wasserstein_radians, 0.0)

    def test_distribution_summary_uses_frozen_linear_quantiles(self) -> None:
        summary = distribution_summary((0.0, 2.0, 4.0, 8.0))

        self.assertEqual(tuple(summary), DISTRIBUTION_SUMMARY_FIELDS)
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["q25"], 1.5)
        self.assertEqual(summary["median"], 3.0)
        self.assertAlmostEqual(summary["q90"], 6.8)


if __name__ == "__main__":
    unittest.main()
