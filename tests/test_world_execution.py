"""Fixture-level checks for v0.3 folds, masks, pairing, and phase isolation."""

from __future__ import annotations

import unittest
from collections import Counter

from umwelt.arena import RectangularArena
from umwelt.datasets.roche_open_field import ROCHE_COORDINATE_FRAME
from umwelt.observations import ObservationSource
from umwelt.spatial import Keypoint2D, Point2D, Pose2D, SpatialContext, SpatialFrame
from umwelt.spatial_protocol import (
    ROCHE_CONTROL_DEVELOPMENT_SUBJECTS,
    ROCHE_CONTROL_FITTING_SUBJECTS,
    ROCHE_RECORDED_TRAJECTORY_PROTOCOL,
)
from umwelt.world_execution import execute_world_experiment
from umwelt.world_protocol import WorldExperimentProtocol
from umwelt.world_recordings import RecordedWorldSeries


def _series(subject_id: str) -> RecordedWorldSeries:
    context = SpatialContext(
        subject_id,
        f"source:{subject_id}",
        ObservationSource.RECORDED,
        ROCHE_COORDINATE_FRAME,
    )
    positions = (
        Point2D(5, 50),
        Point2D(7, 50),
        Point2D(9, 50),
        Point2D(20, 50),
        Point2D(30, 50),
        Point2D(40, 50),
        Point2D(50, 50),
        None,
        Point2D(60, 50),
    )
    frames = tuple(
        SpatialFrame(context, index, Pose2D((Keypoint2D("bodycentre", point),)))
        for index, point in enumerate(positions)
    )
    return RecordedWorldSeries(
        subject_id=subject_id,
        recording_id=context.recording_id,
        arena=RectangularArena(
            f"{subject_id}:arena",
            ROCHE_COORDINATE_FRAME,
            Point2D(0, 0),
            Point2D(100, 100),
        ),
        steps=tuple(ROCHE_RECORDED_TRAJECTORY_PROTOCOL.trajectory_steps(frames)),
        landmark_quality={},
    )


class WorldExecutionTests(unittest.TestCase):
    def test_complete_fixture_run_is_deterministic_and_phase_isolated(self) -> None:
        subjects = ROCHE_CONTROL_FITTING_SUBJECTS + ROCHE_CONTROL_DEVELOPMENT_SUBJECTS
        series = {subject_id: _series(subject_id) for subject_id in subjects}
        protocol = WorldExperimentProtocol(master_seed=1730, replicates=1)
        first = execute_world_experiment(protocol, series)
        second = execute_world_experiment(protocol, series)

        self.assertEqual(first, second)
        self.assertEqual(first.generated_series_count, 34)
        self.assertEqual(
            Counter(row["phase"] for row in first.records),
            {"cross-validation": 12, "legacy-diagnostic": 4, "intervention": 18},
        )
        self.assertEqual(len(first.comparison_rows), 12)
        self.assertEqual(len(first.preference.subject_medians), 6)
        self.assertTrue(
            all(
                row.subject_id in ROCHE_CONTROL_FITTING_SUBJECTS
                for row in first.comparison_rows
            )
        )
        self.assertTrue(
            all(
                row["excluded_from_preference"]
                for row in first.records
                if row["phase"] == "legacy-diagnostic"
            )
        )
        for row in first.records:
            self.assertEqual(row["synthetic"]["source_category"], "synthetic")
            self.assertEqual(
                row["synthetic"]["spatial_measurements"]["quality_control"][
                    "accepted_positions"
                ],
                8,
            )

        paired_seeds: dict[tuple[str, int], set[int]] = {}
        for row in first.records:
            key = (row["subject_id"], row["replicate"])
            paired_seeds.setdefault(key, set()).add(row["seed"])
        self.assertTrue(all(len(seeds) == 1 for seeds in paired_seeds.values()))
        for fold in first.models["cross_validation_folds"]:
            self.assertNotIn(fold["evaluation_subject_id"], fold["fitting_subject_ids"])


if __name__ == "__main__":
    unittest.main()
