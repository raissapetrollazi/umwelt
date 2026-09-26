"""Tests for the v0.4 split and frozen artifact boundary."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from umwelt.calms21 import Calms21Task1Sequence
from umwelt.errors import DataError
from umwelt.social_evaluation import SOCIAL_REPLICATES, aggregate_social_scores
from umwelt.social_lab import (
    _manifest,
    _score_split,
    _verify_development_artifacts,
    partition_calms21_training_ids,
)
from umwelt.social_model import SocialMovementFit
from umwelt.spatial_reporting import write_json


def short_sequence() -> Calms21Task1Sequence:
    coordinates = []
    scores = []
    for index in range(4):
        coordinates.append(
            [
                [[10.0 + index] * 7, [20.0] * 7],
                [[30.0 + index] * 7, [20.0] * 7],
            ]
        )
        scores.append([[0.9] * 7, [0.9] * 7])
    return Calms21Task1Sequence(
        "task1/train/mouse001_task1_annotator1", coordinates, scores
    )


def fit() -> SocialMovementFit:
    return SocialMovementFit(
        resident_alpha=0.5,
        resident_sigma_px=0,
        intruder_alpha=0.5,
        intruder_sigma_px=0,
        resident_directional_drift_px=1,
        fitting_sequence_count=1,
        resident_eligible_triplets=1,
        intruder_eligible_triplets=1,
        resident_excluded_large_step_triplets=0,
        intruder_excluded_large_step_triplets=0,
        resident_excluded_missing_or_gap_triplets=0,
        intruder_excluded_missing_or_gap_triplets=0,
        directional_drift_triplets=1,
    )


class SocialLabTests(unittest.TestCase):
    def test_identifier_partition_is_reproducible_and_disjoint(self) -> None:
        ids = tuple(
            f"task1/train/mouse{index:03d}_task1_annotator1" for index in range(1, 71)
        )

        fitting, development = partition_calms21_training_ids(ids)

        self.assertEqual(len(fitting), 56)
        self.assertEqual(len(development), 14)
        self.assertFalse(set(fitting) & set(development))
        self.assertEqual(
            (fitting, development),
            partition_calms21_training_ids(tuple(reversed(ids))),
        )

    def test_split_scoring_keeps_pair_unit_and_paired_replicates(self) -> None:
        sequence = short_sequence()

        recorded, scores, excluded, eligible = _score_split(
            (sequence,), selected_ids=frozenset((sequence.sequence_id,)), fit=fit()
        )

        self.assertEqual(eligible, (sequence.sequence_id,))
        self.assertEqual(excluded, ())
        self.assertEqual(len(scores), SOCIAL_REPLICATES * 2)
        self.assertEqual(recorded[0]["quality_control"]["frame_count"], 4)
        self.assertEqual(scores[0].resident_seed, scores[1].resident_seed)
        result = aggregate_social_scores(scores, eligible_pair_ids=eligible)
        self.assertEqual(result["eligible_pair_count"], 1)

    def test_changed_development_model_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_json(directory / "model-fit.json", fit().to_dict())
            write_json(directory / "provenance.json", {"source": {}})
            write_json(directory / "split.json", {})
            _manifest(directory)
            self.assertIn("source", _verify_development_artifacts(directory))

            (directory / "model-fit.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(DataError, "hash differs"):
                _verify_development_artifacts(directory)


if __name__ == "__main__":
    unittest.main()
