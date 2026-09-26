"""Tests for equal-pair aggregation and the frozen social preference rule."""

from __future__ import annotations

import unittest
from dataclasses import replace

from umwelt.errors import DataError
from umwelt.social_evaluation import (
    SOCIAL_REPLICATES,
    SocialReplicateScore,
    aggregate_social_scores,
)
from umwelt.social_model import SOCIAL_MODEL_IDS


def score(
    pair_id: str,
    model_id: str,
    replicate: int,
    distance: float | None,
    change: float | None,
) -> SocialReplicateScore:
    return SocialReplicateScore(
        pair_id=pair_id,
        model_id=model_id,
        replicate=replicate,
        resident_seed=1,
        intruder_seed=2,
        distance_wasserstein_px=distance,
        signed_change_wasserstein_px=change,
        recorded_pair_position_count=10,
        synthetic_pair_position_count=10,
        recorded_pair_transition_count=9,
        synthetic_pair_transition_count=9,
        synthetic_measurement_summary={},
    )


def split_scores(*, second_pair_win: bool = True) -> tuple[SocialReplicateScore, ...]:
    records = []
    for pair_id in ("a", "b"):
        for replicate in range(SOCIAL_REPLICATES):
            records.append(score(pair_id, SOCIAL_MODEL_IDS[0], replicate, 10, 5))
            candidate_distance = 9 if pair_id == "a" or second_pair_win else 11
            records.append(
                score(pair_id, SOCIAL_MODEL_IDS[1], replicate, candidate_distance, 5.2)
            )
    return tuple(records)


class SocialEvaluationTests(unittest.TestCase):
    def test_preference_requires_distance_change_and_pair_consistency(self) -> None:
        result = aggregate_social_scores(split_scores(), eligible_pair_ids=("a", "b"))

        self.assertTrue(result["candidate_preferred"])
        self.assertEqual(result["candidate_distance_win_count"], 2)
        self.assertEqual(result["required_distance_win_count"], 2)
        self.assertEqual(
            result["equal_pair_mean_distance_wasserstein_px"],
            {SOCIAL_MODEL_IDS[0]: 10.0, SOCIAL_MODEL_IDS[1]: 9.0},
        )

        failed = aggregate_social_scores(
            split_scores(second_pair_win=False), eligible_pair_ids=("a", "b")
        )
        self.assertFalse(failed["candidate_preferred"])
        self.assertFalse(
            failed["preference_criteria"][
                "distance_better_on_at_least_60_percent_of_pairs"
            ]
        )

    def test_missing_metric_makes_preference_undefined(self) -> None:
        records = list(split_scores())
        records[0] = replace(records[0], signed_change_wasserstein_px=None)

        result = aggregate_social_scores(tuple(records), eligible_pair_ids=("a", "b"))

        self.assertIsNone(result["candidate_preferred"])
        self.assertIsNone(
            result["equal_pair_mean_distance_wasserstein_px"][SOCIAL_MODEL_IDS[0]]
        )

    def test_duplicate_replicate_is_rejected(self) -> None:
        records = list(split_scores())
        records[1] = replace(records[1], replicate=1)

        with self.assertRaisesRegex(DataError, "eight distinct replicates"):
            aggregate_social_scores(tuple(records), eligible_pair_ids=("a", "b"))


if __name__ == "__main__":
    unittest.main()
