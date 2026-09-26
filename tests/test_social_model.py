"""Tests for the first paired social-model comparison."""

from __future__ import annotations

import unittest

from umwelt.calms21 import CALMS21_COORDINATE_FRAME
from umwelt.dyad import DyadFrame, measure_dyad_frames
from umwelt.errors import DataError
from umwelt.observations import ObservationSource
from umwelt.social_model import (
    SOCIAL_MODEL_IDS,
    SocialMovementFit,
    fit_social_movement_model,
    generate_social_dyad,
    social_role_seed,
)
from umwelt.spatial import Keypoint2D, Point2D, Pose2D, SpatialContext, SpatialFrame


def sample(index: int, resident_x: float | None, intruder_x: float) -> DyadFrame:
    def member(role: str, x: float | None) -> SpatialFrame:
        context = SpatialContext(
            f"pair: {role}",
            "pair",
            ObservationSource.RECORDED,
            CALMS21_COORDINATE_FRAME,
        )
        point = Point2D(x, 10.0) if x is not None else None
        return SpatialFrame(context, index, Pose2D((Keypoint2D("neck", point),)))

    return DyadFrame(
        "pair", member("resident", resident_x), member("intruder", intruder_x)
    )


def deterministic_fit(beta: float) -> SocialMovementFit:
    return SocialMovementFit(
        resident_alpha=0,
        resident_sigma_px=0,
        intruder_alpha=0,
        intruder_sigma_px=0,
        resident_directional_drift_px=beta,
        fitting_sequence_count=1,
        resident_eligible_triplets=1,
        intruder_eligible_triplets=1,
        resident_excluded_large_step_triplets=0,
        intruder_excluded_large_step_triplets=0,
        resident_excluded_missing_or_gap_triplets=0,
        intruder_excluded_missing_or_gap_triplets=0,
        directional_drift_triplets=1,
    )


class SocialModelTests(unittest.TestCase):
    def test_fitting_limits_large_steps_and_finds_directional_drift(self) -> None:
        sequence = (
            sample(0, 10, 100),
            sample(1, 11, 101),
            sample(2, 12, 102),
            sample(3, 14, 103),
            sample(4, 200, 104),
        )

        fit = fit_social_movement_model((sequence,))

        self.assertEqual(fit.resident_eligible_triplets, 2)
        self.assertEqual(fit.resident_excluded_large_step_triplets, 1)
        self.assertEqual(fit.intruder_eligible_triplets, 3)
        self.assertGreater(fit.resident_directional_drift_px, 0)
        self.assertEqual(fit.directional_drift_triplets, 2)

    def test_paired_innovations_isolate_the_social_term(self) -> None:
        template = tuple(sample(index, 10, 20) for index in range(3))
        fit = deterministic_fit(beta=2)

        independent = generate_social_dyad(
            fit, template, model_id=SOCIAL_MODEL_IDS[0], master_seed=9, replicate=0
        )
        directed = generate_social_dyad(
            fit, template, model_id=SOCIAL_MODEL_IDS[1], master_seed=9, replicate=0
        )

        self.assertEqual(
            measure_dyad_frames(independent).pair_distances_px, (10, 10, 10)
        )
        self.assertEqual(measure_dyad_frames(directed).pair_distances_px, (10, 8, 6))
        self.assertEqual(
            directed[2].resident.context.source, ObservationSource.SYNTHETIC
        )
        self.assertEqual(
            directed,
            generate_social_dyad(
                fit, template, model_id=SOCIAL_MODEL_IDS[1], master_seed=9, replicate=0
            ),
        )
        self.assertNotEqual(
            social_role_seed(9, "pair", 0, "resident"),
            social_role_seed(9, "pair", 0, "intruder"),
        )

    def test_generation_preserves_missing_mask(self) -> None:
        template = (sample(0, 10, 20), sample(1, None, 20), sample(2, 10, 20))

        generated = generate_social_dyad(
            deterministic_fit(1),
            template,
            model_id=SOCIAL_MODEL_IDS[1],
            master_seed=1,
            replicate=0,
        )

        self.assertIsNone(generated[1].resident.pose.keypoint("neck").point)
        self.assertIsNotNone(generated[2].resident.pose.keypoint("neck").point)
        self.assertEqual(measure_dyad_frames(generated).valid_pair_transition_count, 0)

    def test_generation_rejects_unavailable_initial_pair(self) -> None:
        with self.assertRaisesRegex(DataError, "valid first pair position"):
            generate_social_dyad(
                deterministic_fit(1),
                (sample(0, None, 20),),
                model_id=SOCIAL_MODEL_IDS[1],
                master_seed=1,
                replicate=0,
            )


if __name__ == "__main__":
    unittest.main()
