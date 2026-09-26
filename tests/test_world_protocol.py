"""Tests for the frozen v0.3 development decision and intervention contract."""

from __future__ import annotations

import unittest
from dataclasses import replace

from umwelt.errors import DataError
from umwelt.spatial_evaluation import SpatialComparison
from umwelt.spatial_protocol import (
    ROCHE_CONTROL_DEVELOPMENT_SUBJECTS,
    ROCHE_CONTROL_FITTING_SUBJECTS,
)
from umwelt.world_evaluation import WorldComparison
from umwelt.world_protocol import (
    BASELINE_MODEL_ID,
    BOUNDARY_CONDITIONED_MODEL_ID,
    WORLD_CROSS_VALIDATION_FOLDS,
    WORLD_PREFERENCE_CRITERION_IDS,
    WorldExperimentProtocol,
    WorldReplicateComparison,
    derive_world_replicate_seed,
    evaluate_world_model_preference,
)


def _comparison(
    *, path: float, displacement: float, turning: float, occupancy: float
) -> WorldComparison:
    return WorldComparison(
        spatial=SpatialComparison(
            path_length_signed_difference_px=path * 100.0,
            path_length_absolute_difference_px=abs(path) * 100.0,
            path_length_signed_relative_difference=path,
            path_length_relative_difference=abs(path),
            adjacent_displacement_wasserstein_px=displacement,
            absolute_turning_wasserstein_radians=turning,
        ),
        boundary_band_occupancy_absolute_difference=occupancy,
        zone_positive_displacement_wasserstein_px=(
            ("boundary-band", 1.0),
            ("interior", 1.0),
        ),
    )


def _rows(
    protocol: WorldExperimentProtocol,
    *,
    candidate_paths: tuple[float, ...] = (0.2, 0.2, 0.2, 0.2, 0.32, 0.32),
    candidate_occupancy: float = 0.1,
    candidate_displacement: float = 2.0,
    candidate_turning: float = 0.5,
) -> tuple[WorldReplicateComparison, ...]:
    rows = []
    for subject_index, subject_id in enumerate(ROCHE_CONTROL_FITTING_SUBJECTS):
        for replicate in range(protocol.replicates):
            seed = protocol.seed(subject_id, replicate)
            rows.append(
                WorldReplicateComparison(
                    subject_id,
                    BASELINE_MODEL_ID,
                    replicate,
                    seed,
                    "recorded",
                    _comparison(
                        path=0.3,
                        displacement=2.0,
                        turning=0.5,
                        occupancy=0.2,
                    ),
                )
            )
            rows.append(
                WorldReplicateComparison(
                    subject_id,
                    BOUNDARY_CONDITIONED_MODEL_ID,
                    replicate,
                    seed,
                    "recorded",
                    _comparison(
                        path=candidate_paths[subject_index],
                        displacement=candidate_displacement,
                        turning=candidate_turning,
                        occupancy=candidate_occupancy,
                    ),
                )
            )
    return tuple(rows)


class WorldProtocolTests(unittest.TestCase):
    def test_fold_and_intervention_contract_excludes_legacy_reference(self) -> None:
        protocol = WorldExperimentProtocol(master_seed=1729, replicates=3)
        self.assertEqual(len(WORLD_CROSS_VALIDATION_FOLDS), 6)
        self.assertEqual(
            tuple(fold.evaluation_subject_id for fold in WORLD_CROSS_VALIDATION_FOLDS),
            ROCHE_CONTROL_FITTING_SUBJECTS,
        )
        for fold in WORLD_CROSS_VALIDATION_FOLDS:
            self.assertEqual(len(fold.fitting_subject_ids), 5)
            self.assertNotIn(fold.evaluation_subject_id, fold.fitting_subject_ids)
            self.assertTrue(
                set(fold.fitting_subject_ids).isdisjoint(
                    ROCHE_CONTROL_DEVELOPMENT_SUBJECTS
                )
            )

        serialized = protocol.to_dict()
        self.assertEqual(
            serialized["interventions"]["conditions"],
            [
                {
                    "condition_id": name,
                    "intervention": "isotropic-arena-scale",
                    "scale": scale,
                }
                for name, scale in (
                    ("compact", 0.75),
                    ("recorded", 1.0),
                    ("expanded", 1.25),
                )
            ],
        )
        self.assertTrue(
            serialized["cross_validation"]["legacy_diagnostic_excluded_from_preference"]
        )
        self.assertEqual(
            serialized["cross_validation"]["predictive_quantiles"],
            [0.05, 0.5, 0.95],
        )

    def test_seed_is_deterministic_and_shared_across_model_conditions(self) -> None:
        protocol = WorldExperimentProtocol(master_seed=1729, replicates=3)
        subject_id = ROCHE_CONTROL_FITTING_SUBJECTS[0]
        seed = protocol.seed(subject_id, 0)
        self.assertEqual(seed, derive_world_replicate_seed(1729, subject_id, 0))
        self.assertEqual(seed, protocol.seed(subject_id, 0))
        self.assertNotEqual(seed, protocol.seed(subject_id, 1))
        with self.assertRaisesRegex(DataError, "exceeds"):
            protocol.seed(subject_id, 3)

    def test_preference_uses_animal_medians_and_all_five_rules(self) -> None:
        protocol = WorldExperimentProtocol(master_seed=1729, replicates=3)
        rows = list(_rows(protocol))
        # One extreme replicate must not turn an animal median into a frame count.
        candidate = rows[1]
        rows[1] = replace(
            candidate,
            comparison=_comparison(
                path=0.9, displacement=2.0, turning=0.5, occupancy=0.1
            ),
        )
        decision = evaluate_world_model_preference(protocol, tuple(reversed(rows)))
        self.assertTrue(decision.eligible_for_independent_evaluation)
        self.assertEqual(decision.path_improved_animal_count, 4)
        self.assertEqual(
            tuple(name for name, _ in decision.criteria),
            WORLD_PREFERENCE_CRITERION_IDS,
        )
        self.assertTrue(all(passed for _, passed in decision.criteria))
        self.assertEqual(
            decision.subject_medians[0][2].absolute_relative_path_length, 0.2
        )
        self.assertEqual(
            decision.aggregate_candidate.absolute_relative_path_length, 0.2
        )
        retained = rows[1].to_dict()
        self.assertEqual(
            retained["comparison"]["spatial"]["path_length_signed_relative_difference"],
            0.9,
        )
        self.assertEqual(
            retained["comparison"]["zone_positive_displacement_wasserstein_px"],
            {"boundary-band": 1.0, "interior": 1.0},
        )

    def test_strict_path_and_occupancy_rules_can_reject_candidate(self) -> None:
        protocol = WorldExperimentProtocol(master_seed=1, replicates=1)
        no_path_improvement = evaluate_world_model_preference(
            protocol,
            _rows(protocol, candidate_paths=(0.3,) * 6),
        )
        self.assertFalse(no_path_improvement.eligible_for_independent_evaluation)
        self.assertFalse(
            dict(no_path_improvement.criteria)[WORLD_PREFERENCE_CRITERION_IDS[0]]
        )
        self.assertFalse(
            dict(no_path_improvement.criteria)[WORLD_PREFERENCE_CRITERION_IDS[1]]
        )

        no_occupancy_improvement = evaluate_world_model_preference(
            protocol,
            _rows(protocol, candidate_occupancy=0.2),
        )
        self.assertFalse(no_occupancy_improvement.eligible_for_independent_evaluation)
        self.assertFalse(
            dict(no_occupancy_improvement.criteria)[WORLD_PREFERENCE_CRITERION_IDS[4]]
        )

        for parameter, value, criterion in (
            ("candidate_displacement", 2.01, WORLD_PREFERENCE_CRITERION_IDS[2]),
            ("candidate_turning", 0.51, WORLD_PREFERENCE_CRITERION_IDS[3]),
        ):
            with self.subTest(parameter=parameter):
                rejected = evaluate_world_model_preference(
                    protocol, _rows(protocol, **{parameter: value})
                )
                self.assertFalse(rejected.eligible_for_independent_evaluation)
                self.assertFalse(dict(rejected.criteria)[criterion])

    def test_preference_rejects_missing_duplicate_or_unpaired_replicates(self) -> None:
        protocol = WorldExperimentProtocol(master_seed=1, replicates=1)
        rows = _rows(protocol)
        with self.assertRaisesRegex(DataError, "every declared"):
            evaluate_world_model_preference(protocol, rows[:-1])
        with self.assertRaisesRegex(DataError, "Duplicate"):
            evaluate_world_model_preference(protocol, rows + (rows[0],))
        with self.assertRaisesRegex(DataError, "paired seed"):
            evaluate_world_model_preference(
                protocol, (replace(rows[0], seed=rows[0].seed ^ 1),) + rows[1:]
            )

    def test_development_reference_and_intervention_cannot_enter_preference(
        self,
    ) -> None:
        protocol = WorldExperimentProtocol(master_seed=1, replicates=1)
        row = _rows(protocol)[0]
        with self.assertRaisesRegex(DataError, "cross-validation animal"):
            replace(row, subject_id=ROCHE_CONTROL_DEVELOPMENT_SUBJECTS[0])
        with self.assertRaisesRegex(DataError, "recorded-scale"):
            replace(row, condition_id="compact")
        with self.assertRaisesRegex(DataError, "finite signed relative"):
            replace(
                row,
                comparison=_comparison(
                    path=float("nan"),
                    displacement=2.0,
                    turning=0.5,
                    occupancy=0.2,
                ),
            )


if __name__ == "__main__":
    unittest.main()
