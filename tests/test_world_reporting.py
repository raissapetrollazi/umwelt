"""Tests for paired intervention summaries and declared predictive quantiles."""

from __future__ import annotations

import unittest
from dataclasses import replace

from umwelt.errors import DataError
from umwelt.spatial_protocol import ROCHE_CONTROL_FITTING_SUBJECTS
from umwelt.world_execution import INTERVENTION_VALUE_IDS, InterventionOutcome
from umwelt.world_protocol import WorldExperimentProtocol
from umwelt.world_reporting import (
    build_intervention_comparison,
    quantile_summary,
)


def _outcomes(
    protocol: WorldExperimentProtocol,
) -> tuple[InterventionOutcome, ...]:
    values = {
        "recorded": (100.0, 0.2, 5),
        "compact": (90.0, 0.3, 7),
        "expanded": (110.0, 0.1, 3),
    }
    rows = []
    for subject_id in ROCHE_CONTROL_FITTING_SUBJECTS:
        for replicate in range(protocol.replicates):
            for condition_id, (path, occupancy, reflections) in values.items():
                metrics = {
                    "path_length_px": path,
                    "valid_transition_count": 9,
                    "boundary_band_occupancy_fraction": occupancy,
                    "boundary_band_valid_transition_count": 3,
                    "interior_valid_transition_count": 6,
                    "boundary_band_positive_displacement_count": 3,
                    "boundary_band_positive_displacement_mean_px": 2.0,
                    "boundary_band_positive_displacement_median_px": 2.0,
                    "interior_positive_displacement_count": 6,
                    "interior_positive_displacement_mean_px": 2.0,
                    "interior_positive_displacement_median_px": 2.0,
                    "reflection_count": reflections,
                }
                rows.append(
                    InterventionOutcome(
                        subject_id,
                        replicate,
                        protocol.seed(subject_id, replicate),
                        condition_id,
                        tuple((name, metrics[name]) for name in INTERVENTION_VALUE_IDS),
                    )
                )
    return tuple(rows)


class WorldReportingTests(unittest.TestCase):
    def test_quantiles_use_frozen_linear_interpolation_and_keep_missingness(
        self,
    ) -> None:
        self.assertEqual(
            quantile_summary((0.0, 10.0, None)),
            {
                "present_count": 2,
                "missing_count": 1,
                "p05": 0.5,
                "p50": 5.0,
                "p95": 9.5,
            },
        )

    def test_intervention_changes_are_paired_before_animal_aggregation(self) -> None:
        protocol = WorldExperimentProtocol(1730, 2)
        compared = build_intervention_comparison(protocol, _outcomes(protocol))
        aggregate = compared["aggregate_animal_medians"]
        self.assertEqual(aggregate["compact"]["path_length_px"]["p50"], -10.0)
        self.assertEqual(aggregate["expanded"]["path_length_px"]["p50"], 10.0)
        self.assertEqual(aggregate["compact"]["reflection_count"]["p50"], 2.0)
        self.assertEqual(aggregate["expanded"]["reflection_count"]["p50"], -2.0)
        self.assertEqual(aggregate["compact"]["path_length_px"]["present_count"], 6)
        self.assertFalse(compared["recorded_animal_intervention_counterpart"])

    def test_missing_seed_or_exposure_mismatch_rejects_pairing(self) -> None:
        protocol = WorldExperimentProtocol(1730, 1)
        rows = _outcomes(protocol)
        with self.assertRaisesRegex(DataError, "every declared paired run"):
            build_intervention_comparison(protocol, rows[:-1])
        with self.assertRaisesRegex(DataError, "paired protocol seeds"):
            build_intervention_comparison(
                protocol, (replace(rows[0], seed=rows[0].seed ^ 1),) + rows[1:]
            )
        changed = dict(rows[0].values)
        changed["valid_transition_count"] = 8
        mismatched = replace(
            rows[0],
            values=tuple((name, changed[name]) for name in INTERVENTION_VALUE_IDS),
        )
        with self.assertRaisesRegex(DataError, "equal transition exposure"):
            build_intervention_comparison(protocol, (mismatched,) + rows[1:])


if __name__ == "__main__":
    unittest.main()
