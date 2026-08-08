"""Tests for compact temporal replicate and distribution evaluation."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta

from umwelt.errors import DataError
from umwelt.evaluation import bout_durations
from umwelt.observations import (
    BehavioralState,
    ObservationDataset,
    ObservationSource,
    TimeSeries,
)
from umwelt.temporal_evaluation import (
    build_recorded_reference,
    empirical_distribution_distances,
    evaluate_synthetic_replicate,
    predictive_summary,
    quantile,
)


def _series(subject_id: str, source: ObservationSource, offset: int = 0) -> TimeSeries:
    start = datetime.fromisoformat("2020-01-01 00:00:00")
    timestamps = tuple(start + timedelta(seconds=10 * index) for index in range(12))
    base = (
        BehavioralState.WAKE,
        BehavioralState.WAKE,
        BehavioralState.SLEEP,
        BehavioralState.SLEEP,
        BehavioralState.WAKE,
        None,
        BehavioralState.SLEEP,
        BehavioralState.SLEEP,
        BehavioralState.SLEEP,
        BehavioralState.WAKE,
        BehavioralState.WAKE,
        BehavioralState.SLEEP,
    )
    states = base[offset:] + base[:offset] if offset else base
    activities = tuple(
        None
        if state is None
        else 0.0
        if state is BehavioralState.SLEEP
        else float(index + 1)
        for index, state in enumerate(states)
    )
    return TimeSeries(subject_id, source, timestamps, states, activities)


class DistributionMetricTests(unittest.TestCase):
    def test_hand_computable_empirical_distances(self) -> None:
        self.assertEqual(
            empirical_distribution_distances([0.0], [2.0]),
            {"wasserstein": 2.0, "kolmogorov_smirnov": 1.0},
        )
        shifted = empirical_distribution_distances([0.0, 2.0], [1.0, 3.0])
        self.assertEqual(shifted["wasserstein"], 1.0)
        self.assertEqual(shifted["kolmogorov_smirnov"], 0.5)
        self.assertEqual(
            empirical_distribution_distances([1.0, 2.0], [1.0, 2.0]),
            {"wasserstein": 0.0, "kolmogorov_smirnov": 0.0},
        )

    def test_empty_distribution_is_explicitly_unavailable(self) -> None:
        self.assertEqual(
            empirical_distribution_distances([], [1.0]),
            {"wasserstein": None, "kolmogorov_smirnov": None},
        )

    def test_quantile_and_predictive_interval_are_deterministic(self) -> None:
        self.assertEqual(quantile([0.0, 10.0], 0.25), 2.5)
        summary = predictive_summary([1.0, 2.0, 3.0], 2.5, (0.05, 0.95))
        self.assertEqual(summary["median"], 2.0)
        self.assertTrue(summary["reference_within_interval"])


class TemporalEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.recorded = ObservationDataset(
            "recorded",
            (
                _series("1", ObservationSource.RECORDED),
                _series("2", ObservationSource.RECORDED, offset=2),
            ),
            {"source_category": "recorded"},
        )
        self.reference = build_recorded_reference(
            self.recorded, phase_bins=2, autocorrelation_lags=(1, 2)
        )

    def test_bout_extraction_resets_at_missing_observations(self) -> None:
        bouts = bout_durations(self.recorded.subject("1"), 10)
        self.assertEqual(bouts[BehavioralState.WAKE], [20.0, 10.0, 20.0])
        self.assertEqual(bouts[BehavioralState.SLEEP], [20.0, 30.0, 10.0])

    def test_exact_synthetic_copy_has_zero_distribution_discrepancy(self) -> None:
        synthetic = ObservationDataset(
            "synthetic",
            tuple(
                TimeSeries(
                    series.subject_id,
                    ObservationSource.SYNTHETIC,
                    series.timestamps,
                    series.states,
                    series.activities,
                )
                for series in self.recorded.series
            ),
            {"source_category": "synthetic"},
        )

        result = evaluate_synthetic_replicate(synthetic, self.reference)

        self.assertEqual(
            result["aggregate"]["metrics"]["scalars"]["sleep_fraction"],
            self.reference.aggregate.snapshot["scalars"]["sleep_fraction"],
        )
        self.assertTrue(
            all(value == 0.0 for value in result["aggregate"]["discrepancies"].values())
        )
        self.assertEqual(set(result["subjects"]), {"1", "2"})

    def test_subject_identity_must_match_recorded_reference(self) -> None:
        wrong = ObservationDataset(
            "synthetic",
            (_series("other", ObservationSource.SYNTHETIC),),
            {"source_category": "synthetic"},
        )
        with self.assertRaisesRegex(DataError, "exactly match"):
            evaluate_synthetic_replicate(wrong, self.reference)


if __name__ == "__main__":
    unittest.main()
