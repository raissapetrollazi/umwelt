"""Tests for recorded/synthetic evaluation."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta

from umwelt.evaluation import compare_metrics, describe_dataset
from umwelt.errors import DataError
from umwelt.observations import (
    BehavioralState,
    ObservationDataset,
    ObservationSource,
    TimeSeries,
)


def _dataset(source: ObservationSource, subject_id: str) -> ObservationDataset:
    start = datetime(2020, 1, 1)
    timestamps = tuple(start + timedelta(seconds=10 * index) for index in range(8))
    series = TimeSeries(
        subject_id=subject_id,
        source=source,
        timestamps=timestamps,
        states=(
            BehavioralState.WAKE,
            BehavioralState.WAKE,
            BehavioralState.SLEEP,
            BehavioralState.SLEEP,
            None,
            BehavioralState.WAKE,
            BehavioralState.SLEEP,
            BehavioralState.WAKE,
        ),
        activities=(3.0, 0.0, 0.0, 0.0, None, 4.0, 0.0, 2.0),
    )
    return ObservationDataset(
        dataset_id=f"{source.value}-fixture",
        series=(series,),
        provenance={"source_category": source.value},
    )


class EvaluationTests(unittest.TestCase):
    def test_description_covers_state_activity_bouts_and_dependence(self) -> None:
        metrics = describe_dataset(
            _dataset(ObservationSource.RECORDED, "1"),
            phase_bins=1,
            autocorrelation_lags=(1, 2),
        )

        self.assertEqual(metrics["source_category"], "recorded")
        self.assertEqual(metrics["observed_epochs"], 7)
        self.assertEqual(metrics["missing_epochs"], 1)
        self.assertAlmostEqual(metrics["state_occupancy"]["sleep_fraction"], 3 / 7)
        self.assertEqual(metrics["transitions"]["state_changes"], 3)
        self.assertEqual(metrics["bout_durations_seconds"]["sleep"]["count"], 2)
        self.assertEqual(metrics["bout_durations_seconds"]["sleep"]["maximum"], 20.0)
        self.assertEqual(len(metrics["sleep_state_autocorrelation"]), 2)

    def test_comparison_keeps_source_roles_explicit(self) -> None:
        recorded = describe_dataset(
            _dataset(ObservationSource.RECORDED, "1"), phase_bins=1
        )
        synthetic = describe_dataset(
            _dataset(ObservationSource.SYNTHETIC, "synthetic-1"), phase_bins=1
        )

        comparison = compare_metrics(recorded, synthetic)

        self.assertEqual(comparison["recorded_dataset_id"], "recorded-fixture")
        self.assertEqual(comparison["synthetic_dataset_id"], "synthetic-fixture")
        self.assertEqual(comparison["phase_profile"]["sleep_fraction_rmse"], 0.0)
        self.assertTrue(
            all(item["absolute_difference"] == 0 for item in comparison["scalar_metrics"])
        )

    def test_comparison_rejects_reversed_sources(self) -> None:
        recorded = describe_dataset(
            _dataset(ObservationSource.RECORDED, "1"), phase_bins=1
        )
        synthetic = describe_dataset(
            _dataset(ObservationSource.SYNTHETIC, "synthetic-1"), phase_bins=1
        )

        with self.assertRaisesRegex(DataError, "first comparison input"):
            compare_metrics(synthetic, recorded)


if __name__ == "__main__":
    unittest.main()
