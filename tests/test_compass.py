"""Tests for the pinned COMPASS dataset adapter."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from umwelt.datasets.compass import (
    COMPASS_DOI,
    COMPASS_FILES,
    load_compass_week,
    verify_compass,
)
from umwelt.errors import DataError
from umwelt.observations import BehavioralState, ObservationSource


class CompassAdapterTests(unittest.TestCase):
    def _write_fixture(
        self,
        root: Path,
        *,
        sleep_rows: list[tuple[str, str]] | None = None,
        activity_rows: list[tuple[str, str]] | None = None,
    ) -> None:
        timestamps = [
            "2015-02-26 00:00:00",
            "2015-02-26 00:00:10",
            "2015-02-26 00:00:20",
            "2015-02-26 00:00:30",
        ]
        sleep_values = sleep_rows or list(zip(timestamps, ["0.0", "", "1.0", "0.0"]))
        activity_values = activity_rows or list(
            zip(timestamps, ["12.0", "", "0.0", "3.0"])
        )
        for name, rows in (
            ("24mice_sleep_LD1week.csv", sleep_values),
            ("24mice_activity_LD1week.csv", activity_values),
        ):
            with (root / name).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["Time", "1"])
                writer.writerows(rows)

    def test_manifest_pins_official_record(self) -> None:
        self.assertEqual(COMPASS_DOI, "10.5281/zenodo.160344")
        self.assertEqual(len(COMPASS_FILES), 6)
        self.assertEqual(sum(file.required for file in COMPASS_FILES), 2)
        self.assertTrue(all(len(file.md5) == 32 for file in COMPASS_FILES))

    def test_loader_preserves_recorded_source_and_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_fixture(root)

            dataset = load_compass_week(root, subject_ids=["1"], verify_integrity=False)

        series = dataset.subject("1")
        self.assertEqual(series.source, ObservationSource.RECORDED)
        self.assertEqual(series.valid_epoch_count, 3)
        self.assertEqual(
            series.states,
            (
                BehavioralState.WAKE,
                None,
                BehavioralState.SLEEP,
                BehavioralState.WAKE,
            ),
        )
        self.assertEqual(
            [item.source for item in series.observations()],
            [
                ObservationSource.RECORDED,
                ObservationSource.RECORDED,
                ObservationSource.RECORDED,
            ],
        )
        self.assertEqual(dataset.provenance["doi"], COMPASS_DOI)

    def test_loader_rejects_misaligned_source_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_fixture(
                root,
                sleep_rows=[("2015-02-26 00:00:00", "0.0")],
                activity_rows=[("2015-02-26 00:00:10", "1.0")],
            )
            with self.assertRaisesRegex(DataError, "timestamps diverge"):
                load_compass_week(root, subject_ids=["1"], verify_integrity=False)

    def test_loader_rejects_activity_during_labeled_sleep(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = [("2015-02-26 00:00:00", "1.0")]
            self._write_fixture(root, sleep_rows=row, activity_rows=row)
            with self.assertRaisesRegex(
                DataError, "sleep labels and activity disagree"
            ):
                load_compass_week(root, subject_ids=["1"], verify_integrity=False)

    def test_loader_rejects_different_source_row_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_fixture(
                root,
                sleep_rows=[("2015-02-26 00:00:00", "0.0")],
                activity_rows=[
                    ("2015-02-26 00:00:00", "1.0"),
                    ("2015-02-26 00:00:10", "1.0"),
                ],
            )
            with self.assertRaisesRegex(DataError, "different row counts"):
                load_compass_week(root, subject_ids=["1"], verify_integrity=False)

    def test_verification_reports_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results = verify_compass(temporary)

        self.assertEqual(len(results), 2)
        self.assertTrue(all(not result.valid for result in results))
        self.assertTrue(all(result.actual_md5 is None for result in results))


if __name__ == "__main__":
    unittest.main()
