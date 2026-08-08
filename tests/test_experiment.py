"""End-to-end tests for reproducible v0.1 experiment artifacts."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from umwelt.artifacts import sha256_file
from umwelt.config import load_experiment_config
from umwelt.errors import ConfigurationError
from umwelt.experiment import run_experiment


class ExperimentTests(unittest.TestCase):
    def _prepare(self, root: Path):
        data = root / "data"
        data.mkdir()
        start = datetime.fromisoformat("2020-01-01 00:00:00")
        timestamps = [start + timedelta(seconds=10 * index) for index in range(40)]
        for filename, sleep_file in (
            ("24mice_activity_LD1week.csv", False),
            ("24mice_sleep_LD1week.csv", True),
        ):
            with (data / filename).open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["Time", "1", "2"])
                for index, timestamp in enumerate(timestamps):
                    sleep = 1.0 if index % 8 >= 4 else 0.0
                    if sleep_file:
                        values = [sleep, sleep]
                    else:
                        values = [0.0 if sleep else index % 5, 0.0 if sleep else 2.0]
                    writer.writerow([timestamp.isoformat(sep=" "), *values])

        configuration = {
            "schema_version": "umwelt.experiment.v1",
            "experiment_id": "fixture-experiment",
            "dataset": {
                "id": "compass-zenodo-160344-v1",
                "directory": "data",
                "training_subjects": ["1"],
                "evaluation_subjects": ["2"],
            },
            "model": {
                "type": "phase-conditioned-markov-v1",
                "phase_bins": 1,
                "transition_prior": 0.5,
                "emission_prior": 0.5,
            },
            "simulation": {"seed": 1729, "replicates": 2},
            "evaluation": {"phase_bins": 1, "autocorrelation_lags": [1, 2]},
            "output": {"directory": "run"},
        }
        config_path = root / "experiment.json"
        config_path.write_text(json.dumps(configuration), encoding="utf-8")
        return load_experiment_config(config_path)

    def test_run_writes_complete_source_labeled_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._prepare(root)

            result = run_experiment(config, verify_integrity=False)

            expected = {
                "artifact-manifest.json",
                "comparison.json",
                "model.json",
                "provenance.json",
                "recorded-metrics.json",
                "report.md",
                "resolved-config.json",
                "seeds.json",
                "synthetic-metrics.json",
                "synthetic-observations.csv",
            }
            self.assertEqual(
                {path.name for path in result.output_directory.iterdir()}, expected
            )
            self.assertEqual(result.synthetic_series_count, 2)

            provenance = json.loads(
                (result.output_directory / "provenance.json").read_text("utf-8")
            )
            self.assertEqual(
                provenance["recorded_input"]["source_category"], "recorded"
            )
            self.assertEqual(
                provenance["synthetic_output"]["source_category"], "synthetic"
            )
            self.assertEqual(
                provenance["configuration_source"], str(config.source_path)
            )

            with (result.output_directory / "synthetic-observations.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(rows)
            self.assertEqual({row["source"] for row in rows}, {"synthetic"})

            manifest = json.loads(
                (result.output_directory / "artifact-manifest.json").read_text("utf-8")
            )
            model_entry = next(
                item for item in manifest["artifacts"] if item["name"] == "model.json"
            )
            self.assertEqual(
                model_entry["sha256"],
                sha256_file(result.output_directory / "model.json"),
            )

    def test_run_refuses_to_overwrite_existing_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._prepare(root)
            config.output_directory.mkdir()

            with self.assertRaisesRegex(ConfigurationError, "will not be overwritten"):
                run_experiment(config, verify_integrity=False)


if __name__ == "__main__":
    unittest.main()
