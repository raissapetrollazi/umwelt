"""Integration tests for temporal-lab configuration and artifacts."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from umwelt.artifacts import sha256_file
from umwelt.errors import ConfigurationError
from umwelt.temporal_config import load_temporal_lab_config
from umwelt.temporal_lab import derive_replicate_seed, run_temporal_lab


def _configuration(output: str = "run") -> dict[str, object]:
    return {
        "schema_version": "umwelt.temporal-lab.v1",
        "experiment_id": "fixture-temporal-lab",
        "dataset": {
            "id": "compass-zenodo-160344-v1",
            "directory": "data",
            "training_subjects": ["1"],
            "development_subjects": ["2"],
        },
        "model": {
            "phase_bins": 2,
            "duration_bin_edges_epochs": [1, 2, 4, 8, 16],
            "transition_prior": 0.5,
            "duration_prior_strength": 8.0,
            "emission_prior": 0.5,
        },
        "simulation": {"seed": 1729, "replicates": 2},
        "evaluation": {
            "phase_bins": 2,
            "autocorrelation_lags": [1, 2],
            "replicate_interval": [0.05, 0.95],
        },
        "output": {"directory": output},
    }


def _prepare(root: Path, output: str = "run"):
    data = root / "data"
    data.mkdir()
    start = datetime.fromisoformat("2020-01-01 00:00:00")
    timestamps = [start + timedelta(seconds=10 * index) for index in range(240)]
    for filename, sleep_file in (
        ("24mice_activity_LD1week.csv", False),
        ("24mice_sleep_LD1week.csv", True),
    ):
        with (data / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Time", "1", "2"])
            for index, timestamp in enumerate(timestamps):
                sleep_one = 1.0 if (index // 6) % 2 else 0.0
                sleep_two = 1.0 if ((index + 3) // 8) % 2 else 0.0
                if sleep_file:
                    values = [sleep_one, sleep_two]
                else:
                    values = [
                        0.0 if sleep_one else float(index % 7),
                        0.0 if sleep_two else float((index + 2) % 9),
                    ]
                writer.writerow([timestamp.isoformat(sep=" "), *values])
    path = root / "temporal.json"
    path.write_text(json.dumps(_configuration(output)), encoding="utf-8")
    return load_temporal_lab_config(path)


class TemporalConfigTests(unittest.TestCase):
    def test_paths_and_development_role_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _prepare(root)
            self.assertEqual(config.dataset.directory, (root / "data").resolve())
            self.assertEqual(config.dataset.development_subjects, ("2",))
            self.assertEqual(config.simulation.replicates, 2)

    def test_split_overlap_and_invalid_interval_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            configuration = _configuration()
            configuration["dataset"]["development_subjects"] = ["1"]
            path = root / "invalid.json"
            path.write_text(json.dumps(configuration), encoding="utf-8")
            with self.assertRaisesRegex(ConfigurationError, "must be disjoint"):
                load_temporal_lab_config(path)

            configuration = _configuration()
            configuration["evaluation"]["replicate_interval"] = [0.9, 0.1]
            path.write_text(json.dumps(configuration), encoding="utf-8")
            with self.assertRaisesRegex(ConfigurationError, "lower < upper"):
                load_temporal_lab_config(path)


class TemporalLabTests(unittest.TestCase):
    def test_seed_derivation_uses_every_identifier(self) -> None:
        seed = derive_replicate_seed(1729, "model-a", "2", 1)
        self.assertEqual(seed, derive_replicate_seed(1729, "model-a", "2", 1))
        self.assertNotEqual(seed, derive_replicate_seed(1729, "model-b", "2", 1))
        self.assertNotEqual(seed, derive_replicate_seed(1729, "model-a", "2", 2))

    def test_run_writes_compact_source_labeled_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = _prepare(Path(temporary))

            result = run_temporal_lab(config, verify_integrity=False)

            expected = {
                "artifact-manifest.json",
                "autocorrelation.svg",
                "model-comparison.json",
                "model-comparison.svg",
                "models.json",
                "provenance.json",
                "recorded-reference.json",
                "replicate-metrics.jsonl",
                "report.md",
                "resolved-config.json",
                "seeds.json",
            }
            self.assertEqual(
                {path.name for path in result.output_directory.iterdir()}, expected
            )
            self.assertEqual(result.model_count, 4)
            self.assertEqual(result.replicate_records, 8)
            self.assertEqual(result.generated_series, 8)

            records = [
                json.loads(line)
                for line in (result.output_directory / "replicate-metrics.jsonl")
                .read_text("utf-8")
                .splitlines()
            ]
            self.assertEqual(len(records), 8)
            self.assertEqual({record["replicate"] for record in records}, {1, 2})

            models = json.loads(
                (result.output_directory / "models.json").read_text("utf-8")
            )
            self.assertTrue(
                all(
                    model["training_subject_ids"] == ["1"]
                    for model in models["state_models"]
                )
            )
            provenance = json.loads(
                (result.output_directory / "provenance.json").read_text("utf-8")
            )
            self.assertEqual(
                provenance["protocol"]["held_out_development_subjects"], ["2"]
            )
            self.assertFalse(
                provenance["synthetic_output"]["full_trajectories_retained"]
            )

            manifest = json.loads(
                (result.output_directory / "artifact-manifest.json").read_text("utf-8")
            )
            for item in manifest["artifacts"]:
                self.assertEqual(
                    item["sha256"],
                    sha256_file(result.output_directory / item["name"]),
                )

    def test_scientific_artifacts_reproduce_across_output_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _prepare(root)

            first = run_temporal_lab(
                config, output_directory=root / "run-a", verify_integrity=False
            )
            second = run_temporal_lab(
                config, output_directory=root / "run-b", verify_integrity=False
            )

            for name in (
                "model-comparison.json",
                "models.json",
                "recorded-reference.json",
                "replicate-metrics.jsonl",
                "seeds.json",
            ):
                self.assertEqual(
                    (first.output_directory / name).read_bytes(),
                    (second.output_directory / name).read_bytes(),
                )


if __name__ == "__main__":
    unittest.main()
