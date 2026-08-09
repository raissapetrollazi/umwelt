"""Integration tests for the v0.1 individual-variation laboratory."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from umwelt.artifacts import sha256_file
from umwelt.individual_lab import (
    derive_paired_trajectory_seed,
    run_individual_variation_lab,
)
from umwelt.temporal_config import load_temporal_lab_config


def _configuration() -> dict[str, object]:
    return {
        "schema_version": "umwelt.temporal-lab.v1",
        "experiment_id": "fixture-individual-variation",
        "dataset": {
            "id": "compass-zenodo-160344-v1",
            "directory": "data",
            "training_subjects": ["1", "2"],
            "development_subjects": ["3", "4"],
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
        "output": {"directory": "unused-temporal-output"},
    }


def _prepare(root: Path):
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
            writer.writerow(["Time", "1", "2", "3", "4"])
            for index, timestamp in enumerate(timestamps):
                sleep_values = [
                    1.0 if index % 12 >= 8 else 0.0,
                    1.0 if index % 12 >= 5 else 0.0,
                    1.0 if index % 14 >= 7 else 0.0,
                    1.0 if index % 10 >= 3 else 0.0,
                ]
                if sleep_file:
                    values = sleep_values
                else:
                    values = [
                        0.0 if sleep else float(1 + (index + subject) % 7)
                        for subject, sleep in enumerate(sleep_values)
                    ]
                writer.writerow([timestamp.isoformat(sep=" "), *values])
    path = root / "temporal.json"
    path.write_text(json.dumps(_configuration()), encoding="utf-8")
    return load_temporal_lab_config(path)


class IndividualLabTests(unittest.TestCase):
    def test_paired_seed_uses_subject_and_replicate(self) -> None:
        seed = derive_paired_trajectory_seed(1729, "3", 1)
        self.assertEqual(seed, derive_paired_trajectory_seed(1729, "3", 1))
        self.assertNotEqual(seed, derive_paired_trajectory_seed(1729, "4", 1))
        self.assertNotEqual(seed, derive_paired_trajectory_seed(1729, "3", 2))

    def test_run_keeps_profiles_training_only_and_writes_compact_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _prepare(root)
            result = run_individual_variation_lab(
                config,
                output_directory=root / "individual-run",
                verify_integrity=False,
            )

            expected = {
                "artifact-manifest.json",
                "individual-variation.svg",
                "model-comparison.json",
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
            self.assertEqual(result.model_count, 2)
            self.assertEqual(result.replicate_records, 4)
            self.assertEqual(result.generated_series, 8)

            models = json.loads(
                (result.output_directory / "models.json").read_text("utf-8")
            )
            population = models["population_state_model"]
            self.assertEqual(population["training_subject_ids"], ["1", "2"])
            self.assertEqual(
                {profile["source_subject_id"] for profile in population["profiles"]},
                {"1", "2"},
            )

            seeds = json.loads(
                (result.output_directory / "seeds.json").read_text("utf-8")
            )["series"]
            pooled = [item for item in seeds if item["profile_seed"] is None]
            population_seeds = [
                item for item in seeds if item["profile_seed"] is not None
            ]
            self.assertEqual(len(pooled), len(population_seeds))
            paired = {
                (item["source_subject_id"], item["replicate"]): item["trajectory_seed"]
                for item in pooled
            }
            for item in population_seeds:
                key = (item["source_subject_id"], item["replicate"])
                self.assertEqual(item["trajectory_seed"], paired[key])
                self.assertIn(item["profile_source_subject_id"], {"1", "2"})

            recorded = json.loads(
                (result.output_directory / "recorded-reference.json").read_text("utf-8")
            )
            self.assertIn("sleep_fraction_sd", recorded["between_subject"])

            comparison = json.loads(
                (result.output_directory / "model-comparison.json").read_text("utf-8")
            )
            self.assertEqual(len(comparison["models"]), 2)
            for model in comparison["models"]:
                self.assertIn("between_subject", model["summary"])

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
            first = run_individual_variation_lab(
                config,
                output_directory=root / "run-a",
                verify_integrity=False,
            )
            second = run_individual_variation_lab(
                config,
                output_directory=root / "run-b",
                verify_integrity=False,
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
