"""Tests for strict experiment configuration."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from umwelt.config import load_experiment_config
from umwelt.errors import ConfigurationError


def _configuration() -> dict[str, object]:
    return {
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
            "phase_bins": 24,
            "transition_prior": 0.5,
            "emission_prior": 0.5,
        },
        "simulation": {"seed": 1729, "replicates": 1},
        "evaluation": {"phase_bins": 24, "autocorrelation_lags": [1, 6]},
        "output": {"directory": "runs/fixture"},
    }


class ConfigTests(unittest.TestCase):
    def _load(self, root: Path, configuration: dict[str, object]):
        path = root / "experiment.json"
        path.write_text(json.dumps(configuration), encoding="utf-8")
        return load_experiment_config(path)

    def test_relative_paths_are_resolved_from_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._load(root, _configuration())

            self.assertEqual(config.dataset.directory, (root / "data").resolve())
            self.assertEqual(
                config.output_directory, (root / "runs" / "fixture").resolve()
            )
            self.assertEqual(config.simulation.seed, 1729)

    def test_unknown_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            configuration = _configuration()
            configuration["surprise"] = True
            with self.assertRaisesRegex(ConfigurationError, "unknown keys"):
                self._load(Path(temporary), configuration)

    def test_training_and_evaluation_must_be_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            configuration = _configuration()
            configuration["dataset"]["evaluation_subjects"] = ["1"]
            with self.assertRaisesRegex(ConfigurationError, "must be disjoint"):
                self._load(Path(temporary), configuration)

    def test_invalid_seed_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            configuration = _configuration()
            configuration["simulation"]["seed"] = -1
            with self.assertRaisesRegex(ConfigurationError, "64-bit unsigned"):
                self._load(Path(temporary), configuration)


if __name__ == "__main__":
    unittest.main()
