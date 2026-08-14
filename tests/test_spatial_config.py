"""Tests for the strict v0.2 spatial laboratory configuration."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from umwelt.errors import ConfigurationError
from umwelt.spatial_config import load_spatial_lab_config


def _payload() -> dict[str, object]:
    return {
        "schema_version": "umwelt.spatial-lab.v1",
        "experiment_id": "fixture-spatial",
        "dataset": {"directory": "data"},
        "simulation": {"seed": 1729, "replicates": 2},
        "output": {"directory": "run"},
    }


class SpatialConfigTests(unittest.TestCase):
    def _write(self, root: Path, payload: object) -> Path:
        path = root / "spatial.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_load_resolves_paths_and_round_trips_canonical_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = load_spatial_lab_config(self._write(root, _payload()))

        self.assertEqual(config.experiment_id, "fixture-spatial")
        self.assertEqual(config.dataset_directory, (root / "data").resolve())
        self.assertEqual(config.output_directory, (root / "run").resolve())
        self.assertEqual(config.seed, 1729)
        self.assertEqual(config.replicates, 2)

    def test_rejects_unknown_keys_and_wrong_scalar_types(self) -> None:
        cases = []
        unknown = _payload()
        unknown["result_tuned_metric"] = True
        cases.append(unknown)
        boolean_seed = _payload()
        boolean_seed["simulation"] = {"seed": True, "replicates": 2}
        cases.append(boolean_seed)
        numeric_directory = _payload()
        numeric_directory["dataset"] = {"directory": 42}
        cases.append(numeric_directory)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, payload in enumerate(cases):
                with self.subTest(index=index), self.assertRaises(ConfigurationError):
                    load_spatial_lab_config(self._write(root, payload))

    def test_rejects_invalid_seed_replicates_and_identifier(self) -> None:
        cases = []
        bad_seed = _payload()
        bad_seed["simulation"] = {"seed": -1, "replicates": 2}
        cases.append(bad_seed)
        bad_replicates = _payload()
        bad_replicates["simulation"] = {"seed": 1, "replicates": 101}
        cases.append(bad_replicates)
        bad_identifier = _payload()
        bad_identifier["experiment_id"] = "Has Spaces"
        cases.append(bad_identifier)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, payload in enumerate(cases):
                with self.subTest(index=index), self.assertRaises(ConfigurationError):
                    load_spatial_lab_config(self._write(root, payload))


if __name__ == "__main__":
    unittest.main()
