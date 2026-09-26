"""Tests for the narrow v0.3 world-laboratory configuration."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from umwelt.errors import ConfigurationError
from umwelt.world_config import load_world_lab_config


def _payload() -> dict[str, object]:
    return {
        "schema_version": "umwelt.world-lab.v1",
        "experiment_id": "fixture-world",
        "dataset": {"directory": "data"},
        "simulation": {"seed": 1730, "replicates": 2},
        "output": {"directory": "run"},
    }


class WorldConfigTests(unittest.TestCase):
    def _write(self, root: Path, payload: object) -> Path:
        path = root / "world.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_load_resolves_paths_and_preserves_user_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = load_world_lab_config(self._write(root, _payload()))
            self.assertEqual(config.dataset_directory, (root / "data").resolve())
            self.assertEqual(config.output_directory, (root / "run").resolve())
            self.assertEqual(config.seed, 1730)
            self.assertEqual(config.replicates, 2)
            self.assertEqual(config.to_dict()["schema_version"], "umwelt.world-lab.v1")

    def test_rejects_scientific_knobs_and_invalid_run_inputs(self) -> None:
        cases = []
        extra = _payload()
        extra["zone_fraction"] = 0.25
        cases.append(extra)
        invalid_seed = _payload()
        invalid_seed["simulation"] = {"seed": True, "replicates": 2}
        cases.append(invalid_seed)
        invalid_replicates = _payload()
        invalid_replicates["simulation"] = {"seed": 1, "replicates": 0}
        cases.append(invalid_replicates)
        invalid_path = _payload()
        invalid_path["dataset"] = {"directory": 42}
        cases.append(invalid_path)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, payload in enumerate(cases):
                with self.subTest(index=index), self.assertRaises(ConfigurationError):
                    load_world_lab_config(self._write(root, payload))


if __name__ == "__main__":
    unittest.main()
