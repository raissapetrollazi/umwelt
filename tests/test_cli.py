"""Tests for the minimal headless command-line workflow."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from umwelt.cli import main
from umwelt.observations import (
    BehavioralState,
    ObservationDataset,
    ObservationSource,
    TimeSeries,
)


def _config(root: Path) -> Path:
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
        "simulation": {"seed": 1729, "replicates": 1},
        "evaluation": {"phase_bins": 1, "autocorrelation_lags": [1]},
        "output": {"directory": "run"},
    }
    path = root / "experiment.json"
    path.write_text(json.dumps(configuration), encoding="utf-8")
    return path


def _recorded_fixture() -> ObservationDataset:
    start = datetime.fromisoformat("2020-01-01 00:00:00")
    series = TimeSeries(
        subject_id="2",
        source=ObservationSource.RECORDED,
        timestamps=tuple(start + timedelta(seconds=10 * index) for index in range(3)),
        states=(BehavioralState.WAKE, None, BehavioralState.SLEEP),
        activities=(4.0, None, 0.0),
    )
    return ObservationDataset("fixture", (series,), {"source_category": "recorded"})


class CliTests(unittest.TestCase):
    def test_verify_reports_failure_for_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = io.StringIO()
            exit_code = main(
                ["data", "verify", "--directory", temporary], stdout=output
            )

        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 1)
        self.assertFalse(payload["valid"])

    def test_replay_emits_only_recorded_source_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = _config(root)
            output = io.StringIO()
            with patch(
                "umwelt.cli.load_compass_week", return_value=_recorded_fixture()
            ):
                exit_code = main(
                    [
                        "replay",
                        "--config",
                        str(config_path),
                        "--subject",
                        "2",
                        "--limit",
                        "3",
                    ],
                    stdout=output,
                )

        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(exit_code, 0)
        self.assertEqual(len(rows), 3)
        self.assertEqual({row["source"] for row in rows}, {"recorded"})
        self.assertFalse(rows[1]["available"])

    def test_replay_rejects_unconfigured_subject(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config_path = _config(Path(temporary))
            errors = io.StringIO()
            exit_code = main(
                [
                    "replay",
                    "--config",
                    str(config_path),
                    "--subject",
                    "3",
                ],
                stdout=io.StringIO(),
                stderr=errors,
            )

        self.assertEqual(exit_code, 2)
        self.assertIn("not selected", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
