"""Tests for the minimal headless command-line workflow."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
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

    def test_compare_reports_compact_temporal_run_summary(self) -> None:
        result = SimpleNamespace(
            experiment_id="fixture-temporal-lab",
            output_directory=Path("run"),
            configuration_sha256="abc123",
            model_count=4,
            replicate_records=128,
            generated_series=512,
            artifact_count=11,
        )
        output = io.StringIO()
        with (
            patch("umwelt.cli.load_temporal_lab_config", return_value=object()),
            patch("umwelt.cli.run_temporal_lab", return_value=result),
        ):
            exit_code = main(["compare", "--config", "temporal.json"], stdout=output)

        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["model_count"], 4)
        self.assertEqual(payload["replicate_records"], 128)

    def test_individual_reports_compact_population_run_summary(self) -> None:
        result = SimpleNamespace(
            experiment_id="fixture-individual-variation",
            output_directory=Path("individual-run"),
            configuration_sha256="def456",
            model_count=2,
            replicate_records=64,
            generated_series=256,
            artifact_count=10,
        )
        output = io.StringIO()
        with (
            patch("umwelt.cli.load_temporal_lab_config", return_value=object()),
            patch("umwelt.cli.run_individual_variation_lab", return_value=result),
        ):
            exit_code = main(
                [
                    "individual",
                    "--config",
                    "temporal.json",
                    "--output",
                    "individual-run",
                ],
                stdout=output,
            )

        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["model_count"], 2)
        self.assertEqual(payload["replicate_records"], 64)
        self.assertEqual(payload["generated_series"], 256)

    def test_spatial_reports_compact_run_summary(self) -> None:
        result = SimpleNamespace(
            experiment_id="fixture-spatial",
            output_directory=Path("spatial-run"),
            configuration_sha256="789abc",
            replicate_records=4,
            generated_series=4,
            artifact_count=8,
        )
        output = io.StringIO()
        with (
            patch("umwelt.cli.load_spatial_lab_config", return_value=object()),
            patch("umwelt.cli.run_spatial_lab", return_value=result),
        ):
            exit_code = main(
                ["spatial", "--config", "spatial.json"],
                stdout=output,
            )

        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["experiment_id"], "fixture-spatial")
        self.assertEqual(payload["replicate_records"], 4)
        self.assertEqual(payload["generated_series"], 4)
        self.assertEqual(payload["artifact_count"], 8)


if __name__ == "__main__":
    unittest.main()
