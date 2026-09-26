"""Artifact integrity and atomic publication for the v0.3 world laboratory."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from umwelt.arena import RectangularArena
from umwelt.datasets.roche_open_field import ROCHE_COORDINATE_FRAME
from umwelt.errors import DataError
from umwelt.observations import ObservationSource
from umwelt.spatial import Keypoint2D, Point2D, Pose2D, SpatialContext, SpatialFrame
from umwelt.spatial_protocol import (
    ROCHE_CONTROL_DEVELOPMENT_SUBJECTS,
    ROCHE_CONTROL_FITTING_SUBJECTS,
    ROCHE_RECORDED_TRAJECTORY_PROTOCOL,
)
from umwelt.world_config import WorldLabConfig
from umwelt.world_lab import run_world_lab
from umwelt.world_recordings import RecordedWorldSeries


def _series(subject_id: str) -> RecordedWorldSeries:
    context = SpatialContext(
        subject_id,
        f"source:{subject_id}",
        ObservationSource.RECORDED,
        ROCHE_COORDINATE_FRAME,
    )
    positions = (
        Point2D(5, 50),
        Point2D(7, 50),
        Point2D(9, 50),
        Point2D(20, 50),
        Point2D(30, 50),
        Point2D(40, 50),
        Point2D(50, 50),
        None,
        Point2D(60, 50),
    )
    frames = tuple(
        SpatialFrame(context, index, Pose2D((Keypoint2D("bodycentre", point),)))
        for index, point in enumerate(positions)
    )
    return RecordedWorldSeries(
        subject_id=subject_id,
        recording_id=context.recording_id,
        arena=RectangularArena(
            f"{subject_id}:arena",
            ROCHE_COORDINATE_FRAME,
            Point2D(0, 0),
            Point2D(100, 100),
        ),
        steps=tuple(ROCHE_RECORDED_TRAJECTORY_PROTOCOL.trajectory_steps(frames)),
        landmark_quality={},
    )


class WorldLabTests(unittest.TestCase):
    def _mock_sources(self):
        subjects = ROCHE_CONTROL_FITTING_SUBJECTS + ROCHE_CONTROL_DEVELOPMENT_SUBJECTS
        prepared = {subject_id: _series(subject_id) for subject_id in subjects}
        catalog = tuple(
            SimpleNamespace(subject_id=subject_id) for subject_id in subjects
        )
        return (
            patch("umwelt.world_lab.catalog_roche_open_field", return_value=catalog),
            patch(
                "umwelt.world_lab.load_recorded_world_series",
                side_effect=lambda recording: prepared[recording.subject_id],
            ),
            patch(
                "umwelt.world_lab.roche_provenance",
                return_value={"source_category": "recorded"},
            ),
        )

    def test_atomic_run_retains_hashed_artifacts_and_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "world-run"
            config = WorldLabConfig("fixture-world", root / "data", 1730, 1, target)
            mocks = self._mock_sources()
            with mocks[0], mocks[1], mocks[2]:
                result = run_world_lab(config)

            self.assertEqual(result.generated_series, 34)
            self.assertEqual(result.replicate_records, 34)
            manifest = json.loads((target / "artifact-manifest.json").read_text())
            self.assertEqual(len(manifest["artifacts"]) + 1, result.artifact_count)
            for entry in manifest["artifacts"]:
                content = (target / entry["name"]).read_bytes()
                self.assertEqual(entry["bytes"], len(content))
                self.assertEqual(entry["sha256"], hashlib.sha256(content).hexdigest())
            self.assertEqual(
                json.loads((target / "resolved-config.json").read_text())["output"][
                    "directory"
                ],
                str(target),
            )
            comparison = json.loads((target / "model-comparison.json").read_text())
            self.assertEqual(len(comparison["cross_validation"]), 12)
            self.assertEqual(len(comparison["legacy_diagnostic"]), 4)
            self.assertFalse(comparison["legacy_diagnostic_used_for_model_selection"])
            interventions = json.loads(
                (target / "intervention-comparison.json").read_text()
            )
            self.assertFalse(interventions["recorded_animal_intervention_counterpart"])
            with self.assertRaisesRegex(DataError, "already exists"):
                run_world_lab(config)

    def test_failed_publication_cleans_temporary_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "world-run"
            config = WorldLabConfig("fixture-world", root / "data", 1730, 1, target)
            mocks = self._mock_sources()
            with (
                mocks[0],
                mocks[1],
                mocks[2],
                patch("umwelt.world_lab.write_json", side_effect=OSError("disk error")),
            ):
                with self.assertRaisesRegex(OSError, "disk error"):
                    run_world_lab(config)
            self.assertFalse(target.exists())
            self.assertEqual(list(root.glob(".world-run-*")), [])


if __name__ == "__main__":
    unittest.main()
