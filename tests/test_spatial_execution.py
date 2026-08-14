"""Integration tests for compact and reproducible v0.2 spatial artifacts."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from umwelt.datasets.roche_open_field import RocheRecording
from umwelt.observations import ObservationSource
from umwelt.spatial import (
    Keypoint2D,
    LandmarkSet2D,
    Point2D,
    Pose2D,
    SpatialContext,
    SpatialFrame,
)
from umwelt.spatial_config import SpatialLabConfig
from umwelt.spatial_execution import execute_spatial_lab
from umwelt.spatial_protocol import (
    ROCHE_CONTROL_DEVELOPMENT_SUBJECTS,
    ROCHE_CONTROL_FITTING_SUBJECTS,
)
from umwelt.datasets.roche_open_field import ROCHE_COORDINATE_FRAME


@dataclass(frozen=True)
class _Recording:
    values: tuple[SpatialFrame, ...]

    def frames(self) -> tuple[SpatialFrame, ...]:
        return self.values


def _landmarks() -> LandmarkSet2D:
    return LandmarkSet2D(
        (
            Keypoint2D("tl", Point2D(0, 0)),
            Keypoint2D("tr", Point2D(10, 0)),
            Keypoint2D("bl", Point2D(0, 10)),
            Keypoint2D("br", Point2D(10, 10)),
        )
    )


def _recording(subject_id: str) -> _Recording:
    context = SpatialContext(
        subject_id,
        f"recording:{subject_id}",
        ObservationSource.RECORDED,
        ROCHE_COORDINATE_FRAME,
    )
    points = (Point2D(2, 2), Point2D(3, 2), Point2D(4, 2), None, Point2D(4, 3))
    frames = []
    for index, point in enumerate(points):
        pose = (
            None if point is None else Pose2D((Keypoint2D("bodycentre", point, 0.9),))
        )
        frames.append(SpatialFrame(context, index, pose, _landmarks()))
    return _Recording(tuple(frames))


class SpatialExecutionTests(unittest.TestCase):
    def test_execution_writes_frozen_metrics_and_equal_exposure(self) -> None:
        subjects = ROCHE_CONTROL_FITTING_SUBJECTS + ROCHE_CONTROL_DEVELOPMENT_SUBJECTS
        recordings = cast(
            dict[str, RocheRecording],
            {subject_id: _recording(subject_id) for subject_id in subjects},
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "run"
            config = SpatialLabConfig(
                experiment_id="fixture-spatial",
                dataset_directory=root / "data",
                seed=1729,
                replicates=2,
                output_directory=target,
            )
            _, record_count, generated, artifact_count = execute_spatial_lab(
                config,
                recordings,
                target,
                source_provenance={"fixture": True},
            )

            protocol = json.loads((target / "protocol.json").read_text())
            records = [
                json.loads(line)
                for line in (target / "replicate-metrics.jsonl")
                .read_text()
                .splitlines()
            ]
            report = (target / "report.md").read_text()
            manifest = json.loads((target / "artifact-manifest.json").read_text())

        metric_ids = [
            item["id"] for item in protocol["evaluation"]["registered_metrics"]
        ]
        self.assertEqual(
            metric_ids,
            [
                "path-length-px",
                "adjacent-displacement-px-distribution",
                "absolute-turning-radians-distribution",
            ],
        )
        self.assertEqual(record_count, 4)
        self.assertEqual(generated, 4)
        self.assertEqual(artifact_count, 8)
        self.assertEqual(len(records), 4)
        for record in records:
            recorded_qc = record["recorded"]["quality_control"]
            synthetic_qc = record["synthetic"]["quality_control"]
            self.assertEqual(
                recorded_qc["valid_transition_count"],
                synthetic_qc["valid_transition_count"],
            )
            self.assertEqual(
                set(record["comparison"]),
                {
                    "path_length_signed_difference_px",
                    "path_length_absolute_difference_px",
                    "path_length_signed_relative_difference",
                    "path_length_relative_difference",
                    "adjacent_displacement_wasserstein_px",
                    "absolute_turning_wasserstein_radians",
                },
            )
        self.assertIn("two designated development animals", report)
        self.assertNotIn("boundary_distance_wasserstein", report)
        self.assertEqual(len(manifest), 7)


if __name__ == "__main__":
    unittest.main()
