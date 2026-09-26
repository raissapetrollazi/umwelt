"""Tests for the optional CalMS21 Task 1 adapter."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from umwelt.calms21 import (
    Calms21Task1Sequence,
    iter_calms21_task1_sequences,
    scan_calms21_task1_sequence_ids,
)
from umwelt.calms21_audit import summarize_calms21_training_sequence
from umwelt.dyad import measure_dyad_frames
from umwelt.errors import DataError
from umwelt.spatial import Point2D


def source_frame(resident_x: float | None, intruder_x: float) -> dict[str, object]:
    positions = []
    scores = []
    for x in (resident_x, intruder_x):
        positions.append([[x] * 7, [20.0] * 7])
        scores.append([0.9, 0.8, 0.7, 0.01, 0.6, 0.5, 0.4])
    return {"keypoints": positions, "scores": scores}


class Calms21AdapterTests(unittest.TestCase):
    def test_preserves_roles_scores_and_missing_neck_without_threshold(self) -> None:
        first = source_frame(1.0, 4.0)
        second = source_frame(None, 5.0)
        sequence = Calms21Task1Sequence(
            "task1/train/mouse001_task1_annotator1",
            [first["keypoints"], second["keypoints"]],
            [first["scores"], second["scores"]],
        )

        frames = tuple(sequence.frames())
        self.assertEqual(sequence.frame_count, 2)
        self.assertEqual(frames[0].resident.pose.keypoint("neck").point, Point2D(1, 20))
        self.assertEqual(frames[0].intruder.pose.keypoint("neck").point, Point2D(4, 20))
        self.assertEqual(frames[0].resident.pose.keypoint("neck").confidence, 0.01)
        self.assertEqual(len(frames[0].resident.pose.keypoints), 7)
        self.assertIsNone(frames[1].resident.pose.keypoint("neck").point)
        self.assertEqual(measure_dyad_frames(frames).pair_distances_px, (3.0,))

    def test_invalid_shape_and_score_are_rejected(self) -> None:
        frame = source_frame(1.0, 4.0)
        bad_shape = Calms21Task1Sequence(
            "task1/train/mouse001", [frame["keypoints"]], [[[0.5] * 7]]
        )
        with self.assertRaisesRegex(DataError, "mouse scores must have length 2"):
            tuple(bad_shape.frames())

        scores = frame["scores"]
        scores[0][3] = 1.2
        bad_score = Calms21Task1Sequence(
            "task1/train/mouse001", [frame["keypoints"]], [scores]
        )
        with self.assertRaisesRegex(DataError, "confidence must be between"):
            tuple(bad_score.frames())

    def test_archive_reader_selects_only_requested_json(self) -> None:
        frame = source_frame(1.0, 4.0)
        record = {"keypoints": [frame["keypoints"]], "scores": [frame["scores"]]}
        train = {"annotator-id_0": {"task1/train/mouse001_task1_annotator1": record}}
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "task1.zip"
            with ZipFile(archive_path, "w") as archive:
                archive.writestr("task1/calms21_task1_train.json", json.dumps(train))
                archive.writestr("task1/calms21_task1_test.json", "not valid JSON")

            sequences = tuple(iter_calms21_task1_sequences(archive_path))

        self.assertEqual(len(sequences), 1)
        self.assertEqual(sequences[0].frame_count, 1)
        self.assertEqual(len(tuple(sequences[0].frames())), 1)

    def test_identity_scan_reads_only_sequence_keys(self) -> None:
        frame = source_frame(1.0, 4.0)
        record = {"keypoints": [frame["keypoints"]], "scores": [frame["scores"]]}
        test = {"annotator-id_0": {"task1/test/mouse002_task1_annotator1": record}}
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "task1.zip"
            with ZipFile(archive_path, "w") as archive:
                archive.writestr("task1/calms21_task1_test.json", json.dumps(test))

            identifiers = scan_calms21_task1_sequence_ids(archive_path, split="test")

        self.assertEqual(identifiers, ("task1/test/mouse002_task1_annotator1",))

    def test_training_audit_reports_low_scores_and_outside_image(self) -> None:
        frame = source_frame(1100.0, 4.0)
        sequence = Calms21Task1Sequence(
            "task1/train/mouse001", [frame["keypoints"]], [frame["scores"]]
        )

        report = summarize_calms21_training_sequence(sequence)

        self.assertEqual(report["frame_count"], 1)
        self.assertEqual(report["source_keypoint_score_present_count"], 14)
        self.assertEqual(report["source_keypoint_score_below_0_1_count"], 2)
        self.assertEqual(report["neck_position_outside_image_count"], 1)
        self.assertEqual(
            report["pair_geometry"]["quality_control"]["valid_pair_position_count"], 1
        )


if __name__ == "__main__":
    unittest.main()
