"""Tests for the pinned Roche open-field pose adapter."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from umwelt.datasets.roche_open_field import (
    ROCHE_KEYPOINTS,
    ROCHE_METADATA_COLUMNS,
    ROCHE_METADATA_FILE,
    ROCHE_OPEN_FIELD_DOI,
    ROCHE_RECORDING_COUNT,
    catalog_roche_open_field,
    load_roche_metadata,
    roche_provenance,
    select_roche_recordings,
    verify_roche_pose_archive,
)
from umwelt.errors import DataError
from umwelt.observations import ObservationSource


class RocheOpenFieldAdapterTests(unittest.TestCase):
    def _write_metadata(self, root: Path, *, columns=ROCHE_METADATA_COLUMNS) -> None:
        with (root / ROCHE_METADATA_FILE).open(
            "w", newline="", encoding="utf-8-sig"
        ) as handle:
            writer = csv.writer(handle, delimiter=";")
            writer.writerow(columns)
            for index in range(ROCHE_RECORDING_COUNT):
                if index < 8:
                    group, dosage = "control", "saline"
                elif index < 16:
                    group, dosage = "yohimbine", "1"
                elif index < 24:
                    group, dosage = "yohimbine", "3"
                else:
                    group, dosage = "yohimbine", "6"
                writer.writerow(
                    [
                        f"mouse-{index + 1:02d}",
                        f"mouse-{index + 1:02d}.csv",
                        group,
                        dosage,
                        f"mouse-{index + 1:02d}.mp4",
                    ]
                )

    def _write_pose(
        self,
        path: Path,
        *,
        indices: tuple[int, ...] = (0, 1, 2),
        keypoints: tuple[str, ...] = ROCHE_KEYPOINTS,
    ) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["scorer", *("fixture-scorer" for _ in range(len(keypoints) * 3))])
            writer.writerow(
                ["bodyparts", *(name for name in keypoints for _ in range(3))]
            )
            writer.writerow(
                [
                    "coords",
                    *(
                        coordinate
                        for _ in keypoints
                        for coordinate in ("x", "y", "likelihood")
                    ),
                ]
            )
            for frame_index in indices:
                values: list[object] = [frame_index]
                for keypoint_index, _ in enumerate(keypoints):
                    x = frame_index * 10.0 + keypoint_index
                    values.extend((x, x + 0.5, 0.9))
                writer.writerow(values)

    def _write_fixture(self, root: Path) -> None:
        self._write_metadata(root)
        for index in range(ROCHE_RECORDING_COUNT):
            self._write_pose(root / f"mouse-{index + 1:02d}.csv")

    def test_metadata_uses_verified_semicolon_bom_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_fixture(root)
            records = load_roche_metadata(root)

        self.assertEqual(len(records), 32)
        self.assertEqual(len({record.animal_id for record in records}), 32)
        self.assertEqual(len({record.dlc_file for record in records}), 32)
        self.assertEqual(records[0].group, "control")
        self.assertEqual(records[8].dosage, "1")

    def test_catalog_matches_all_metadata_rows_to_pinned_dlc_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_fixture(root)
            recordings = catalog_roche_open_field(root)

        self.assertEqual(len(recordings), 32)
        self.assertEqual(recordings[0].subject_id, "mouse-01")
        self.assertEqual(recordings[0].source, ObservationSource.RECORDED)
        self.assertEqual(recordings[0].coordinate_frame.unit, "px")
        self.assertIsNone(recordings[0].sampling_rate_hz)
        self.assertEqual(recordings[0].scorer, "fixture-scorer")

    def test_recording_stream_preserves_all_keypoints_and_frame_indices(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_fixture(root)
            recording = catalog_roche_open_field(root)[0]
            frames = tuple(recording.frames(validate_frame_count=False))

        self.assertEqual([frame.frame_index for frame in frames], [0, 1, 2])
        pose = frames[0].pose
        assert pose is not None
        self.assertEqual(tuple(keypoint.name for keypoint in pose.keypoints), ROCHE_KEYPOINTS)
        self.assertEqual(pose.keypoint("bodycentre").point.x, 9.0)
        self.assertEqual(pose.keypoint("bodycentre").confidence, 0.9)
        self.assertEqual(pose.keypoint("tl").point.y, 0.5)

    def test_canonical_frame_count_is_checked_when_stream_is_exhausted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_fixture(root)
            recording = catalog_roche_open_field(root)[0]
            with self.assertRaisesRegex(DataError, "contains 3 frames"):
                tuple(recording.frames())

    def test_frame_indices_must_start_at_zero_and_be_sequential(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_fixture(root)
            self._write_pose(root / "mouse-01.csv", indices=(0, 2))
            recording = catalog_roche_open_field(root)[0]
            with self.assertRaisesRegex(DataError, "expected 1, found 2"):
                tuple(recording.frames(validate_frame_count=False))

    def test_catalog_rejects_wrong_keypoint_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_fixture(root)
            wrong = ("nose", *tuple(name for name in ROCHE_KEYPOINTS if name != "nose"))
            self._write_pose(root / "mouse-01.csv", keypoints=wrong)
            with self.assertRaisesRegex(DataError, "keypoints do not match"):
                catalog_roche_open_field(root)

    def test_metadata_rejects_wrong_columns_and_incomplete_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_metadata(root, columns=("Animal ID", "DLC file"))
            with self.assertRaisesRegex(DataError, "columns do not match"):
                load_roche_metadata(root)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_fixture(root)
            (root / "mouse-32.csv").unlink()
            with self.assertRaisesRegex(DataError, "resolved to 0 local files"):
                catalog_roche_open_field(root)

    def test_selection_is_by_unique_animal_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_fixture(root)
            catalog = catalog_roche_open_field(root)
            selected = select_roche_recordings(catalog, ["mouse-01", "mouse-09"])

        self.assertEqual([item.subject_id for item in selected], ["mouse-01", "mouse-09"])
        with self.assertRaisesRegex(DataError, "duplicates"):
            select_roche_recordings(catalog, ["mouse-01", "mouse-01"])
        with self.assertRaisesRegex(DataError, "Unknown Roche"):
            select_roche_recordings(catalog, ["missing"])

    def test_provenance_keeps_unknown_physical_calibration_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_fixture(root)
            provenance = roche_provenance(root, subject_ids=["mouse-01"])

        self.assertEqual(provenance["doi"], ROCHE_OPEN_FIELD_DOI)
        self.assertIsNone(provenance["sampling_rate_hz"])
        self.assertIsNone(provenance["physical_arena_dimensions"])
        self.assertEqual(provenance["coordinate_frame"]["unit"], "px")
        self.assertIsNone(provenance["coordinate_frame"]["physical_calibration"])
        self.assertEqual(len(provenance["selected_recordings"]), 1)

    def test_archive_verification_reports_missing_or_mismatched_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = verify_roche_pose_archive(root)
            (root / "data.zip").write_bytes(b"not-the-published-archive")
            mismatched = verify_roche_pose_archive(root)

        self.assertFalse(missing.valid)
        self.assertIsNone(missing.actual_md5)
        self.assertFalse(mismatched.valid)
        self.assertIsNotNone(mismatched.actual_md5)


if __name__ == "__main__":
    unittest.main()
