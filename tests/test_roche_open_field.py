"""Tests for the pinned Roche open-field pose adapter."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from umwelt.datasets.roche_open_field import (
    ROCHE_ARENA_LANDMARKS,
    ROCHE_KEYPOINTS,
    ROCHE_METADATA_COLUMNS,
    ROCHE_METADATA_FILE,
    ROCHE_OPEN_FIELD_DOI,
    ROCHE_MOUSE_KEYPOINTS,
    ROCHE_POSE_MANIFEST,
    ROCHE_RECORDING_COUNT,
    catalog_roche_open_field,
    load_roche_metadata,
    roche_provenance,
    select_roche_recordings,
    verify_roche_metadata,
    verify_roche_pose_archive,
)
from umwelt.errors import DataError, DataIntegrityError
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
                    group, dosage = "Control", "0"
                elif index < 16:
                    group, dosage = "Yohimbine", "1"
                elif index < 24:
                    group, dosage = "Yohimbine", "3"
                else:
                    group, dosage = "Yohimbine", "6"
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
            with self.assertRaisesRegex(DataIntegrityError, "published by Zenodo"):
                load_roche_metadata(root)
            records = load_roche_metadata(root, verify_integrity=False)

        self.assertEqual(len(records), 32)
        self.assertEqual(len({record.animal_id for record in records}), 32)
        self.assertEqual(len({record.dlc_file for record in records}), 32)
        self.assertEqual(records[0].group, "Control")
        self.assertEqual(records[0].dosage, "0")
        self.assertEqual(records[8].dosage, "1")

    def test_catalog_matches_all_metadata_rows_to_pinned_dlc_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_fixture(root)
            recordings = catalog_roche_open_field(root, verify_integrity=False)

        self.assertEqual(len(recordings), 32)
        self.assertEqual(recordings[0].subject_id, "mouse-01")
        self.assertEqual(recordings[0].source, ObservationSource.RECORDED)
        self.assertEqual(recordings[0].coordinate_frame.unit, "px")
        self.assertIsNone(recordings[0].sampling_rate_hz)
        self.assertEqual(recordings[0].scorer, "fixture-scorer")

    def test_pose_manifest_covers_every_recording_with_exact_totals(self) -> None:
        self.assertEqual(len(ROCHE_POSE_MANIFEST), ROCHE_RECORDING_COUNT)
        self.assertEqual(
            sum(entry.size_bytes for entry in ROCHE_POSE_MANIFEST.values()),
            1_607_375_060,
        )
        self.assertEqual(
            sum(entry.frame_count for entry in ROCHE_POSE_MANIFEST.values()),
            1_726_595,
        )
        self.assertTrue(
            all(len(entry.sha256) == 64 for entry in ROCHE_POSE_MANIFEST.values())
        )

    def test_recording_stream_preserves_all_keypoints_and_frame_indices(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_fixture(root)
            recording = catalog_roche_open_field(root, verify_integrity=False)[0]
            frames = tuple(recording.frames(validate_frame_count=False))

        self.assertEqual([frame.frame_index for frame in frames], [0, 1, 2])
        pose = frames[0].pose
        assert pose is not None
        self.assertEqual(
            tuple(keypoint.name for keypoint in pose.keypoints), ROCHE_MOUSE_KEYPOINTS
        )
        self.assertEqual(pose.keypoint("bodycentre").point.x, 9.0)
        self.assertEqual(pose.keypoint("bodycentre").confidence, 0.9)
        landmarks = frames[0].landmarks
        assert landmarks is not None
        self.assertEqual(
            tuple(landmark.name for landmark in landmarks.landmarks),
            ROCHE_ARENA_LANDMARKS,
        )
        self.assertEqual(landmarks.landmark("tl").point.y, 0.5)
        with self.assertRaisesRegex(DataError, "Unknown pose keypoint"):
            pose.keypoint("tl")

    def test_canonical_frame_count_is_checked_when_stream_is_exhausted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_fixture(root)
            recording = catalog_roche_open_field(root, verify_integrity=False)[0]
            with self.assertRaisesRegex(DataError, "contains 3 frames"):
                tuple(recording.frames())

    def test_frame_indices_must_start_at_zero_and_be_sequential(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_fixture(root)
            self._write_pose(root / "mouse-01.csv", indices=(0, 2))
            recording = catalog_roche_open_field(root, verify_integrity=False)[0]
            with self.assertRaisesRegex(DataError, "expected 1, found 2"):
                tuple(recording.frames(validate_frame_count=False))

    def test_catalog_rejects_wrong_keypoint_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_fixture(root)
            wrong = ("nose", *tuple(name for name in ROCHE_KEYPOINTS if name != "nose"))
            self._write_pose(root / "mouse-01.csv", keypoints=wrong)
            with self.assertRaisesRegex(DataError, "keypoints do not match"):
                catalog_roche_open_field(root, verify_integrity=False)

    def test_metadata_rejects_wrong_columns_and_incomplete_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_metadata(root, columns=("Animal ID", "DLC file"))
            with self.assertRaisesRegex(DataError, "columns do not match"):
                load_roche_metadata(root, verify_integrity=False)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_fixture(root)
            (root / "mouse-32.csv").unlink()
            with self.assertRaisesRegex(DataError, "resolved to 0 local files"):
                catalog_roche_open_field(root, verify_integrity=False)

    def test_metadata_rejects_wrong_categories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_metadata(root)
            path = root / ROCHE_METADATA_FILE
            text = path.read_text(encoding="utf-8-sig")
            path.write_text(
                text.replace("Control;0", "Placebo;0"), encoding="utf-8-sig"
            )

            with self.assertRaisesRegex(DataError, "does not match the verified"):
                load_roche_metadata(root, verify_integrity=False)

    def test_metadata_rejects_empty_or_mismatched_source_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_metadata(root)
            path = root / ROCHE_METADATA_FILE
            text = path.read_text(encoding="utf-8-sig")
            path.write_text(text.replace("Control;0", "Invented;"), encoding="utf-8-sig")

            with self.assertRaisesRegex(DataError, "empty dosage"):
                load_roche_metadata(root, verify_integrity=False)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_metadata(root)
            path = root / ROCHE_METADATA_FILE
            text = path.read_text(encoding="utf-8-sig")
            path.write_text(
                text.replace("mouse-02.mp4", "mouse-01.mp4"), encoding="utf-8-sig"
            )

            with self.assertRaisesRegex(DataError, "video filenames must be unique"):
                load_roche_metadata(root, verify_integrity=False)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_metadata(root)
            path = root / ROCHE_METADATA_FILE
            text = path.read_text(encoding="utf-8-sig")
            path.write_text(
                text.replace("mouse-02.mp4", "different.mp4"), encoding="utf-8-sig"
            )

            with self.assertRaisesRegex(DataError, "same recording"):
                load_roche_metadata(root, verify_integrity=False)

    def test_pose_resolution_rejects_direct_and_extracted_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_fixture(root)
            extracted = root / "data/Yohimbine_Roche"
            extracted.mkdir(parents=True)
            self._write_pose(extracted / "mouse-01.csv")

            with self.assertRaisesRegex(DataError, "resolved to 2 local files"):
                catalog_roche_open_field(root, verify_integrity=False)

    def test_selection_is_by_unique_animal_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_fixture(root)
            catalog = catalog_roche_open_field(root, verify_integrity=False)
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
            provenance = roche_provenance(
                root, subject_ids=["mouse-01"], verify_integrity=False
            )

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

    def test_published_metadata_verification_accepts_only_the_real_source(self) -> None:
        real_root = Path("data/raw/roche-open-field")
        if not real_root.is_dir():
            self.skipTest("Downloaded Roche source is not available locally.")

        verification = verify_roche_metadata(real_root)
        records = load_roche_metadata(real_root)

        self.assertTrue(verification.valid)
        self.assertEqual(len(records), 32)


if __name__ == "__main__":
    unittest.main()
