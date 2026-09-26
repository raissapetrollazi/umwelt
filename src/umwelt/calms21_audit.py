"""Reproducible source audit before fitting a CalMS21 social model."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from math import hypot
from pathlib import Path
from typing import cast
from zipfile import ZipFile

from umwelt.calms21 import (
    CALMS21_ARCHIVE_MD5,
    Calms21Task1Sequence,
    iter_calms21_task1_sequences,
    scan_calms21_task1_sequence_ids,
)
from umwelt.dyad import DyadFrame, measure_dyad_frames
from umwelt.errors import DataError
from umwelt.spatial import Point2D


def _hash_archive(path: Path) -> tuple[str, str]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            md5.update(block)
            sha256.update(block)
    return md5.hexdigest(), sha256.hexdigest()


def _animal_key(sequence_id: str) -> str | None:
    match = re.fullmatch(
        r"task1/(?:train|test)/(mouse[0-9]+)_task1_annotator[0-9]+",
        sequence_id,
    )
    return match.group(1) if match is not None else None


def summarize_calms21_training_sequence(
    sequence: Calms21Task1Sequence,
) -> dict[str, object]:
    """Inspect the training poses without choosing a confidence cutoff."""

    score_present = 0
    score_missing = 0
    score_below_tenth = 0
    point_missing = 0
    neck_outside_image = 0
    neck_score_below_tenth = 0
    neck_score_below_half = 0
    neck_jumps_over_50 = 0
    neck_jumps_over_100 = 0
    neck_jumps_over_200 = 0
    neck_jumps_over_100_low_score = 0
    neck_max_adjacent_displacement = 0.0
    previous_necks: list[Point2D | None] = [None, None]

    def inspected_frames() -> Iterator[DyadFrame]:
        nonlocal score_present, score_missing, score_below_tenth
        nonlocal point_missing, neck_outside_image
        nonlocal neck_score_below_tenth, neck_score_below_half
        nonlocal neck_jumps_over_50, neck_jumps_over_100, neck_jumps_over_200
        nonlocal neck_jumps_over_100_low_score, neck_max_adjacent_displacement
        for frame in sequence.frames():
            for mouse_index, member in enumerate((frame.resident, frame.intruder)):
                if member.pose is None:
                    continue
                for keypoint in member.pose.keypoints:
                    if keypoint.confidence is None:
                        score_missing += 1
                    else:
                        score_present += 1
                        if keypoint.confidence < 0.1:
                            score_below_tenth += 1
                    if keypoint.point is None:
                        point_missing += 1
                neck_keypoint = member.pose.keypoint("neck")
                neck = neck_keypoint.point
                neck_score = neck_keypoint.confidence
                if neck_score is not None:
                    neck_score_below_tenth += neck_score < 0.1
                    neck_score_below_half += neck_score < 0.5
                if neck is not None and not (0 <= neck.x < 1024 and 0 <= neck.y < 570):
                    neck_outside_image += 1
                previous = previous_necks[mouse_index]
                if neck is not None and previous is not None:
                    step = hypot(neck.x - previous.x, neck.y - previous.y)
                    neck_max_adjacent_displacement = max(
                        neck_max_adjacent_displacement, step
                    )
                    neck_jumps_over_50 += step > 50
                    neck_jumps_over_100 += step > 100
                    neck_jumps_over_200 += step > 200
                    neck_jumps_over_100_low_score += (
                        step > 100 and neck_score is not None and neck_score < 0.5
                    )
                previous_necks[mouse_index] = neck
            yield frame

    measurement = measure_dyad_frames(inspected_frames())
    return {
        "sequence_id": sequence.sequence_id,
        "frame_count": sequence.frame_count,
        "source_keypoint_score_present_count": score_present,
        "source_keypoint_score_missing_count": score_missing,
        "source_keypoint_score_below_0_1_count": score_below_tenth,
        "source_keypoint_coordinate_missing_count": point_missing,
        "neck_position_outside_image_count": neck_outside_image,
        "neck_score_below_0_1_count": neck_score_below_tenth,
        "neck_score_below_0_5_count": neck_score_below_half,
        "neck_adjacent_displacement_over_50_px_count": neck_jumps_over_50,
        "neck_adjacent_displacement_over_100_px_count": neck_jumps_over_100,
        "neck_adjacent_displacement_over_200_px_count": neck_jumps_over_200,
        "neck_adjacent_displacement_over_100_px_with_current_score_below_0_5_count": (
            neck_jumps_over_100_low_score
        ),
        "neck_max_adjacent_displacement_px": neck_max_adjacent_displacement,
        "pair_geometry": measurement.compact_dict(),
    }


def audit_calms21_task1(archive_path: str | Path) -> dict[str, object]:
    """Verify bytes and train poses; inspect test sequence keys only."""

    path = Path(archive_path)
    if not path.is_file():
        raise DataError("CalMS21 Task 1 archive file is missing.")
    md5, sha256 = _hash_archive(path)
    if md5 != CALMS21_ARCHIVE_MD5:
        raise DataError("CalMS21 Task 1 archive MD5 differs from the published MD5.")

    with ZipFile(path) as archive:
        members = [
            {
                "name": info.filename,
                "compressed_bytes": info.compress_size,
                "uncompressed_bytes": info.file_size,
            }
            for info in archive.infolist()
        ]

    training_summaries = [
        summarize_calms21_training_sequence(sequence)
        for sequence in iter_calms21_task1_sequences(path)
    ]
    training_ids = tuple(str(item["sequence_id"]) for item in training_summaries)
    if len(training_ids) != len(set(training_ids)):
        raise DataError("CalMS21 training sequences have duplicate identifiers.")
    test_ids = scan_calms21_task1_sequence_ids(path, split="test")
    train_keys = {_animal_key(identifier) for identifier in training_ids}
    test_keys = {_animal_key(identifier) for identifier in test_ids}

    return {
        "dataset": "calms21-task1-v1",
        "source_doi": "10.22002/D1.1991",
        "source_license": "CC-BY-NC-SA-2.0 as linked by dataset paper",
        "archive_path": str(path.resolve()),
        "archive_bytes": path.stat().st_size,
        "archive_md5": md5,
        "archive_sha256": sha256,
        "archive_members": members,
        "train_sequence_count": len(training_ids),
        "test_sequence_count": len(test_ids),
        "train_frame_count": sum(
            cast(int, item["frame_count"]) for item in training_summaries
        ),
        "train_sequence_ids": list(training_ids),
        "test_sequence_ids": list(test_ids),
        "sequence_animal_key_overlap": sorted(
            key for key in train_keys & test_keys if key is not None
        ),
        "all_sequence_animal_keys_parsed": None not in train_keys | test_keys,
        "identity_limit": (
            "sequence names expose one mouse key; within-recording resident and "
            "intruder roles are known, but cross-recording intruder identities "
            "cannot be independently verified from these keys"
        ),
        "test_trajectory_outcomes_inspected": False,
        "training_sequences": training_summaries,
    }
