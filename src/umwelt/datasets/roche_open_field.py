"""Adapter for the pinned Roche open-field pose dataset used by Umwelt v0.2."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path, PurePosixPath

from umwelt.errors import DataError
from umwelt.observations import ObservationSource
from umwelt.spatial import (
    CoordinateFrame,
    Keypoint2D,
    LandmarkSet2D,
    Point2D,
    Pose2D,
    SpatialFrame,
)

ROCHE_OPEN_FIELD_DATASET_ID = "roche-open-field-zenodo-8188683-v1"
ROCHE_OPEN_FIELD_DOI = "10.5281/zenodo.8188683"
ROCHE_OPEN_FIELD_RECORD_URL = "https://zenodo.org/records/8188683"
ROCHE_OPEN_FIELD_LICENSE = "CC-BY-4.0"
ROCHE_METADATA_FILE = "METADATA_ROCHE.csv"
ROCHE_POSE_ARCHIVE = "data.zip"
ROCHE_POSE_ARCHIVE_MD5 = "8b48f2060146c03d578d5d6971aa70e9"
ROCHE_RECORDING_COUNT = 32
ROCHE_MIN_FRAME_COUNT = 53_914
ROCHE_MAX_FRAME_COUNT = 53_964
ROCHE_METADATA_COLUMNS = (
    "Animal ID",
    "DLC file",
    "Group",
    "Dosage",
    "Video",
)
ROCHE_ARENA_LANDMARKS = (
    "tl",
    "tr",
    "bl",
    "br",
)
ROCHE_MOUSE_KEYPOINTS = (
    "nose",
    "headcentre",
    "neck",
    "earl",
    "earr",
    "bodycentre",
    "bcl",
    "bcr",
    "hipl",
    "hipr",
    "tailbase",
    "tailcentre",
    "tailtip",
)
# Ordered columns in the published DeepLabCut CSVs. Arena landmarks are source
# measurements, but they are not part of the animal's body pose.
ROCHE_KEYPOINTS = ROCHE_ARENA_LANDMARKS + ROCHE_MOUSE_KEYPOINTS
ROCHE_COORDINATE_FRAME = CoordinateFrame("roche-open-field-image", "px")


@dataclass(frozen=True, slots=True)
class RocheRecordingMetadata:
    """One source metadata row without interpreting treatment as behavior."""

    animal_id: str
    dlc_file: str
    group: str
    dosage: str
    video: str

    def __post_init__(self) -> None:
        if not self.animal_id:
            raise DataError("Roche metadata contains an empty animal identifier.")
        if not self.dlc_file:
            raise DataError("Roche metadata contains an empty DLC filename.")

    def to_dict(self) -> dict[str, str]:
        """Return the original metadata categories in a JSON-compatible shape."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class ArchiveVerification:
    """MD5 verification result for the published pose archive."""

    path: str
    expected_md5: str
    actual_md5: str | None
    valid: bool


@dataclass(frozen=True, slots=True)
class RocheRecording:
    """One validated source recording that can stream explicit spatial frames."""

    metadata: RocheRecordingMetadata
    path: Path
    scorer: str

    @property
    def subject_id(self) -> str:
        return self.metadata.animal_id

    @property
    def recording_id(self) -> str:
        return self.metadata.dlc_file

    @property
    def source(self) -> ObservationSource:
        return ObservationSource.RECORDED

    @property
    def coordinate_frame(self) -> CoordinateFrame:
        return ROCHE_COORDINATE_FRAME

    @property
    def sampling_rate_hz(self) -> None:
        """The downloaded source material does not establish a sampling rate."""

        return None

    def frames(self, *, validate_frame_count: bool = True) -> Iterator[SpatialFrame]:
        """Stream frames and validate indices; canonical count is checked on exhaustion."""

        yield from _iter_pose_frames(self.path, validate_frame_count=validate_frame_count)


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_roche_pose_archive(directory: str | Path) -> ArchiveVerification:
    """Verify data.zip against the MD5 published by the Zenodo record."""

    path = Path(directory) / ROCHE_POSE_ARCHIVE
    if not path.is_file():
        return ArchiveVerification(
            path=str(path),
            expected_md5=ROCHE_POSE_ARCHIVE_MD5,
            actual_md5=None,
            valid=False,
        )
    actual = _md5(path)
    return ArchiveVerification(
        path=str(path),
        expected_md5=ROCHE_POSE_ARCHIVE_MD5,
        actual_md5=actual,
        valid=actual == ROCHE_POSE_ARCHIVE_MD5,
    )


def load_roche_metadata(directory: str | Path) -> tuple[RocheRecordingMetadata, ...]:
    """Load and validate the semicolon-delimited UTF-8-BOM metadata table."""

    path = Path(directory) / ROCHE_METADATA_FILE
    try:
        handle = path.open(newline="", encoding="utf-8-sig")
    except OSError as error:
        raise DataError(f"Could not open Roche metadata file: {error}") from error

    records: list[RocheRecordingMetadata] = []
    with handle:
        reader = csv.DictReader(handle, delimiter=";")
        columns = tuple(reader.fieldnames or ())
        if columns != ROCHE_METADATA_COLUMNS:
            raise DataError(
                "Roche metadata columns do not match the pinned schema: "
                + ", ".join(columns)
            )
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise DataError(
                    f"Roche metadata contains extra fields at CSV row {row_number}."
                )
            records.append(
                RocheRecordingMetadata(
                    animal_id=(row["Animal ID"] or "").strip(),
                    dlc_file=(row["DLC file"] or "").strip(),
                    group=(row["Group"] or "").strip(),
                    dosage=(row["Dosage"] or "").strip(),
                    video=(row["Video"] or "").strip(),
                )
            )

    if len(records) != ROCHE_RECORDING_COUNT:
        raise DataError(
            f"Roche metadata must contain exactly {ROCHE_RECORDING_COUNT} recordings; "
            f"found {len(records)}."
        )
    animal_ids = [record.animal_id for record in records]
    dlc_files = [record.dlc_file for record in records]
    if len(animal_ids) != len(set(animal_ids)):
        raise DataError("Roche metadata animal identifiers must be unique.")
    if len(dlc_files) != len(set(dlc_files)):
        raise DataError("Roche metadata DLC filenames must be unique.")

    group_counts = Counter((record.group, record.dosage) for record in records)
    if sorted(group_counts.values()) != [8, 8, 8, 8]:
        raise DataError(
            "Roche metadata does not preserve the verified four groups of eight recordings."
        )
    return tuple(records)


def _safe_source_path(text: str) -> PurePosixPath:
    path = PurePosixPath(text.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise DataError(f"Unsafe Roche DLC source path: {text}")
    return path


def resolve_roche_pose_files(
    directory: str | Path,
    metadata: Iterable[RocheRecordingMetadata] | None = None,
) -> dict[str, Path]:
    """Resolve every metadata DLC filename to exactly one extracted CSV file."""

    root = Path(directory)
    records = tuple(metadata) if metadata is not None else load_roche_metadata(root)
    resolved: dict[str, Path] = {}
    claimed: set[Path] = set()

    for record in records:
        relative = _safe_source_path(record.dlc_file)
        direct = root.joinpath(*relative.parts)
        matches = [direct] if direct.is_file() else [
            candidate
            for candidate in root.rglob(relative.name)
            if candidate.is_file()
        ]
        if len(matches) != 1:
            raise DataError(
                f"Roche DLC file {record.dlc_file!r} resolved to {len(matches)} "
                "local files; expected exactly one."
            )
        path = matches[0].resolve()
        if path in claimed:
            raise DataError("Two Roche metadata rows resolve to the same DLC file.")
        claimed.add(path)
        resolved[record.animal_id] = path

    if len(resolved) != ROCHE_RECORDING_COUNT:
        raise DataError(
            f"Roche pose-file resolution must produce {ROCHE_RECORDING_COUNT} recordings."
        )
    return resolved


def _validate_header_rows(path: Path, rows: tuple[list[str], list[str], list[str]]) -> str:
    scorer_row, bodypart_row, coordinate_row = rows
    expected_columns = 1 + len(ROCHE_KEYPOINTS) * 3
    if any(len(row) != expected_columns for row in rows):
        raise DataError(f"Roche pose header has an unexpected column count: {path}")

    if scorer_row[0].strip().lower() != "scorer":
        raise DataError(f"Roche pose file is missing the DeepLabCut scorer header: {path}")
    scorers = {value.strip() for value in scorer_row[1:] if value.strip()}
    if len(scorers) != 1 or any(not value.strip() for value in scorer_row[1:]):
        raise DataError(f"Roche pose file must declare one scorer for all keypoints: {path}")

    expected_bodyparts = tuple(
        keypoint for keypoint in ROCHE_KEYPOINTS for _ in range(3)
    )
    if bodypart_row[0].strip().lower() != "bodyparts":
        raise DataError(f"Roche pose file is missing the DeepLabCut bodyparts header: {path}")
    if tuple(value.strip() for value in bodypart_row[1:]) != expected_bodyparts:
        raise DataError(f"Roche pose keypoints do not match the pinned schema: {path}")

    expected_coordinates = tuple(
        coordinate
        for _ in ROCHE_KEYPOINTS
        for coordinate in ("x", "y", "likelihood")
    )
    if coordinate_row[0].strip().lower() != "coords":
        raise DataError(f"Roche pose file is missing the DeepLabCut coords header: {path}")
    if tuple(value.strip().lower() for value in coordinate_row[1:]) != expected_coordinates:
        raise DataError(
            f"Roche pose coordinate columns do not match x/y/likelihood triples: {path}"
        )
    return next(iter(scorers))


def _read_header(path: Path) -> str:
    try:
        handle = path.open(newline="", encoding="utf-8-sig")
    except OSError as error:
        raise DataError(f"Could not open Roche pose file {path}: {error}") from error
    with handle:
        reader = csv.reader(handle)
        try:
            rows = (next(reader), next(reader), next(reader))
        except StopIteration as error:
            raise DataError(f"Roche pose file has fewer than three header rows: {path}") from error
    return _validate_header_rows(path, rows)


def catalog_roche_open_field(directory: str | Path) -> tuple[RocheRecording, ...]:
    """Validate metadata/file correspondence and return lightweight recordings."""

    root = Path(directory)
    records = load_roche_metadata(root)
    paths = resolve_roche_pose_files(root, records)
    return tuple(
        RocheRecording(
            metadata=record,
            path=paths[record.animal_id],
            scorer=_read_header(paths[record.animal_id]),
        )
        for record in records
    )


def select_roche_recordings(
    recordings: Iterable[RocheRecording],
    subject_ids: Iterable[str],
) -> tuple[RocheRecording, ...]:
    """Select recordings by unique animal identifier without file-level leakage."""

    catalog = tuple(recordings)
    by_id = {recording.subject_id: recording for recording in catalog}
    selected_ids = tuple(subject_ids)
    if not selected_ids:
        raise DataError("At least one Roche animal must be selected.")
    if len(selected_ids) != len(set(selected_ids)):
        raise DataError("Roche animal selection contains duplicates.")
    unknown = sorted(set(selected_ids) - set(by_id))
    if unknown:
        raise DataError("Unknown Roche animal identifiers: " + ", ".join(unknown))
    return tuple(by_id[subject_id] for subject_id in selected_ids)


def _optional_float(text: str, *, label: str) -> float | None:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        value = float(stripped)
    except ValueError as error:
        raise DataError(f"Invalid {label}: {text!r}") from error
    if not isfinite(value):
        raise DataError(f"{label} must be finite.")
    return value


def _parse_spatial_frame(
    values: list[str], *, path: Path, row_number: int
) -> tuple[Pose2D, LandmarkSet2D]:
    keypoints: list[Keypoint2D] = []
    for index, name in enumerate(ROCHE_KEYPOINTS):
        offset = 1 + index * 3
        x = _optional_float(
            values[offset], label=f"{name} x coordinate in {path.name} row {row_number}"
        )
        y = _optional_float(
            values[offset + 1], label=f"{name} y coordinate in {path.name} row {row_number}"
        )
        likelihood = _optional_float(
            values[offset + 2], label=f"{name} likelihood in {path.name} row {row_number}"
        )
        if (x is None) != (y is None):
            raise DataError(
                f"Roche pose coordinates must be missing together for {name} "
                f"in {path.name} row {row_number}."
            )
        point = None if x is None else Point2D(x, y)  # type: ignore[arg-type]
        keypoints.append(Keypoint2D(name=name, point=point, confidence=likelihood))
    by_name = {keypoint.name: keypoint for keypoint in keypoints}
    pose = Pose2D(tuple(by_name[name] for name in ROCHE_MOUSE_KEYPOINTS))
    landmarks = LandmarkSet2D(
        tuple(by_name[name] for name in ROCHE_ARENA_LANDMARKS)
    )
    return pose, landmarks


def _iter_pose_frames(path: Path, *, validate_frame_count: bool) -> Iterator[SpatialFrame]:
    try:
        handle = path.open(newline="", encoding="utf-8-sig")
    except OSError as error:
        raise DataError(f"Could not open Roche pose file {path}: {error}") from error

    expected_columns = 1 + len(ROCHE_KEYPOINTS) * 3
    with handle:
        reader = csv.reader(handle)
        try:
            header = (next(reader), next(reader), next(reader))
        except StopIteration as error:
            raise DataError(f"Roche pose file has fewer than three header rows: {path}") from error
        _validate_header_rows(path, header)

        expected_index = 0
        for row_number, values in enumerate(reader, start=4):
            if len(values) != expected_columns:
                raise DataError(
                    f"Roche pose row {row_number} in {path.name} has {len(values)} "
                    f"columns; expected {expected_columns}."
                )
            try:
                frame_index = int(values[0])
            except ValueError as error:
                raise DataError(
                    f"Invalid Roche frame index in {path.name} row {row_number}."
                ) from error
            if frame_index != expected_index:
                raise DataError(
                    f"Roche frame indices must start at 0 and be sequential in {path.name}; "
                    f"expected {expected_index}, found {frame_index}."
                )
            pose, landmarks = _parse_spatial_frame(
                values, path=path, row_number=row_number
            )
            yield SpatialFrame(
                frame_index=frame_index,
                pose=pose,
                landmarks=landmarks,
            )
            expected_index += 1

    if validate_frame_count and not (
        ROCHE_MIN_FRAME_COUNT <= expected_index <= ROCHE_MAX_FRAME_COUNT
    ):
        raise DataError(
            f"Roche pose file {path.name} contains {expected_index} frames; expected "
            f"{ROCHE_MIN_FRAME_COUNT}-{ROCHE_MAX_FRAME_COUNT}."
        )


def roche_provenance(
    directory: str | Path,
    *,
    subject_ids: Iterable[str] | None = None,
    verify_archive: bool = False,
) -> dict[str, object]:
    """Return source provenance without materializing pose trajectories."""

    catalog = catalog_roche_open_field(directory)
    selected = catalog if subject_ids is None else select_roche_recordings(catalog, subject_ids)
    archive = verify_roche_pose_archive(directory) if verify_archive else None
    return {
        "source_category": ObservationSource.RECORDED.value,
        "record_title": (
            "Raw video and pose estimation data of top view open field mouse behavior "
            "recordings after yohimbine injections"
        ),
        "doi": ROCHE_OPEN_FIELD_DOI,
        "record_url": ROCHE_OPEN_FIELD_RECORD_URL,
        "license": ROCHE_OPEN_FIELD_LICENSE,
        "authors": [
            "Lukas M. von Ziegler",
            "Fabienne K. Roessler",
            "Oliver Sturman",
            "Eoin C. O'Connor",
            "Johannes Bohacek",
        ],
        "adapter": "umwelt.datasets.roche_open_field",
        "metadata_file": ROCHE_METADATA_FILE,
        "pose_archive": {
            "name": ROCHE_POSE_ARCHIVE,
            "published_md5": ROCHE_POSE_ARCHIVE_MD5,
            "verification": (
                asdict(archive) if archive is not None else {"verification_skipped": True}
            ),
        },
        "coordinate_frame": {
            "frame_id": ROCHE_COORDINATE_FRAME.frame_id,
            "unit": ROCHE_COORDINATE_FRAME.unit,
            "physical_calibration": None,
        },
        "sampling_rate_hz": None,
        "physical_arena_dimensions": None,
        "mouse_keypoints": list(ROCHE_MOUSE_KEYPOINTS),
        "arena_landmarks": list(ROCHE_ARENA_LANDMARKS),
        "source_keypoint_columns": list(ROCHE_KEYPOINTS),
        "recording_count": len(catalog),
        "selected_recordings": [
            {
                **recording.metadata.to_dict(),
                "resolved_pose_path": str(recording.path),
                "scorer": recording.scorer,
            }
            for recording in selected
        ],
        "interpretation_boundary": (
            "Group and dosage are recorded source metadata, not behavioral causes or "
            "model-internal state."
        ),
    }
