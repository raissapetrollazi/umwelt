"""Adapter for the pinned Roche open-field pose dataset used by Umwelt v0.2."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path, PurePosixPath
from types import MappingProxyType

from umwelt.errors import DataError, DataIntegrityError
from umwelt.observations import ObservationSource
from umwelt.spatial import (
    AxisOrientation,
    CoordinateFrame,
    Keypoint2D,
    LandmarkSet2D,
    Point2D,
    Pose2D,
    SpatialContext,
    SpatialFrame,
)

ROCHE_OPEN_FIELD_DATASET_ID = "roche-open-field-zenodo-8188683-v1"
ROCHE_OPEN_FIELD_DOI = "10.5281/zenodo.8188683"
ROCHE_OPEN_FIELD_RECORD_URL = "https://zenodo.org/records/8188683"
ROCHE_OPEN_FIELD_LICENSE = "CC-BY-4.0"
ROCHE_METADATA_FILE = "METADATA_ROCHE.csv"
ROCHE_METADATA_SIZE = 6_266
ROCHE_METADATA_MD5 = "096e21e4d319130370aa4bb244b670f8"
ROCHE_POSE_ARCHIVE = "data.zip"
ROCHE_POSE_ARCHIVE_SIZE = 514_028_268
ROCHE_POSE_ARCHIVE_MD5 = "8b48f2060146c03d578d5d6971aa70e9"
ROCHE_POSE_DIRECTORY = PurePosixPath("data/Yohimbine_Roche")
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
ROCHE_EXPECTED_GROUP_COUNTS = {
    ("Control", "0"): 8,
    ("Yohimbine", "1"): 8,
    ("Yohimbine", "3"): 8,
    ("Yohimbine", "6"): 8,
}
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
ROCHE_COORDINATE_FRAME = CoordinateFrame(
    "roche-open-field-image", "px", AxisOrientation.X_RIGHT_Y_DOWN
)


@dataclass(frozen=True, slots=True)
class RocheRecordingMetadata:
    """One source metadata row without interpreting treatment as behavior."""

    animal_id: str
    dlc_file: str
    group: str
    dosage: str
    video: str

    def __post_init__(self) -> None:
        for label, value in (
            ("animal identifier", self.animal_id),
            ("DLC filename", self.dlc_file),
            ("group", self.group),
            ("dosage", self.dosage),
            ("video filename", self.video),
        ):
            if not value:
                raise DataError(f"Roche metadata contains an empty {label}.")

    def to_dict(self) -> dict[str, str]:
        """Return the original metadata categories in a JSON-compatible shape."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class RochePoseFileManifestEntry:
    """Integrity facts derived from one member of the verified pose archive."""

    animal_id: str
    dlc_file: str
    size_bytes: int
    frame_count: int
    sha256: str


def _canonical_dlc_file(animal_id: str) -> str:
    return (
        f"02052023_001_{animal_id}"
        "DLC_resnet50_RocheNetworkOmnitechLMAJan2shuffle1_1030000.csv"
    )


def _pose_manifest_entry(
    animal_id: str, size_bytes: int, frame_count: int, sha256: str
) -> RochePoseFileManifestEntry:
    return RochePoseFileManifestEntry(
        animal_id=animal_id,
        dlc_file=_canonical_dlc_file(animal_id),
        size_bytes=size_bytes,
        frame_count=frame_count,
        sha256=sha256,
    )


# Zenodo publishes an MD5 for data.zip, not hashes for its individual members.
# These SHA-256 values, sizes, and frame counts were derived locally from the
# archive after verifying that published size and MD5.
ROCHE_POSE_MANIFEST: Mapping[str, RochePoseFileManifestEntry] = MappingProxyType(
    {
        entry.animal_id: entry
        for entry in (
            _pose_manifest_entry(
                "16459-67049",
                50_164_222,
                53_954,
                "be45ae2447746ab8aec6c07041cb3ef37cc8d73d6204a265ff36d25f6929e1c4",
            ),
            _pose_manifest_entry(
                "16459-67050",
                50_129_035,
                53_954,
                "42b69780a3252a083ae7de46317417a20943462d99492ff2daafa909680e9c78",
            ),
            _pose_manifest_entry(
                "16459-67051",
                50_291_971,
                53_955,
                "914de7d7dc6f3c24c04027a493502441fac3f5b45daaff77de7edc18c482ba4d",
            ),
            _pose_manifest_entry(
                "16459-67052",
                50_198_521,
                53_955,
                "640a838fe8a490902a70a266951d652660e6be577a20629d93caac55e98c029b",
            ),
            _pose_manifest_entry(
                "16459-67053",
                50_178_663,
                53_914,
                "2b458d2a4c3959e48120d10adaa037902ea13c190b941c5e8f80cfde8ddf2453",
            ),
            _pose_manifest_entry(
                "16459-67054",
                50_169_396,
                53_915,
                "7c886b59f2f2ba251dd35909f7159749095169f6f8eba032b91dacb218fc0cf5",
            ),
            _pose_manifest_entry(
                "16459-67055",
                50_393_902,
                53_915,
                "39ffc07ad8b2b38989364f2e536b78693fe20f8db3ab949004b120444b071153",
            ),
            _pose_manifest_entry(
                "16459-67056",
                50_278_901,
                53_915,
                "8802db4df630b44e3dfd0fd4fadf50cf09f4a6753af6aec55865c206665bdecc",
            ),
            _pose_manifest_entry(
                "16459-67057",
                50_192_929,
                53_964,
                "0fa221aaf47d8fa92170cdb8f34493afdf99ca784c578f9f9db66bb49fc51505",
            ),
            _pose_manifest_entry(
                "16459-67058",
                50_200_706,
                53_962,
                "cfafa48a74582c71042954b17fef8e0964104422d257b869369b911f3091de8a",
            ),
            _pose_manifest_entry(
                "16459-67059",
                50_418_147,
                53_964,
                "34e3ecc85f58c7cd72138f44a4853f8bc78a592e72e833b78f9dc3a15b219ff7",
            ),
            _pose_manifest_entry(
                "16459-67060",
                50_265_985,
                53_963,
                "0b0e7875e209ba71c2bee64aa4655bb194dfba25080471ec70e255c2412f4fd9",
            ),
            _pose_manifest_entry(
                "16459-67061",
                50_199_160,
                53_964,
                "4af3432e8449bec7e4f3e5cf07378100253580fbd91a7637d09f7464f9dc9603",
            ),
            _pose_manifest_entry(
                "16459-67062",
                50_189_634,
                53_962,
                "6b64751bd4045b6792ab89d3842a47de89c59229c6590a69cc6b2db8042d75f7",
            ),
            _pose_manifest_entry(
                "16459-67063",
                50_202_968,
                53_963,
                "c73e91ff6df5aca717f5af1876e8af527f26dc96160ed1576b60de8ff2205d98",
            ),
            _pose_manifest_entry(
                "16459-67064",
                50_215_592,
                53_962,
                "7a2e84d00f2356371d0aa2220fa8dac8852f3cc0ee8dd3e2ca3c92c600e4b5e9",
            ),
            _pose_manifest_entry(
                "16459-67065",
                50_180_820,
                53_964,
                "5b522ceb79155919a196c11224b3093911e8deda1890c8073151b7b038df71b0",
            ),
            _pose_manifest_entry(
                "16459-67066",
                50_259_872,
                53_963,
                "cbbb666406cba10a79213051a1fdd628d496f5a9168244275165543059e6f12e",
            ),
            _pose_manifest_entry(
                "16459-67067",
                50_264_103,
                53_964,
                "48f97efe96bc79a73510a00357fbaebbbbdaeec288393dd031b94388065cc57d",
            ),
            _pose_manifest_entry(
                "16459-67068",
                50_203_956,
                53_963,
                "b2aad10d644a478664ccf72ce8de57247bf6291387137cf38ba31a51c6932d04",
            ),
            _pose_manifest_entry(
                "16459-67069",
                50_296_340,
                53_964,
                "e9129a8780f889e0736d7834646cfb12bd825e67f094d2ad3ab0c1fab0710a3f",
            ),
            _pose_manifest_entry(
                "16459-67070",
                50_226_377,
                53_963,
                "dfaff2087f8aaed46a834498c97b7389748fb1c64f688ece817c83d4cc4a4337",
            ),
            _pose_manifest_entry(
                "16459-67071",
                50_293_658,
                53_964,
                "9ebe7f9f2ef8d32e4b0e9f06f75f92429bfe33a082995946e38a88d50afe5d9e",
            ),
            _pose_manifest_entry(
                "16459-67072",
                50_221_185,
                53_962,
                "910633362dec01302984de59ffe90b6d933357fb229a03bf96f0898264973ecc",
            ),
            _pose_manifest_entry(
                "16459-67073",
                50_180_300,
                53_963,
                "d8e3da48f1f6f2bcc68b83f6eddb60482f3c5ca52b6fb3ed19ca13e7336656e3",
            ),
            _pose_manifest_entry(
                "16459-67074",
                50_240_183,
                53_964,
                "4b7e0db547a5408907fe16ea6b0489f24d89df559f60f88e134993f93a2d0f7f",
            ),
            _pose_manifest_entry(
                "16459-67075",
                50_259_600,
                53_963,
                "614172dba76d34b191b64ce2016226bd4dfabad7db6318d0acb6ace8640643ca",
            ),
            _pose_manifest_entry(
                "16459-67076",
                50_223_767,
                53_964,
                "b0eeb4b7e21739b82b037adebd4415895a61c9d33ebd9870deee755a5ba756be",
            ),
            _pose_manifest_entry(
                "16459-67077",
                50_186_250,
                53_963,
                "274045de5d4404efbf857fcbd1c76ca736dbf3a3cb0495107fb361815d5b18a5",
            ),
            _pose_manifest_entry(
                "16459-67078",
                50_225_479,
                53_964,
                "492d43f5fc53ca50cac67ff920b7f7c6fae49823dd61208c4acdd4612a8ed8ab",
            ),
            _pose_manifest_entry(
                "16459-67079",
                50_248_917,
                53_964,
                "b2d7ead224d34e10c8457833027d67a5b0ab8ca406c0fea36aac426c9b7f3b7e",
            ),
            _pose_manifest_entry(
                "16459-67080",
                50_174_521,
                53_962,
                "67a47b99a3a77ac5de96122131a84f2e8df4f97188bb5ff37da5596eef4bd91d",
            ),
        )
    }
)


@dataclass(frozen=True, slots=True)
class FileVerification:
    """Size and MD5 verification result for one published source file."""

    path: str
    expected_size: int
    actual_size: int | None
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
    def context(self) -> SpatialContext:
        return SpatialContext(
            subject_id=self.subject_id,
            recording_id=self.recording_id,
            source=self.source,
            coordinate_frame=self.coordinate_frame,
        )

    @property
    def sampling_rate_hz(self) -> None:
        """The downloaded source material does not establish a sampling rate."""

        return None

    def frames(self, *, validate_frame_count: bool = True) -> Iterator[SpatialFrame]:
        """Stream frames and validate indices; canonical count is checked on exhaustion."""

        yield from _iter_pose_frames(
            self.path,
            context=self.context,
            validate_frame_count=validate_frame_count,
        )


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file(
    path: Path, *, expected_size: int, expected_md5: str
) -> FileVerification:
    if not path.is_file():
        return FileVerification(
            path=str(path),
            expected_size=expected_size,
            actual_size=None,
            expected_md5=expected_md5,
            actual_md5=None,
            valid=False,
        )
    actual_size = path.stat().st_size
    actual = _md5(path)
    return FileVerification(
        path=str(path),
        expected_size=expected_size,
        actual_size=actual_size,
        expected_md5=expected_md5,
        actual_md5=actual,
        valid=actual_size == expected_size and actual == expected_md5,
    )


def verify_roche_pose_archive(directory: str | Path) -> FileVerification:
    """Verify data.zip against the size and MD5 published by Zenodo."""

    return _verify_file(
        Path(directory) / ROCHE_POSE_ARCHIVE,
        expected_size=ROCHE_POSE_ARCHIVE_SIZE,
        expected_md5=ROCHE_POSE_ARCHIVE_MD5,
    )


def verify_roche_metadata(directory: str | Path) -> FileVerification:
    """Verify METADATA_ROCHE.csv against the published size and MD5."""

    return _verify_file(
        Path(directory) / ROCHE_METADATA_FILE,
        expected_size=ROCHE_METADATA_SIZE,
        expected_md5=ROCHE_METADATA_MD5,
    )


def load_roche_metadata(
    directory: str | Path, *, verify_integrity: bool = True
) -> tuple[RocheRecordingMetadata, ...]:
    """Load and validate the semicolon-delimited UTF-8-BOM metadata table."""

    root = Path(directory)
    path = root / ROCHE_METADATA_FILE
    if verify_integrity:
        verification = verify_roche_metadata(root)
        if not verification.valid:
            raise DataIntegrityError(
                "Roche metadata does not match the size and MD5 published by Zenodo."
            )
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
    videos = [record.video for record in records]
    if len(animal_ids) != len(set(animal_ids)):
        raise DataError("Roche metadata animal identifiers must be unique.")
    if len(dlc_files) != len(set(dlc_files)):
        raise DataError("Roche metadata DLC filenames must be unique.")
    if len(videos) != len(set(videos)):
        raise DataError("Roche metadata video filenames must be unique.")
    if any(
        Path(record.dlc_file).stem != Path(record.video).stem for record in records
    ):
        raise DataError(
            "Roche metadata DLC and video filenames must identify the same recording."
        )

    group_counts = Counter((record.group, record.dosage) for record in records)
    if group_counts != ROCHE_EXPECTED_GROUP_COUNTS:
        raise DataError(
            "Roche metadata does not match the verified Control/0 and "
            "Yohimbine/1/3/6 groups of eight recordings."
        )
    return tuple(records)


def _safe_source_path(text: str) -> PurePosixPath:
    path = PurePosixPath(text.replace("\\", "/"))
    if (
        path.is_absolute()
        or ".." in path.parts
        or len(path.parts) != 1
        or path.suffix.lower() != ".csv"
    ):
        raise DataError(f"Unsafe Roche DLC source path: {text}")
    return path


def resolve_roche_pose_files(
    directory: str | Path,
    metadata: Iterable[RocheRecordingMetadata] | None = None,
) -> dict[str, Path]:
    """Resolve every metadata DLC filename to exactly one extracted CSV file."""

    root = Path(directory)
    records = (
        tuple(metadata)
        if metadata is not None
        else load_roche_metadata(root, verify_integrity=True)
    )
    resolved: dict[str, Path] = {}
    claimed: set[Path] = set()

    for record in records:
        relative = _safe_source_path(record.dlc_file)
        direct = root / relative.name
        extracted = root.joinpath(*ROCHE_POSE_DIRECTORY.parts, relative.name)
        matches = [candidate for candidate in (direct, extracted) if candidate.is_file()]
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


def catalog_roche_open_field(
    directory: str | Path, *, verify_integrity: bool = True
) -> tuple[RocheRecording, ...]:
    """Validate metadata/file correspondence and return lightweight recordings."""

    root = Path(directory)
    records = load_roche_metadata(root, verify_integrity=verify_integrity)
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


def _iter_pose_frames(
    path: Path,
    *,
    context: SpatialContext,
    validate_frame_count: bool,
) -> Iterator[SpatialFrame]:
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
                context=context,
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
    verify_integrity: bool = True,
    verify_archive: bool = False,
) -> dict[str, object]:
    """Return source provenance without materializing pose trajectories."""

    catalog = catalog_roche_open_field(
        directory, verify_integrity=verify_integrity
    )
    selected = catalog if subject_ids is None else select_roche_recordings(catalog, subject_ids)
    archive = verify_roche_pose_archive(directory) if verify_archive else None
    metadata = verify_roche_metadata(directory) if verify_integrity else None
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
        "metadata_file": {
            "name": ROCHE_METADATA_FILE,
            "published_size": ROCHE_METADATA_SIZE,
            "published_md5": ROCHE_METADATA_MD5,
            "verification": (
                asdict(metadata)
                if metadata is not None
                else {"verification_skipped": True}
            ),
        },
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
            "axis_orientation": ROCHE_COORDINATE_FRAME.axis_orientation.value,
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
