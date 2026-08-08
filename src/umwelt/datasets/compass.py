"""Adapter for the pinned COMPASS sleep/activity dataset release."""

from __future__ import annotations

import csv
import hashlib
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from umwelt.errors import DataError, DataIntegrityError
from umwelt.observations import (
    BehavioralState,
    ObservationDataset,
    ObservationSource,
    TimeSeries,
)

COMPASS_DATASET_ID = "compass-zenodo-160344-v1"
COMPASS_DOI = "10.5281/zenodo.160344"
COMPASS_RECORD_URL = "https://zenodo.org/records/160344"
COMPASS_ARTICLE_DOI = "10.12688/wellcomeopenres.9892.2"
COMPASS_LICENSE = "CC0-1.0"
COMPASS_SUBJECT_IDS = tuple(str(index) for index in range(1, 25))
CORE_FILE_NAMES = (
    "24mice_activity_LD1week.csv",
    "24mice_sleep_LD1week.csv",
)


@dataclass(frozen=True, slots=True)
class CompassFile:
    """A file pinned to the published Zenodo record."""

    name: str
    size: int
    md5: str
    role: str
    required: bool = False

    @property
    def url(self) -> str:
        """Return the stable Zenodo content endpoint."""

        return f"https://zenodo.org/api/records/160344/files/{self.name}/content"


COMPASS_FILES = (
    CompassFile(
        "1sensorPIRvsEEGdata.csv",
        978_693,
        "42adb8856347f03ab0c45f31684caf26",
        "PIR measurements used in comparison with EEG scoring",
    ),
    CompassFile(
        "1monthPIRsleep.csv",
        13_788_195,
        "25c06c4a518c0a7faaff3d991c66c77c",
        "One month of PIR activity measurements",
    ),
    CompassFile(
        "24mice_activity_LD1week.csv",
        8_707_872,
        "95add81ff28407668d4b8b25cc5a5f13",
        "Seven days of 10-second PIR activity measurements for 24 mice",
        required=True,
    ),
    CompassFile(
        "24mice_sleep_LD1week.csv",
        8_037_598,
        "8b6407ad4306e88463d96256b2302002",
        "Seven days of 10-second behaviorally defined sleep labels for 24 mice",
        required=True,
    ),
    CompassFile(
        "blandAltLandD.csv",
        3_863,
        "bade6a7ef72f1c0113b8792cfab2dd60",
        "Aggregated PIR and EEG sleep estimates",
    ),
    CompassFile(
        "EEG_4mice10sec.csv",
        216_225,
        "173a040be95789b6e7ddaa1343797911",
        "Manual EEG sleep scoring for four mice",
    ),
)


@dataclass(frozen=True, slots=True)
class FileVerification:
    """Integrity status for one published data file."""

    name: str
    path: str
    expected_size: int
    actual_size: int | None
    expected_md5: str
    actual_md5: str | None
    valid: bool


@dataclass(frozen=True, slots=True)
class DownloadResult:
    """Outcome of one requested dataset download."""

    name: str
    path: str
    status: str
    size: int
    md5: str


def compass_files(include_all: bool = False) -> tuple[CompassFile, ...]:
    """Return the complete or v0.1-required pinned file set."""

    if include_all:
        return COMPASS_FILES
    return tuple(file for file in COMPASS_FILES if file.required)


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_compass(
    directory: str | Path, *, include_all: bool = False
) -> tuple[FileVerification, ...]:
    """Verify local COMPASS files against sizes and MD5 hashes from Zenodo."""

    root = Path(directory)
    results: list[FileVerification] = []
    for spec in compass_files(include_all):
        path = root / spec.name
        if not path.is_file():
            results.append(
                FileVerification(
                    name=spec.name,
                    path=str(path),
                    expected_size=spec.size,
                    actual_size=None,
                    expected_md5=spec.md5,
                    actual_md5=None,
                    valid=False,
                )
            )
            continue
        actual_size = path.stat().st_size
        actual_md5 = _md5(path)
        results.append(
            FileVerification(
                name=spec.name,
                path=str(path),
                expected_size=spec.size,
                actual_size=actual_size,
                expected_md5=spec.md5,
                actual_md5=actual_md5,
                valid=actual_size == spec.size and actual_md5 == spec.md5,
            )
        )
    return tuple(results)


def _download_file(spec: CompassFile, target: Path, repair: bool) -> str:
    if target.is_file():
        actual_size = target.stat().st_size
        actual_md5 = _md5(target)
        if actual_size == spec.size and actual_md5 == spec.md5:
            return "present"
        if not repair:
            raise DataIntegrityError(
                f"Existing file does not match the published checksum: {target}. "
                "Use repair mode to replace it."
            )

    partial = target.with_name(f"{target.name}.part")
    offset = partial.stat().st_size if partial.is_file() else 0
    headers = {"User-Agent": "umwelt-dataset-client/0.1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"

    try:
        with urlopen(Request(spec.url, headers=headers), timeout=60) as response:
            append = offset > 0 and response.status == 206
            mode = "ab" if append else "wb"
            with partial.open(mode) as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
    except (HTTPError, URLError, TimeoutError) as error:
        raise DataError(f"Could not download {spec.name}: {error}") from error

    actual_size = partial.stat().st_size
    actual_md5 = _md5(partial)
    if actual_size != spec.size or actual_md5 != spec.md5:
        raise DataIntegrityError(
            f"Downloaded file failed verification: {spec.name} "
            f"(size={actual_size}, md5={actual_md5})."
        )
    os.replace(partial, target)
    return "repaired" if target.exists() and repair else "downloaded"


def download_compass(
    directory: str | Path,
    *,
    include_all: bool = False,
    repair: bool = False,
) -> tuple[DownloadResult, ...]:
    """Download the pinned COMPASS files and verify every completed transfer."""

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    results: list[DownloadResult] = []
    for spec in compass_files(include_all):
        target = root / spec.name
        existed = target.exists()
        status = _download_file(spec, target, repair)
        if status == "repaired" and not existed:
            status = "downloaded"
        results.append(
            DownloadResult(
                name=spec.name,
                path=str(target),
                status=status,
                size=spec.size,
                md5=spec.md5,
            )
        )
    return tuple(results)


def _parse_subject_ids(subject_ids: Iterable[str] | None) -> tuple[str, ...]:
    selected = COMPASS_SUBJECT_IDS if subject_ids is None else tuple(subject_ids)
    if not selected:
        raise DataError("At least one COMPASS subject must be selected.")
    if len(selected) != len(set(selected)):
        raise DataError("COMPASS subject selection contains duplicates.")
    unknown = sorted(set(selected) - set(COMPASS_SUBJECT_IDS), key=int)
    if unknown:
        raise DataError(f"Unknown COMPASS subject identifiers: {', '.join(unknown)}")
    return selected


def _require_valid_core_files(directory: Path) -> tuple[FileVerification, ...]:
    checks = verify_compass(directory)
    invalid = [check.name for check in checks if not check.valid]
    if invalid:
        raise DataIntegrityError(
            "COMPASS source files are missing or invalid: " + ", ".join(invalid)
        )
    return checks


def load_compass_week(
    directory: str | Path,
    *,
    subject_ids: Iterable[str] | None = None,
    verify_integrity: bool = True,
) -> ObservationDataset:
    """Load paired weekly PIR activity and behaviorally defined sleep data."""

    root = Path(directory)
    selected = _parse_subject_ids(subject_ids)
    checks = _require_valid_core_files(root) if verify_integrity else ()
    activity_path = root / CORE_FILE_NAMES[0]
    sleep_path = root / CORE_FILE_NAMES[1]

    timestamps: list[datetime] = []
    states: dict[str, list[BehavioralState | None]] = {
        subject: [] for subject in selected
    }
    activities: dict[str, list[float | None]] = {subject: [] for subject in selected}

    try:
        activity_handle = activity_path.open(newline="", encoding="utf-8-sig")
        sleep_handle = sleep_path.open(newline="", encoding="utf-8-sig")
    except OSError as error:
        raise DataError(f"Could not open COMPASS source files: {error}") from error

    with activity_handle, sleep_handle:
        activity_rows = csv.DictReader(activity_handle)
        sleep_rows = csv.DictReader(sleep_handle)
        required_columns = {"Time", *selected}
        for name, reader in (("activity", activity_rows), ("sleep", sleep_rows)):
            columns = set(reader.fieldnames or ())
            missing_columns = sorted(required_columns - columns)
            if missing_columns:
                raise DataError(
                    f"COMPASS {name} file is missing columns: "
                    + ", ".join(missing_columns)
                )

        for row_number, (activity_row, sleep_row) in enumerate(
            zip(activity_rows, sleep_rows, strict=True), start=2
        ):
            if activity_row["Time"] != sleep_row["Time"]:
                raise DataError(
                    f"COMPASS timestamps diverge at CSV row {row_number}."
                )
            try:
                timestamp = datetime.fromisoformat(activity_row["Time"])
            except ValueError as error:
                raise DataError(
                    f"Invalid COMPASS timestamp at CSV row {row_number}."
                ) from error
            if timestamps and timestamp <= timestamps[-1]:
                raise DataError("COMPASS timestamps are not strictly increasing.")
            timestamps.append(timestamp)

            for subject in selected:
                activity_text = activity_row[subject].strip()
                sleep_text = sleep_row[subject].strip()
                if not activity_text and not sleep_text:
                    states[subject].append(None)
                    activities[subject].append(None)
                    continue
                if not activity_text or not sleep_text:
                    raise DataError(
                        "COMPASS activity and sleep values must be missing together "
                        f"for subject {subject} at CSV row {row_number}."
                    )
                try:
                    activity = float(activity_text)
                    sleep = float(sleep_text)
                except ValueError as error:
                    raise DataError(
                        f"Invalid COMPASS value for subject {subject} "
                        f"at CSV row {row_number}."
                    ) from error
                if sleep not in (0.0, 1.0):
                    raise DataError(
                        f"Invalid binary sleep label for subject {subject} "
                        f"at CSV row {row_number}: {sleep_text}"
                    )
                if activity < 0:
                    raise DataError(
                        f"Negative activity for subject {subject} "
                        f"at CSV row {row_number}."
                    )
                if sleep == 1.0 and activity != 0.0:
                    raise DataError(
                        "COMPASS sleep labels and activity disagree for subject "
                        f"{subject} at CSV row {row_number}."
                    )
                states[subject].append(
                    BehavioralState.SLEEP if sleep == 1.0 else BehavioralState.WAKE
                )
                activities[subject].append(activity)

    shared_timestamps = tuple(timestamps)
    loaded_series = tuple(
        TimeSeries(
            subject_id=subject,
            source=ObservationSource.RECORDED,
            timestamps=shared_timestamps,
            states=tuple(states[subject]),
            activities=tuple(activities[subject]),
        )
        for subject in selected
    )
    provenance_files = (
        [asdict(check) for check in checks]
        if checks
        else [
            {
                "name": name,
                "path": str(root / name),
                "verification_skipped": True,
            }
            for name in CORE_FILE_NAMES
        ]
    )
    return ObservationDataset(
        dataset_id=COMPASS_DATASET_ID,
        series=loaded_series,
        provenance={
            "source_category": ObservationSource.RECORDED.value,
            "record_title": (
                "PIR data and EEG scoring for Wellcome Open Research methods "
                "paper (Brown et al 2016)"
            ),
            "doi": COMPASS_DOI,
            "record_url": COMPASS_RECORD_URL,
            "license": COMPASS_LICENSE,
            "authors": [
                "Laurence A. Brown",
                "Sibah Hasan",
                "Russell G. Foster",
                "Stuart N. Peirson",
            ],
            "related_article_doi": COMPASS_ARTICLE_DOI,
            "adapter": "umwelt.datasets.compass",
            "sleep_label": "behaviorally defined from at least 40 seconds of immobility",
            "experimental_context": {
                "species": "Mus musculus",
                "strain": "C57BL/6J",
                "sex": "male",
                "housing": "individually housed in four groups of six cages",
                "lighting": "12-hour light / 12-hour dark cycle; light begins at time zero",
                "measurement": "passive infrared activity sensing",
                "sampling_interval_seconds": 10,
            },
            "files": provenance_files,
        },
    )
