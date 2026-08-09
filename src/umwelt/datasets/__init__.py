"""Recorded dataset adapters."""

from umwelt.datasets.compass import (
    COMPASS_DATASET_ID,
    COMPASS_DOI,
    download_compass,
    load_compass_week,
    verify_compass,
)
from umwelt.datasets.roche_open_field import (
    ROCHE_KEYPOINTS,
    ROCHE_OPEN_FIELD_DATASET_ID,
    ROCHE_OPEN_FIELD_DOI,
    RocheRecording,
    RocheRecordingMetadata,
    catalog_roche_open_field,
    load_roche_metadata,
    roche_provenance,
    select_roche_recordings,
    verify_roche_pose_archive,
)

__all__ = [
    "COMPASS_DATASET_ID",
    "COMPASS_DOI",
    "ROCHE_KEYPOINTS",
    "ROCHE_OPEN_FIELD_DATASET_ID",
    "ROCHE_OPEN_FIELD_DOI",
    "RocheRecording",
    "RocheRecordingMetadata",
    "catalog_roche_open_field",
    "download_compass",
    "load_compass_week",
    "load_roche_metadata",
    "roche_provenance",
    "select_roche_recordings",
    "verify_compass",
    "verify_roche_pose_archive",
]
