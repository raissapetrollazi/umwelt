"""Reproducible recorded-to-synthetic spatial experiment for Umwelt v0.2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from umwelt.datasets.roche_open_field import catalog_roche_open_field
from umwelt.errors import DataError
from umwelt.spatial_config import SpatialLabConfig
from umwelt.spatial_execution import execute_spatial_lab
from umwelt.spatial_protocol import (
    ROCHE_CONTROL_DEVELOPMENT_SUBJECTS,
    ROCHE_CONTROL_FITTING_SUBJECTS,
)


@dataclass(frozen=True, slots=True)
class SpatialLabResult:
    experiment_id: str
    output_directory: Path
    configuration_sha256: str
    replicate_records: int
    generated_series: int
    artifact_count: int


def _recording_map(directory: Path):
    recordings = catalog_roche_open_field(directory)
    by_subject = {recording.subject_id: recording for recording in recordings}
    selected = ROCHE_CONTROL_FITTING_SUBJECTS + ROCHE_CONTROL_DEVELOPMENT_SUBJECTS
    for subject_id in selected:
        if subject_id not in by_subject:
            raise DataError(f"Pinned Roche control animal is missing: {subject_id}")
        metadata = by_subject[subject_id].metadata
        if metadata.group != "Control" or metadata.dosage != "0":
            raise DataError("Pinned v0.2 animals must remain Control dose 0.")
    return by_subject


def run_spatial_lab(
    config: SpatialLabConfig, *, output_directory: str | Path | None = None
) -> SpatialLabResult:
    """Fit on pinned controls and compare seeded replicas on development animals."""

    target = Path(output_directory or config.output_directory).resolve()
    if target.exists():
        raise DataError(f"Spatial output directory already exists: {target}")
    recordings = _recording_map(config.dataset_directory)
    config_hash, records, generated, artifacts = execute_spatial_lab(
        config, recordings, target
    )
    return SpatialLabResult(
        experiment_id=config.experiment_id,
        output_directory=target,
        configuration_sha256=config_hash,
        replicate_records=records,
        generated_series=generated,
        artifact_count=artifacts,
    )
