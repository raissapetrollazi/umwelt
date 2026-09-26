"""Reproducible, atomic v0.3 behavioral-world laboratory workflow."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

from umwelt import __version__
from umwelt.datasets.roche_open_field import (
    ROCHE_POSE_ARCHIVE,
    catalog_roche_open_field,
    roche_provenance,
)
from umwelt.errors import DataError
from umwelt.spatial_evaluation import ROCHE_SPATIAL_EVALUATION_PROTOCOL
from umwelt.spatial_protocol import (
    ROCHE_CONTROL_DEVELOPMENT_SUBJECTS,
    ROCHE_CONTROL_FITTING_SUBJECTS,
    ROCHE_RECORDED_TRAJECTORY_PROTOCOL,
)
from umwelt.spatial_reporting import file_sha256, write_json
from umwelt.world import V03_ARENA_SCALE_CONDITIONS
from umwelt.world_config import WorldLabConfig
from umwelt.world_execution import execute_world_experiment
from umwelt.world_model import BOUNDARY_CONDITIONED_MODEL_ID
from umwelt.world_protocol import WorldExperimentProtocol
from umwelt.world_recordings import load_recorded_world_series
from umwelt.world_reporting import (
    build_intervention_comparison,
    build_model_comparison,
    render_world_report,
)


@dataclass(frozen=True, slots=True)
class WorldLabResult:
    experiment_id: str
    output_directory: Path
    configuration_sha256: str
    replicate_records: int
    generated_series: int
    artifact_count: int


def _software_provenance() -> dict[str, object]:
    repository = Path(__file__).resolve().parents[2]
    revision: str | None = None
    dirty: bool | None = None
    if (repository / ".git").exists():
        try:
            revision_result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=3,
                cwd=repository,
            )
            status_result = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                check=True,
                capture_output=True,
                text=True,
                timeout=3,
                cwd=repository,
            )
            revision = revision_result.stdout.strip() or None
            dirty = bool(status_result.stdout.strip())
        except (OSError, subprocess.SubprocessError):
            pass
    return {
        "name": "umwelt",
        "version": __version__,
        "git_revision": revision,
        "git_tracked_worktree_dirty": dirty,
    }


def _write_artifact_manifest(directory: Path) -> int:
    artifacts: list[dict[str, object]] = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.name != "artifact-manifest.json":
            artifacts.append(
                {
                    "name": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    write_json(
        directory / "artifact-manifest.json",
        {
            "schema_version": "umwelt.artifacts.v1",
            "algorithm": "sha256",
            "artifacts": artifacts,
        },
    )
    return len(artifacts) + 1


def run_world_lab(
    config: WorldLabConfig, *, output_directory: str | Path | None = None
) -> WorldLabResult:
    """Validate pinned inputs, execute the frozen protocol, and publish once."""

    target = Path(output_directory or config.output_directory).resolve()
    if target.exists():
        raise DataError(f"World output directory already exists: {target}")
    resolved_config = replace(config, output_directory=target)
    protocol = WorldExperimentProtocol(config.seed, config.replicates)
    selected = ROCHE_CONTROL_FITTING_SUBJECTS + ROCHE_CONTROL_DEVELOPMENT_SUBJECTS
    catalog = catalog_roche_open_field(config.dataset_directory)
    by_subject = {recording.subject_id: recording for recording in catalog}
    if any(subject_id not in by_subject for subject_id in selected):
        raise DataError("A pinned Roche control animal is missing from the source.")
    series = {
        subject_id: load_recorded_world_series(by_subject[subject_id])
        for subject_id in selected
    }
    source_provenance = roche_provenance(
        config.dataset_directory,
        subject_ids=selected,
        verify_archive=(config.dataset_directory / ROCHE_POSE_ARCHIVE).is_file(),
    )
    run = execute_world_experiment(protocol, series)
    model_comparison = build_model_comparison(protocol, run)
    intervention_comparison = build_intervention_comparison(
        protocol, run.intervention_outcomes
    )
    config_payload = json.dumps(
        resolved_config.to_dict(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    config_hash = hashlib.sha256(config_payload).hexdigest()

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise DataError(f"World output directory already exists: {target}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        write_json(temporary / "resolved-config.json", resolved_config.to_dict())
        write_json(
            temporary / "protocol.json",
            {
                "trajectory": ROCHE_RECORDED_TRAJECTORY_PROTOCOL.to_dict(),
                "spatial_evaluation": ROCHE_SPATIAL_EVALUATION_PROTOCOL.to_dict(),
                "world_experiment": protocol.to_dict(),
                "pre_result_source_quality_amendment": (
                    "retain recorded outside positions in spatial metrics; "
                    "exclude unclassified starts only from zone fitting"
                ),
            },
        )
        write_json(temporary / "world.json", run.worlds)
        write_json(temporary / "models.json", run.models)
        write_json(
            temporary / "interventions.json",
            {
                "model_id": BOUNDARY_CONDITIONED_MODEL_ID,
                "template_subject_ids": list(ROCHE_CONTROL_FITTING_SUBJECTS),
                "conditions": [
                    condition.to_dict() for condition in V03_ARENA_SCALE_CONDITIONS
                ],
                "paired_seed": "subject-and-replicate-only",
                "recorded_intervention_counterpart": False,
                "reflection_count_scope": (
                    "full-generated-trajectory-before-observation-mask"
                ),
            },
        )
        write_json(
            temporary / "seeds.json",
            {
                "master_seed": config.seed,
                "derived": [
                    {
                        "subject_id": subject_id,
                        "replicate": replicate,
                        "seed": protocol.seed(subject_id, replicate),
                    }
                    for subject_id in selected
                    for replicate in range(protocol.replicates)
                ],
                "shared_across_models_and_conditions": True,
            },
        )
        with (temporary / "replicate-metrics.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as handle:
            for record in run.records:
                handle.write(
                    json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n"
                )
        write_json(temporary / "model-comparison.json", model_comparison)
        write_json(temporary / "intervention-comparison.json", intervention_comparison)
        write_json(
            temporary / "provenance.json",
            {
                "dataset": source_provenance,
                "configuration_sha256": config_hash,
                "software": _software_provenance(),
                "information_categories": [
                    "recorded",
                    "derived",
                    "model",
                    "intervention-input",
                    "synthetic",
                ],
                "cross_validation_evaluation_subject_used_for_fitting": False,
                "legacy_diagnostic_used_for_model_selection": False,
                "recorded_intervention_counterpart": False,
                "synthetic_trajectories_retained": False,
            },
        )
        (temporary / "report.md").write_text(
            render_world_report(
                config=resolved_config,
                run=run,
                intervention_comparison=intervention_comparison,
            ),
            encoding="utf-8",
            newline="\n",
        )
        artifact_count = _write_artifact_manifest(temporary)
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return WorldLabResult(
        experiment_id=config.experiment_id,
        output_directory=target,
        configuration_sha256=config_hash,
        replicate_records=len(run.records),
        generated_series=run.generated_series_count,
        artifact_count=artifact_count,
    )
