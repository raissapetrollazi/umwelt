"""End-to-end execution of replicated Umwelt v0.1 temporal comparisons."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from uuid import uuid4

from umwelt import __version__
from umwelt.artifacts import write_artifact_manifest, write_json
from umwelt.datasets.compass import load_compass_week
from umwelt.errors import ConfigurationError
from umwelt.models.temporal_hazard import (
    TemporalHazardModel,
    fit_temporal_model_ladder,
    simulate_temporal_from_template,
)
from umwelt.observations import ObservationDataset, ObservationSource
from umwelt.temporal_config import TemporalLabConfig
from umwelt.temporal_evaluation import (
    build_recorded_reference,
    evaluate_synthetic_replicate,
    summarize_model_replicates,
)
from umwelt.temporal_reporting import (
    write_autocorrelation_svg,
    write_model_comparison_svg,
    write_temporal_report,
)


@dataclass(frozen=True, slots=True)
class TemporalLabResult:
    """Compact result returned after a temporal-lab run is published."""

    experiment_id: str
    output_directory: Path
    configuration_sha256: str
    model_count: int
    replicate_records: int
    generated_series: int
    artifact_count: int
    comparison: dict[str, object]


def canonical_hash(value: object) -> str:
    """Hash one JSON-compatible value using Umwelt's canonical encoding."""

    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def derive_replicate_seed(
    master_seed: int, model_id: str, subject_id: str, replicate: int
) -> int:
    """Derive a stable 64-bit seed from all synthetic-series identifiers."""

    if replicate < 1:
        raise ConfigurationError("Replicate numbers begin at one.")
    payload = json.dumps(
        [master_seed, model_id, subject_id, replicate],
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _git_revision() -> str | None:
    repository = Path(__file__).resolve().parents[2]
    if not (repository / ".git").exists():
        return None
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
            cwd=repository,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _git_dirty() -> bool | None:
    repository = Path(__file__).resolve().parents[2]
    if not (repository / ".git").exists():
        return None
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
            cwd=repository,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(completed.stdout.strip())


def _subset(
    dataset: ObservationDataset, subject_ids: tuple[str, ...], role: str
) -> ObservationDataset:
    return ObservationDataset(
        dataset_id=f"{dataset.dataset_id}:{role}",
        series=tuple(dataset.subject(subject_id) for subject_id in subject_ids),
        provenance={
            **dataset.provenance,
            "selection_role": role,
            "selected_subject_ids": list(subject_ids),
        },
    )


def _activity_emission_artifact(model: TemporalHazardModel) -> dict[str, object]:
    activity = model.activity_model
    return {
        "schema_version": "umwelt.activity-emission.v1",
        "emission_id": "phase-conditioned-zero-inflated-gamma-v1",
        "phase_bins": activity.phase_bins,
        "epoch_seconds": activity.epoch_seconds,
        "emission_prior": activity.emission_prior,
        "training_subject_ids": list(activity.training_subject_ids),
        "training_observations": activity.training_observations,
        "phases": [
            {
                "phase": phase.phase,
                "wake_activity": asdict(phase.wake_activity),
                "sleep_activity": asdict(phase.sleep_activity),
            }
            for phase in activity.phases
        ],
        "assumptions": [
            "Activity depends on explicit synthetic state and daily phase.",
            "Zero probability and positive gamma moments are pooled from training subjects.",
            "Positive draws are capped at the corresponding observed training maximum.",
            "The same fitted emission object is used by every temporal state model.",
        ],
    }


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record, sort_keys=True, separators=(",", ":"), ensure_ascii=True
                )
                + "\n"
            )


def run_temporal_lab(
    config: TemporalLabConfig,
    *,
    output_directory: str | Path | None = None,
    verify_integrity: bool = True,
) -> TemporalLabResult:
    """Fit, replicate, evaluate, and atomically publish the temporal model ladder."""

    if output_directory is not None:
        config = replace(config, output_directory=Path(output_directory).resolve())
    destination = config.output_directory
    if destination.exists():
        raise ConfigurationError(
            f"Output directory already exists and will not be overwritten: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)

    selected_subjects = (
        config.dataset.training_subjects + config.dataset.development_subjects
    )
    source_dataset = load_compass_week(
        config.dataset.directory,
        subject_ids=selected_subjects,
        verify_integrity=verify_integrity,
    )
    training = _subset(source_dataset, config.dataset.training_subjects, "training")
    development = _subset(
        source_dataset,
        config.dataset.development_subjects,
        "held-out-development",
    )
    models = fit_temporal_model_ladder(
        training,
        phase_bins=config.model.phase_bins,
        duration_bin_edges_epochs=config.model.duration_bin_edges_epochs,
        transition_prior=config.model.transition_prior,
        duration_prior_strength=config.model.duration_prior_strength,
        emission_prior=config.model.emission_prior,
    )
    if any(
        model.training_subject_ids != config.dataset.training_subjects
        for model in models
    ):
        raise ConfigurationError(
            "Fitted model subjects diverged from the training split."
        )
    if any(model.activity_model is not models[0].activity_model for model in models):
        raise AssertionError("Temporal ladder models must share one activity emission.")

    recorded = build_recorded_reference(
        development,
        phase_bins=config.evaluation.phase_bins,
        autocorrelation_lags=config.evaluation.autocorrelation_lags,
    )
    replicate_records: list[dict[str, object]] = []
    seed_records: list[dict[str, object]] = []
    comparison_models = []

    for ladder_index, model in enumerate(models, start=1):
        model_records = []
        for replicate in range(1, config.simulation.replicates + 1):
            synthetic_series = []
            replicate_seeds = []
            for template in development.series:
                seed = derive_replicate_seed(
                    config.simulation.seed,
                    model.spec.model_id,
                    template.subject_id,
                    replicate,
                )
                seed_record = {
                    "model_id": model.spec.model_id,
                    "source_subject_id": template.subject_id,
                    "replicate": replicate,
                    "derived_seed": seed,
                }
                replicate_seeds.append(seed_record)
                seed_records.append(seed_record)
                synthetic_series.append(
                    simulate_temporal_from_template(
                        model,
                        template,
                        subject_id=template.subject_id,
                        seed=seed,
                    )
                )
            synthetic = ObservationDataset(
                dataset_id=(
                    f"{config.experiment_id}:synthetic:{model.spec.model_id}:"
                    f"r{replicate:03d}"
                ),
                series=tuple(synthetic_series),
                provenance={
                    "source_category": ObservationSource.SYNTHETIC.value,
                    "model_id": model.spec.model_id,
                    "replicate": replicate,
                    "template_dataset_id": development.dataset_id,
                    "template_use": "timestamps and missingness only",
                },
            )
            record = {
                "schema_version": "umwelt.temporal-replicate-metrics.v1",
                "model_id": model.spec.model_id,
                "replicate": replicate,
                "seeds": replicate_seeds,
                "evaluation": evaluate_synthetic_replicate(synthetic, recorded),
            }
            replicate_records.append(record)
            model_records.append(record)
            del synthetic_series, synthetic

        comparison_models.append(
            {
                "ladder_index": ladder_index,
                "model_id": model.spec.model_id,
                "label": model.spec.label,
                "hypothesis": model.spec.hypothesis,
                "mechanisms": {
                    "phase_conditioning": model.spec.use_phase,
                    "duration_conditioning": model.spec.use_duration,
                },
                "parameter_count": model.parameter_count,
                "summary": summarize_model_replicates(
                    model_records,
                    recorded,
                    interval=config.evaluation.replicate_interval,
                ),
            }
        )

    comparison = {
        "schema_version": "umwelt.temporal-model-comparison.v1",
        "interpretation": (
            "Descriptive model-development comparison; no acceptance threshold or "
            "confirmatory test was preregistered."
        ),
        "replicates_per_model": config.simulation.replicates,
        "replicate_interval": list(config.evaluation.replicate_interval),
        "models": comparison_models,
    }
    resolved_config = config.to_dict()
    configuration_sha256 = canonical_hash(resolved_config)
    model_artifact = {
        "schema_version": "umwelt.temporal-model-ladder.v1",
        "shared_activity_emission": _activity_emission_artifact(models[0]),
        "state_models": [model.to_dict() for model in models],
    }
    provenance = {
        "schema_version": "umwelt.temporal-provenance.v1",
        "experiment_id": config.experiment_id,
        "configuration_sha256": configuration_sha256,
        "configuration_source": (
            str(config.source_path) if config.source_path is not None else None
        ),
        "software": {
            "name": "umwelt",
            "version": __version__,
            "git_revision": _git_revision(),
            "git_tracked_worktree_dirty": _git_dirty(),
            "python": sys.version,
            "platform": platform.platform(),
        },
        "recorded_input": source_dataset.provenance,
        "protocol": {
            "training_subjects": list(config.dataset.training_subjects),
            "held_out_development_subjects": list(config.dataset.development_subjects),
            "development_exposure_caveat": (
                "The original baseline results for these subjects were inspected before "
                "the temporal ladder was specified; this is not a pristine confirmatory test."
            ),
            "subject_specific_calibration": False,
        },
        "synthetic_output": {
            "source_category": ObservationSource.SYNTHETIC.value,
            "model_ids": [model.spec.model_id for model in models],
            "master_seed": config.simulation.seed,
            "seed_derivation": (
                "SHA-256 JSON array [master_seed, model_id, source_subject_id, "
                "replicate], first 64 bits interpreted as an unsigned integer"
            ),
            "replicates_per_model": config.simulation.replicates,
            "generated_series": len(seed_records),
            "template_use": "timestamps and missingness only",
            "full_trajectories_retained": False,
        },
        "information_categories": {
            "recorded_observations": (
                "COMPASS PIR activity and behaviorally defined immobility labels"
            ),
            "derived_recorded_metrics": (
                "descriptive aggregate and subject-level summaries"
            ),
            "fitted_model_parameters": (
                "state hazards and shared activity-emission parameters fitted on training subjects"
            ),
            "synthetic_observations": (
                "ephemeral generated sequences summarized into replicate metrics"
            ),
            "synthetic_internal_state": (
                "explicit wake/sleep model state, not real-animal physiology or mental state"
            ),
        },
    }

    partial = destination.with_name(f".{destination.name}.partial-{uuid4().hex}")
    partial.mkdir()
    try:
        write_json(partial / "resolved-config.json", resolved_config)
        write_json(partial / "provenance.json", provenance)
        write_json(partial / "models.json", model_artifact)
        write_json(partial / "recorded-reference.json", recorded.to_dict())
        _write_jsonl(partial / "replicate-metrics.jsonl", replicate_records)
        write_json(partial / "model-comparison.json", comparison)
        write_json(
            partial / "seeds.json",
            {
                "schema_version": "umwelt.temporal-seeds.v1",
                "master_seed": config.simulation.seed,
                "derivation": provenance["synthetic_output"]["seed_derivation"],
                "series": seed_records,
            },
        )
        write_temporal_report(
            partial / "report.md",
            config=config,
            provenance=provenance,
            recorded=recorded.to_dict(),
            comparison=comparison,
        )
        write_model_comparison_svg(partial / "model-comparison.svg", comparison)
        write_autocorrelation_svg(
            partial / "autocorrelation.svg",
            comparison=comparison,
            recorded=recorded.to_dict(),
            lags=config.evaluation.autocorrelation_lags,
        )
        manifest = write_artifact_manifest(partial)
        partial.replace(destination)
    except Exception:
        if partial.exists():
            shutil.rmtree(partial)
        raise

    return TemporalLabResult(
        experiment_id=config.experiment_id,
        output_directory=destination,
        configuration_sha256=configuration_sha256,
        model_count=len(models),
        replicate_records=len(replicate_records),
        generated_series=len(seed_records),
        artifact_count=len(manifest["artifacts"]) + 1,
        comparison=comparison,
    )
