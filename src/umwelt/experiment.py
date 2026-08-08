"""End-to-end execution of the v0.1 recorded-to-synthetic experiment."""

from __future__ import annotations

import hashlib
import json
import platform
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from umwelt import __version__
from umwelt.artifacts import (
    write_artifact_manifest,
    write_json,
    write_report,
    write_synthetic_csv,
)
from umwelt.config import ExperimentConfig
from umwelt.datasets.compass import load_compass_week
from umwelt.errors import ConfigurationError
from umwelt.evaluation import compare_metrics, describe_dataset
from umwelt.models.circadian_markov import (
    fit_phase_conditioned_markov,
    simulate_from_template,
)
from umwelt.observations import ObservationDataset, ObservationSource


@dataclass(frozen=True, slots=True)
class RunResult:
    """Summary returned after a complete experiment is materialized."""

    experiment_id: str
    output_directory: Path
    configuration_sha256: str
    synthetic_series_count: int
    artifact_count: int
    comparison: dict[str, object]


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_revision() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = completed.stdout.strip()
    return revision or None


def _subset(
    dataset: ObservationDataset, subject_ids: tuple[str, ...], suffix: str
) -> ObservationDataset:
    return ObservationDataset(
        dataset_id=f"{dataset.dataset_id}:{suffix}",
        series=tuple(dataset.subject(subject_id) for subject_id in subject_ids),
        provenance={
            **dataset.provenance,
            "selection_role": suffix,
            "selected_subject_ids": list(subject_ids),
        },
    )


def run_experiment(
    config: ExperimentConfig,
    *,
    output_directory: str | Path | None = None,
    verify_integrity: bool = True,
) -> RunResult:
    """Execute v0.1 and atomically publish its complete artifact directory."""

    if output_directory is not None:
        config = replace(config, output_directory=Path(output_directory).resolve())
    destination = config.output_directory
    if destination.exists():
        raise ConfigurationError(
            f"Output directory already exists and will not be overwritten: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)

    selected_subjects = (
        config.dataset.training_subjects + config.dataset.evaluation_subjects
    )
    source_dataset = load_compass_week(
        config.dataset.directory,
        subject_ids=selected_subjects,
        verify_integrity=verify_integrity,
    )
    training = _subset(source_dataset, config.dataset.training_subjects, "training")
    recorded_evaluation = _subset(
        source_dataset, config.dataset.evaluation_subjects, "evaluation"
    )
    model = fit_phase_conditioned_markov(
        training,
        phase_bins=config.model.phase_bins,
        transition_prior=config.model.transition_prior,
        emission_prior=config.model.emission_prior,
    )

    master_generator = random.Random(config.simulation.seed)
    derived_seeds: dict[str, int] = {}
    synthetic_series = []
    for replicate in range(1, config.simulation.replicates + 1):
        for template in recorded_evaluation.series:
            synthetic_id = (
                f"synthetic-r{replicate:03d}-source-{template.subject_id}"
            )
            seed = master_generator.getrandbits(64)
            derived_seeds[synthetic_id] = seed
            synthetic_series.append(
                simulate_from_template(
                    model,
                    template,
                    subject_id=synthetic_id,
                    seed=seed,
                )
            )
    synthetic_dataset = ObservationDataset(
        dataset_id=f"{config.experiment_id}:synthetic",
        series=tuple(synthetic_series),
        provenance={
            "source_category": ObservationSource.SYNTHETIC.value,
            "model_type": model.to_dict()["model_type"],
            "template_dataset_id": recorded_evaluation.dataset_id,
            "template_use": "timestamps and missingness only",
            "master_seed": config.simulation.seed,
            "derived_seeds": derived_seeds,
        },
    )

    metric_arguments = {
        "phase_bins": config.evaluation.phase_bins,
        "autocorrelation_lags": config.evaluation.autocorrelation_lags,
    }
    recorded_metrics = describe_dataset(recorded_evaluation, **metric_arguments)
    synthetic_metrics = describe_dataset(synthetic_dataset, **metric_arguments)
    comparison = compare_metrics(recorded_metrics, synthetic_metrics)
    model_artifact = model.to_dict()
    resolved_config = config.to_dict()
    configuration_sha256 = _canonical_hash(resolved_config)
    provenance = {
        "schema_version": "umwelt.provenance.v1",
        "experiment_id": config.experiment_id,
        "executed_at_utc": datetime.now(UTC).isoformat(),
        "configuration_sha256": configuration_sha256,
        "software": {
            "name": "umwelt",
            "version": __version__,
            "git_revision": _git_revision(),
            "python": sys.version,
            "platform": platform.platform(),
        },
        "recorded_input": recorded_evaluation.provenance,
        "training_input": training.provenance,
        "synthetic_output": synthetic_dataset.provenance,
        "information_categories": {
            "recorded": "COMPASS observations and behaviorally defined labels",
            "model": "fitted phase-conditioned transition and emission parameters",
            "synthetic": "new sequences generated by the explicit model",
        },
    }

    partial = destination.with_name(f".{destination.name}.partial-{uuid4().hex}")
    partial.mkdir()
    try:
        write_json(partial / "resolved-config.json", resolved_config)
        write_json(partial / "provenance.json", provenance)
        write_json(partial / "model.json", model_artifact)
        write_json(partial / "recorded-metrics.json", recorded_metrics)
        write_json(partial / "synthetic-metrics.json", synthetic_metrics)
        write_json(partial / "comparison.json", comparison)
        write_json(partial / "seeds.json", {
            "master_seed": config.simulation.seed,
            "derived_seeds": derived_seeds,
        })
        write_synthetic_csv(
            partial / "synthetic-observations.csv", synthetic_dataset.series
        )
        write_report(
            partial / "report.md",
            config=config,
            model=model_artifact,
            recorded_metrics=recorded_metrics,
            synthetic_metrics=synthetic_metrics,
            comparison=comparison,
            derived_seeds=derived_seeds,
        )
        manifest = write_artifact_manifest(partial)
        partial.replace(destination)
    except Exception:
        if partial.exists():
            shutil.rmtree(partial)
        raise

    return RunResult(
        experiment_id=config.experiment_id,
        output_directory=destination,
        configuration_sha256=configuration_sha256,
        synthetic_series_count=len(synthetic_series),
        artifact_count=len(manifest["artifacts"]) + 1,
        comparison=comparison,
    )
