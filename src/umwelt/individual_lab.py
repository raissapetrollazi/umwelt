"""Replicated Umwelt v0.1 experiment for training-population individual variation."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import pstdev
from uuid import uuid4

from umwelt.artifacts import write_artifact_manifest, write_json
from umwelt.datasets.compass import load_compass_week
from umwelt.errors import ConfigurationError
from umwelt.individual_reporting import (
    write_individual_variation_report,
    write_individual_variation_svg,
)
from umwelt.individual_variation import (
    POPULATION_MODEL_ID,
    derive_profile_seed,
    fit_population_temporal_model,
    simulate_pooled_from_template,
    simulate_population_from_template,
)
from umwelt.models.temporal_hazard import (
    PHASE_DURATION_MODEL,
    TemporalHazardModel,
    fit_temporal_model_ladder,
)
from umwelt.observations import ObservationDataset, ObservationSource
from umwelt.temporal_config import TemporalLabConfig
from umwelt.temporal_evaluation import (
    RecordedReference,
    build_recorded_reference,
    evaluate_synthetic_replicate,
    predictive_summary,
    summarize_model_replicates,
)
from umwelt.temporal_lab import (
    _activity_emission_artifact,
    _git_dirty,
    _git_revision,
    _subset,
    _write_jsonl,
    canonical_hash,
)

POOLED_MODEL_ID = "phase-duration-hazard-v1:pooled-individuality-control"
BETWEEN_SUBJECT_SCALARS = (
    "sleep_fraction",
    "mean_activity",
    "state_changes_per_hour",
    "wake_bout_median_seconds",
    "sleep_bout_median_seconds",
)


@dataclass(frozen=True, slots=True)
class IndividualLabResult:
    """Compact result returned after an individual-variation run is published."""

    experiment_id: str
    output_directory: Path
    configuration_sha256: str
    model_count: int
    replicate_records: int
    generated_series: int
    artifact_count: int
    comparison: dict[str, object]


def derive_paired_stream_seed(
    master_seed: int, subject_id: str, replicate: int, stream: str
) -> int:
    """Derive a model-independent state or activity seed for paired comparison."""

    if replicate < 1:
        raise ConfigurationError("Replicate numbers begin at one.")
    if stream not in {"state", "activity"}:
        raise ConfigurationError("Paired stream must be 'state' or 'activity'.")
    payload = json.dumps(
        [master_seed, "individual-variation", subject_id, replicate, stream],
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _dispersion(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"standard_deviation": None, "range": None}
    return {
        "standard_deviation": pstdev(values),
        "range": max(values) - min(values),
    }


def recorded_between_subject(reference: RecordedReference) -> dict[str, float | None]:
    """Describe observed between-subject spread without inferential language."""

    metrics: dict[str, float | None] = {}
    for name in BETWEEN_SUBJECT_SCALARS:
        values = [
            float(unit.snapshot["scalars"][name])
            for unit in reference.subjects.values()
            if unit.snapshot["scalars"][name] is not None
        ]
        spread = _dispersion(values)
        metrics[f"{name}_sd"] = spread["standard_deviation"]
        metrics[f"{name}_range"] = spread["range"]
    return metrics


def synthetic_between_subject(
    evaluation: dict[str, object],
) -> dict[str, float | None]:
    """Describe between-subject spread inside one synthetic replicate."""

    subjects = evaluation["subjects"]
    metrics: dict[str, float | None] = {}
    for name in BETWEEN_SUBJECT_SCALARS:
        values = [
            float(scope["metrics"]["scalars"][name])
            for scope in subjects.values()
            if scope["metrics"]["scalars"][name] is not None
        ]
        spread = _dispersion(values)
        metrics[f"{name}_sd"] = spread["standard_deviation"]
        metrics[f"{name}_range"] = spread["range"]
    return metrics


def summarize_between_subject(
    records: list[dict[str, object]],
    reference: dict[str, float | None],
    interval: tuple[float, float],
) -> dict[str, object]:
    """Summarize stochastic between-subject spread across replicate populations."""

    summaries = {}
    for name, recorded_value in reference.items():
        values = [
            float(record["evaluation"]["between_subject"][name])
            for record in records
            if record["evaluation"]["between_subject"][name] is not None
        ]
        summaries[name] = predictive_summary(values, recorded_value, interval)
    return summaries


def _base_phase_duration_model(
    training: ObservationDataset, config: TemporalLabConfig
) -> TemporalHazardModel:
    models = fit_temporal_model_ladder(
        training,
        phase_bins=config.model.phase_bins,
        duration_bin_edges_epochs=config.model.duration_bin_edges_epochs,
        transition_prior=config.model.transition_prior,
        duration_prior_strength=config.model.duration_prior_strength,
        emission_prior=config.model.emission_prior,
    )
    for model in models:
        if model.spec.model_id == PHASE_DURATION_MODEL:
            return model
    raise AssertionError("Temporal model ladder did not contain phase+duration model.")


def _model_summary(
    records: list[dict[str, object]],
    recorded: RecordedReference,
    between_subject_reference: dict[str, float | None],
    interval: tuple[float, float],
) -> dict[str, object]:
    summary = summarize_model_replicates(records, recorded, interval=interval)
    summary["between_subject"] = summarize_between_subject(
        records, between_subject_reference, interval
    )
    return summary


def run_individual_variation_lab(
    config: TemporalLabConfig,
    *,
    output_directory: str | Path,
    verify_integrity: bool = True,
) -> IndividualLabResult:
    """Compare pooled phase+duration dynamics with training-population variation."""

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
    pooled_model = _base_phase_duration_model(training, config)
    population_model = fit_population_temporal_model(training, pooled_model)

    if pooled_model.training_subject_ids != config.dataset.training_subjects:
        raise ConfigurationError(
            "Fitted model subjects diverged from the configured training split."
        )
    if population_model.training_subject_ids != config.dataset.training_subjects:
        raise ConfigurationError(
            "Population profiles diverged from the configured training split."
        )

    recorded = build_recorded_reference(
        development,
        phase_bins=config.evaluation.phase_bins,
        autocorrelation_lags=config.evaluation.autocorrelation_lags,
    )
    between_subject_reference = recorded_between_subject(recorded)
    recorded_artifact = recorded.to_dict()
    recorded_artifact["between_subject"] = between_subject_reference

    model_records: dict[str, list[dict[str, object]]] = {
        POOLED_MODEL_ID: [],
        POPULATION_MODEL_ID: [],
    }
    replicate_records: list[dict[str, object]] = []
    seed_records: list[dict[str, object]] = []
    generated_series = 0

    for replicate in range(1, config.simulation.replicates + 1):
        pooled_series = []
        population_series = []
        pooled_seed_records = []
        population_seed_records = []
        profile_assignments = []

        for template in development.series:
            state_seed = derive_paired_stream_seed(
                config.simulation.seed, template.subject_id, replicate, "state"
            )
            activity_seed = derive_paired_stream_seed(
                config.simulation.seed, template.subject_id, replicate, "activity"
            )
            profile_seed = derive_profile_seed(
                config.simulation.seed, template.subject_id, replicate
            )
            pooled_series.append(
                simulate_pooled_from_template(
                    pooled_model,
                    template,
                    subject_id=template.subject_id,
                    state_seed=state_seed,
                    activity_seed=activity_seed,
                )
            )
            population_generated, profile = simulate_population_from_template(
                population_model,
                template,
                subject_id=template.subject_id,
                state_seed=state_seed,
                activity_seed=activity_seed,
                profile_seed=profile_seed,
            )
            population_series.append(population_generated)
            pooled_seed_records.append(
                {
                    "source_subject_id": template.subject_id,
                    "replicate": replicate,
                    "state_seed": state_seed,
                    "activity_seed": activity_seed,
                    "profile_seed": None,
                    "profile_source_subject_id": None,
                }
            )
            population_seed_records.append(
                {
                    "source_subject_id": template.subject_id,
                    "replicate": replicate,
                    "state_seed": state_seed,
                    "activity_seed": activity_seed,
                    "profile_seed": profile_seed,
                    "profile_source_subject_id": profile.source_subject_id,
                }
            )
            profile_assignments.append(
                {
                    "synthetic_subject_id": template.subject_id,
                    "training_profile_subject_id": profile.source_subject_id,
                }
            )
            generated_series += 2

        for model_id, synthetic_series, seeds, assignments in (
            (POOLED_MODEL_ID, pooled_series, pooled_seed_records, []),
            (
                POPULATION_MODEL_ID,
                population_series,
                population_seed_records,
                profile_assignments,
            ),
        ):
            synthetic = ObservationDataset(
                dataset_id=(
                    f"{config.experiment_id}:individual-variation:{model_id}:"
                    f"r{replicate:03d}"
                ),
                series=tuple(synthetic_series),
                provenance={
                    "source_category": ObservationSource.SYNTHETIC.value,
                    "model_id": model_id,
                    "replicate": replicate,
                    "template_dataset_id": development.dataset_id,
                    "template_use": "timestamps and missingness only",
                },
            )
            evaluation = evaluate_synthetic_replicate(synthetic, recorded)
            evaluation["between_subject"] = synthetic_between_subject(evaluation)
            record = {
                "schema_version": "umwelt.individual-variation-replicate.v1",
                "model_id": model_id,
                "replicate": replicate,
                "paired_state_random_stream": True,
                "seeds": seeds,
                "profile_assignments": assignments,
                "evaluation": evaluation,
            }
            model_records[model_id].append(record)
            replicate_records.append(record)
            seed_records.extend(
                {"model_id": model_id, **seed_record} for seed_record in seeds
            )

    pooled_summary = _model_summary(
        model_records[POOLED_MODEL_ID],
        recorded,
        between_subject_reference,
        config.evaluation.replicate_interval,
    )
    population_summary = _model_summary(
        model_records[POPULATION_MODEL_ID],
        recorded,
        between_subject_reference,
        config.evaluation.replicate_interval,
    )
    comparison = {
        "schema_version": "umwelt.individual-variation-comparison.v1",
        "interpretation": (
            "Descriptive development comparison of pooled state dynamics against a "
            "training-population heterogeneity mechanism; no confirmatory threshold was preregistered."
        ),
        "replicates_per_model": config.simulation.replicates,
        "replicate_interval": list(config.evaluation.replicate_interval),
        "paired_state_random_streams": True,
        "matched_activity_seed_initialization": True,
        "models": [
            {
                "model_id": POOLED_MODEL_ID,
                "label": "Phase + duration (pooled)",
                "hypothesis": (
                    "Control model with one phase+duration state-dynamics parameterization "
                    "shared by all synthetic individuals."
                ),
                "population_variation": False,
                "parameter_count": pooled_model.parameter_count,
                "summary": pooled_summary,
            },
            {
                "model_id": POPULATION_MODEL_ID,
                "label": population_model.label,
                "hypothesis": population_model.hypothesis,
                "population_variation": True,
                "parameter_count": population_model.parameter_count,
                "summary": population_summary,
            },
        ],
    }

    resolved = {
        "schema_version": "umwelt.individual-variation-run.v1",
        "temporal_lab_configuration": config.to_dict(),
        "analysis": {
            "base_model_id": PHASE_DURATION_MODEL,
            "population_model_id": POPULATION_MODEL_ID,
            "population_profile_source": "training subjects only",
            "profile_sampling": "uniform empirical resampling with replacement",
            "paired_state_random_streams": True,
            "separate_activity_random_stream": True,
        },
    }
    configuration_sha256 = canonical_hash(resolved)
    provenance = {
        "schema_version": "umwelt.individual-variation-provenance.v1",
        "experiment_id": f"{config.experiment_id}-individual-variation",
        "configuration_sha256": configuration_sha256,
        "configuration_source": (
            str(config.source_path) if config.source_path is not None else None
        ),
        "software": {
            "git_revision": _git_revision(),
            "git_tracked_worktree_dirty": _git_dirty(),
        },
        "recorded_input": source_dataset.provenance,
        "protocol": {
            "training_subjects": list(config.dataset.training_subjects),
            "held_out_development_subjects": list(config.dataset.development_subjects),
            "development_exposure_caveat": (
                "Results for these development subjects were previously inspected; "
                "this experiment is model-development evidence, not a pristine confirmatory test."
            ),
            "subject_specific_development_calibration": False,
            "population_profiles_fit_from": list(config.dataset.training_subjects),
            "population_profile_dimensions": [
                "sleep occupancy logit offset",
                "wake-leaving hazard logit offset",
                "sleep-leaving hazard logit offset",
            ],
            "random_stream_design": (
                "State and activity randomness use separate generators. Pooled and population "
                "variants share the same state seed for a subject/replicate, so the state "
                "generator consumes one paired draw per available epoch in both variants. "
                "Activity seeds also match, but activity-generator consumption may diverge "
                "after state trajectories diverge."
            ),
        },
        "synthetic_output": {
            "source_category": ObservationSource.SYNTHETIC.value,
            "master_seed": config.simulation.seed,
            "replicates_per_model": config.simulation.replicates,
            "generated_series": generated_series,
            "paired_state_random_streams": True,
            "activity_randomness_isolated_from_state": True,
            "profile_seed_derivation": (
                "SHA-256 over master seed, individual-profile role, source subject, and replicate"
            ),
            "template_use": "timestamps and missingness only",
            "full_trajectories_retained": False,
        },
        "information_categories": {
            "recorded_observations": (
                "COMPASS PIR activity and behaviorally defined immobility labels"
            ),
            "derived_population_profiles": (
                "three smoothed training-subject logit offsets relative to pooled training dynamics"
            ),
            "synthetic_individual_profile": (
                "one empirically resampled training-derived computational profile per generated subject"
            ),
            "synthetic_internal_state": (
                "explicit wake/sleep model state, not real-animal physiology, personality, or mental state"
            ),
        },
    }
    model_artifact = {
        "schema_version": "umwelt.individual-variation-models.v1",
        "shared_activity_emission": _activity_emission_artifact(pooled_model),
        "pooled_state_model": pooled_model.to_dict(),
        "population_state_model": population_model.to_dict(),
    }

    partial = destination.with_name(f".{destination.name}.partial-{uuid4().hex}")
    partial.mkdir()
    try:
        write_json(partial / "resolved-config.json", resolved)
        write_json(partial / "provenance.json", provenance)
        write_json(partial / "models.json", model_artifact)
        write_json(partial / "recorded-reference.json", recorded_artifact)
        _write_jsonl(partial / "replicate-metrics.jsonl", replicate_records)
        write_json(partial / "model-comparison.json", comparison)
        write_json(
            partial / "seeds.json",
            {
                "schema_version": "umwelt.individual-variation-seeds.v1",
                "master_seed": config.simulation.seed,
                "paired_state_random_streams": True,
                "separate_activity_random_stream": True,
                "series": seed_records,
            },
        )
        write_individual_variation_report(
            partial / "report.md",
            config=config,
            provenance=provenance,
            recorded=recorded_artifact,
            comparison=comparison,
        )
        write_individual_variation_svg(
            partial / "individual-variation.svg",
            config=config,
            recorded=recorded_artifact,
            comparison=comparison,
        )
        manifest = write_artifact_manifest(partial)
        partial.replace(destination)
    except Exception:
        if partial.exists():
            shutil.rmtree(partial)
        raise

    return IndividualLabResult(
        experiment_id=f"{config.experiment_id}-individual-variation",
        output_directory=destination,
        configuration_sha256=configuration_sha256,
        model_count=2,
        replicate_records=len(replicate_records),
        generated_series=generated_series,
        artifact_count=len(manifest["artifacts"]) + 1,
        comparison=comparison,
    )
