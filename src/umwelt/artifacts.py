"""Writers for reproducible, source-labeled experiment artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

from umwelt.config import ExperimentConfig
from umwelt.observations import TimeSeries


def write_json(path: Path, value: object) -> None:
    """Write stable, human-readable JSON."""

    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for one artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_synthetic_csv(path: Path, series_collection: Iterable[TimeSeries]) -> None:
    """Write synthetic observations with source and missingness made explicit."""

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ["timestamp", "subject_id", "source", "available", "state", "activity"]
        )
        for series in series_collection:
            for timestamp, state, activity in zip(
                series.timestamps, series.states, series.activities, strict=True
            ):
                writer.writerow(
                    [
                        timestamp.isoformat(sep=" "),
                        series.subject_id,
                        series.source.value,
                        "true" if state is not None else "false",
                        state.value if state is not None else "",
                        f"{activity:.6f}" if activity is not None else "",
                    ]
                )


def _format_number(value: object, digits: int = 4) -> str:
    if value is None:
        return "not available"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def write_report(
    path: Path,
    *,
    config: ExperimentConfig,
    model: Mapping[str, object],
    recorded_metrics: Mapping[str, object],
    synthetic_metrics: Mapping[str, object],
    comparison: Mapping[str, object],
    derived_seeds: Mapping[str, int],
) -> None:
    """Write a restrained automatic report of results and limitations."""

    recorded_occupancy = recorded_metrics["state_occupancy"]
    synthetic_occupancy = synthetic_metrics["state_occupancy"]
    recorded_activity = recorded_metrics["activity"]
    synthetic_activity = synthetic_metrics["activity"]
    recorded_transitions = recorded_metrics["transitions"]
    synthetic_transitions = synthetic_metrics["transitions"]

    lines = [
        f"# Experiment report: {config.experiment_id}",
        "",
        "## Interpretation boundary",
        "",
        (
            "This report describes a computational comparison. Agreement does not "
            "establish that the model reproduces the biological mechanism that "
            "generated the recorded observations. Synthetic states are model states, "
            "not evidence about a real animal's subjective state."
        ),
        "",
        "## Recorded source",
        "",
        "- Dataset: COMPASS weekly PIR activity and behaviorally defined sleep data",
        "- DOI: 10.5281/zenodo.160344",
        "- License: CC0-1.0",
        "- Training subjects: " + ", ".join(config.dataset.training_subjects),
        "- Held-out evaluation subjects: "
        + ", ".join(config.dataset.evaluation_subjects),
        "- Sleep labels: based on at least 40 seconds of immobility, not EEG labels",
        "",
        "## Model",
        "",
        f"- Type: `{model['model_type']}`",
        f"- Daily phase bins: {model['phase_bins']}",
        f"- Observation interval: {model['epoch_seconds']} seconds",
        f"- Master seed: {config.simulation.seed}",
        f"- Synthetic replicates: {config.simulation.replicates}",
        "",
        "Model assumptions:",
        "",
    ]
    lines.extend(f"- {assumption}" for assumption in model["assumptions"])
    lines.extend(
        [
            "",
            "## Aggregate comparison",
            "",
            "| Metric | Recorded | Synthetic |",
            "| --- | ---: | ---: |",
            (
                "| Sleep fraction | "
                f"{_format_number(recorded_occupancy['sleep_fraction'])} | "
                f"{_format_number(synthetic_occupancy['sleep_fraction'])} |"
            ),
            (
                "| Mean activity | "
                f"{_format_number(recorded_activity['all_epochs']['mean'])} | "
                f"{_format_number(synthetic_activity['all_epochs']['mean'])} |"
            ),
            (
                "| Nonzero activity fraction | "
                f"{_format_number(recorded_activity['nonzero_fraction'])} | "
                f"{_format_number(synthetic_activity['nonzero_fraction'])} |"
            ),
            (
                "| State changes per hour | "
                f"{_format_number(recorded_transitions['state_changes_per_hour'])} | "
                f"{_format_number(synthetic_transitions['state_changes_per_hour'])} |"
            ),
            "",
            "Circadian-profile discrepancies:",
            "",
            (
                "- Sleep-fraction RMSE: "
                + _format_number(comparison["phase_profile"]["sleep_fraction_rmse"])
            ),
            (
                "- Mean-activity RMSE: "
                + _format_number(comparison["phase_profile"]["mean_activity_rmse"])
            ),
            "",
            "## Largest descriptive mismatches",
            "",
            (
                "No pass/fail threshold was preregistered. The following ranking uses "
                "relative difference only to expose where this baseline diverges most."
            ),
            "",
        ]
    )
    mismatches = comparison["largest_relative_discrepancies"]
    if mismatches:
        for mismatch in mismatches:
            lines.append(
                f"- `{mismatch['metric']}`: recorded "
                f"{_format_number(mismatch['recorded'])}, synthetic "
                f"{_format_number(mismatch['synthetic'])}, relative difference "
                f"{_format_number(mismatch['relative_difference'])}"
            )
    else:
        lines.append("- No rankable relative differences were available.")
    lines.extend(
        [
            "",
            "## Known limitations",
            "",
            "- The weekly sleep labels are behaviorally defined from PIR immobility.",
            "- The model pools training subjects and does not represent individual effects.",
            "- First-order state dependence cannot generally reproduce full bout-duration structure.",
            "- Daily phase is represented as discrete bins rather than a continuous rhythm.",
            "- Activity emissions are descriptive and conditionally independent across epochs.",
            "- EEG files in the COMPASS deposit are not integrated into this experiment.",
            "- This run provides no causal, veterinary, medical, or consciousness-related inference.",
            "",
            "## Reproducibility",
            "",
            (
                "The resolved configuration, fitted parameters, master seed, derived "
                "per-series seeds, source checksums, software provenance, metrics, and "
                "artifact checksums are stored beside this report."
            ),
            "",
            f"Derived seeds recorded: {len(derived_seeds)}.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def write_artifact_manifest(directory: Path) -> dict[str, object]:
    """Hash every completed run artifact except the manifest itself."""

    artifacts = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.name == "artifact-manifest.json":
            continue
        artifacts.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "schema_version": "umwelt.artifacts.v1",
        "algorithm": "sha256",
        "artifacts": artifacts,
    }
    write_json(directory / "artifact-manifest.json", manifest)
    return manifest
