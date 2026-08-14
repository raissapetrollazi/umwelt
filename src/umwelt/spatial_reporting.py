"""Compact reporting helpers for the Umwelt v0.2 spatial laboratory."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from statistics import median

from umwelt.datasets.roche_open_field import ROCHE_OPEN_FIELD_DATASET_ID
from umwelt.spatial_config import SpatialLabConfig
from umwelt.spatial_evaluation import SpatialComparison
from umwelt.spatial_protocol import (
    ROCHE_CONTROL_DEVELOPMENT_SUBJECTS,
    ROCHE_CONTROL_FITTING_SUBJECTS,
)


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_spatial_report(
    *,
    config: SpatialLabConfig,
    model: dict[str, object],
    comparisons: Sequence[tuple[str, SpatialComparison]],
) -> str:
    metrics = (
        (
            "path_length_signed_relative_difference",
            "ratio: (synthetic - recorded) / recorded",
        ),
        ("adjacent_displacement_wasserstein_px", "px"),
        ("absolute_turning_wasserstein_radians", "radian"),
    )
    lines = [
        f"# Experiment report: {config.experiment_id}",
        "",
        "## Interpretation boundary",
        "",
        (
            "This development experiment asks whether a compact persistent "
            "reflecting random walk reproduces three frozen image-space movement "
            "summaries. Similarity does not establish a biological navigation "
            "mechanism, intention, anxiety, motivation, or subjective state."
        ),
        "",
        "## Protocol",
        "",
        f"- Dataset: `{ROCHE_OPEN_FIELD_DATASET_ID}`",
        f"- Fitting animals: {', '.join(ROCHE_CONTROL_FITTING_SUBJECTS)}",
        f"- Development animals: {', '.join(ROCHE_CONTROL_DEVELOPMENT_SUBJECTS)}",
        f"- Replicates: {config.replicates}",
        f"- Master seed: {config.seed}",
        (
            "- Position: recorded `bodycentre`, no invented likelihood cutoff, "
            "interpolation, or smoothing"
        ),
        "- Units: image pixels; no speed or physical-distance claims",
        (
            "- Registered metrics: path length with transition exposure, adjacent "
            "displacement, and absolute turning"
        ),
        (
            "- Arena bounds constrain the model but boundary, center, and occupancy "
            "metrics remain unregistered"
        ),
        (
            "- Synthetic evaluation copies only the recorded bodycentre availability "
            "mask, never recorded coordinate values"
        ),
        "- Synthetic trajectories are discarded after compact evaluation.",
        "",
        "## Model",
        "",
        "```json",
        json.dumps(model, indent=2, sort_keys=True),
        "```",
        "",
        "## Development comparison",
        "",
    ]
    for name, unit in metrics:
        subject_medians: list[tuple[str, float]] = []
        for subject_id in ROCHE_CONTROL_DEVELOPMENT_SUBJECTS:
            values = [
                float(value)
                for current_subject, comparison in comparisons
                if current_subject == subject_id
                and (value := getattr(comparison, name)) is not None
            ]
            if values:
                subject_medians.append((subject_id, median(values)))
        if subject_medians:
            rendered = ", ".join(
                f"`{subject_id}`={value:.6g}" for subject_id, value in subject_medians
            )
            animal_median = median(value for _, value in subject_medians)
            lines.append(f"- `{name}` ({unit}): {rendered}")
            lines.append(
                f"  - Median of the {len(subject_medians)} animal medians: "
                f"{animal_median:.6g}"
            )
    lines += [
        "",
        (
            "These summaries are descriptive model-development evidence from two "
            "designated development animals. Replicate frames are not treated as "
            "independent animals. No pass/fail threshold or confirmatory population "
            "claim is assigned."
        ),
        "",
    ]
    return "\n".join(lines)
