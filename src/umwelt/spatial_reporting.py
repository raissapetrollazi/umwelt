"""Compact reporting helpers for the Umwelt v0.2 spatial laboratory."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import median

from umwelt.datasets.roche_open_field import ROCHE_OPEN_FIELD_DATASET_ID
from umwelt.spatial_config import SpatialLabConfig
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
    records: list[dict[str, object]],
) -> str:
    metric_names = (
        "path_length_relative_difference",
        "displacement_wasserstein",
        "turning_wasserstein",
        "boundary_distance_wasserstein",
        "center_fraction_absolute_difference",
        "occupancy_total_variation",
    )
    lines = [
        f"# Experiment report: {config.experiment_id}",
        "",
        "## Interpretation boundary",
        "",
        "This development experiment asks whether a compact persistent reflecting random walk reproduces selected image-space geometric and kinematic summaries. Similarity does not establish a biological navigation mechanism, intention, anxiety, motivation, or subjective state.",
        "",
        "## Protocol",
        "",
        f"- Dataset: `{ROCHE_OPEN_FIELD_DATASET_ID}`",
        f"- Fitting animals: {', '.join(ROCHE_CONTROL_FITTING_SUBJECTS)}",
        f"- Development animals: {', '.join(ROCHE_CONTROL_DEVELOPMENT_SUBJECTS)}",
        f"- Replicates: {config.replicates}",
        f"- Master seed: {config.seed}",
        "- Position: recorded `bodycentre`, no invented likelihood cutoff, interpolation, or smoothing",
        "- Units: image pixels; no speed or physical-distance claims",
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
    for name in metric_names:
        values = [
            float(record["comparison"][name])
            for record in records
            if record["comparison"].get(name) is not None
        ]
        if values:
            lines.append(
                f"- `{name}` median across subject-replicates: {median(values):.6g}"
            )
    lines += [
        "",
        "These summaries are descriptive model-development evidence from two designated development animals. No pass/fail threshold or confirmatory population claim is assigned.",
        "",
    ]
    return "\n".join(lines)
