"""Markdown and dependency-free SVG reporting for individual-variation experiments."""

from __future__ import annotations

from collections.abc import Mapping
from html import escape
from pathlib import Path

from umwelt.temporal_config import TemporalLabConfig

POOLED_COLOR = "#006d77"
POPULATION_COLOR = "#6a4c93"
RECORDED_COLOR = "#111111"


def _number(value: object, digits: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def _interval(summary: Mapping[str, object], digits: int = 4) -> str:
    return (
        f"{_number(summary['median'], digits)} "
        f"[{_number(summary['lower'], digits)}, {_number(summary['upper'], digits)}]"
    )


def _aggregate_metric(model: Mapping[str, object], name: str) -> Mapping[str, object]:
    return model["summary"]["aggregate"]["metrics"][name]


def _aggregate_discrepancy(
    model: Mapping[str, object], name: str
) -> Mapping[str, object]:
    return model["summary"]["aggregate"]["discrepancies"][name]


def _between_subject(model: Mapping[str, object], name: str) -> Mapping[str, object]:
    return model["summary"]["between_subject"][name]


def _metric_table(
    models: list[Mapping[str, object]],
    recorded: Mapping[str, object],
    names: tuple[tuple[str, str], ...],
) -> list[str]:
    lines = [
        "| Metric | Recorded | "
        + " | ".join(str(model["label"]) for model in models)
        + " |",
        "| --- | ---: | " + " | ".join("---:" for _ in models) + " |",
    ]
    for name, label in names:
        lines.append(
            f"| {label} | {_number(recorded[name])} | "
            + " | ".join(_interval(_aggregate_metric(model, name)) for model in models)
            + " |"
        )
    return lines


def write_individual_variation_report(
    path: Path,
    *,
    config: TemporalLabConfig,
    provenance: Mapping[str, object],
    recorded: Mapping[str, object],
    comparison: Mapping[str, object],
) -> None:
    """Write a restrained report focused on between-individual variation."""

    models = list(comparison["models"])
    pooled, population = models
    recorded_scalars = recorded["aggregate"]["scalars"]
    recorded_between = recorded["between_subject"]
    lower, upper = config.evaluation.replicate_interval
    interval_label = f"{lower * 100:g}th-{upper * 100:g}th"

    lines = [
        f"# Experiment report: {provenance['experiment_id']}",
        "",
        "## Interpretation boundary",
        "",
        (
            "This experiment asks whether a minimal training-population variation mechanism "
            "can reproduce stable differences among synthetic individuals. The fitted offsets "
            "are computational parameters, not evidence of personality, physiology, intention, "
            "subjective state, or a biological random-effects mechanism."
        ),
        "",
        "## Evaluation protocol",
        "",
        "- Source: COMPASS weekly PIR activity and behaviorally defined sleep data",
        "- Dataset DOI: `10.5281/zenodo.160344`",
        f"- Observation interval: {recorded['epoch_seconds']} seconds",
        "- Sleep labels: behaviorally defined from immobility; not EEG-defined sleep",
        "- Training subjects: " + ", ".join(config.dataset.training_subjects),
        "- Held-out development subjects: "
        + ", ".join(config.dataset.development_subjects),
        (
            "- Exposure caveat: these development subjects were inspected in earlier v0.1 "
            "experiments, so this remains model-development evidence rather than a pristine "
            "confirmatory test."
        ),
        "- Development-subject behavior is not used to fit or select individual profiles.",
        "",
        "## Scientific question",
        "",
        (
            "Is training-population variation in a small set of state-dynamics parameters "
            "sufficient to reproduce the between-mouse heterogeneity missed by the pooled "
            "phase+duration model?"
        ),
        "",
        "## Population mechanism",
        "",
        (
            "The control is the existing pooled `phase-duration-hazard-v1` model. The population "
            "variant keeps its phase conditioning, duration bins, smoothing, and shared activity "
            "emissions unchanged. Each synthetic individual instead samples one fixed profile "
            "from the training population."
        ),
        "",
        "Each training profile contains only three smoothed logit offsets:",
        "",
        "- overall sleep occupancy relative to pooled training occupancy;",
        "- probability of leaving wake relative to the pooled wake hazard;",
        "- probability of leaving sleep relative to the pooled sleep hazard.",
        "",
        (
            "Profiles are sampled uniformly with replacement. They are global offsets applied "
            "to the pooled phase+duration dynamics; the model does not memorize a training "
            "mouse's trajectory and does not calibrate to a development mouse."
        ),
        "",
        "## Reproducibility",
        "",
        f"- Git revision: `{provenance['software']['git_revision']}`",
        f"- Configuration SHA-256: `{provenance['configuration_sha256']}`",
        f"- Master seed: {config.simulation.seed}",
        f"- Replicates per model: {config.simulation.replicates}",
        "- Pooled and population variants use paired trajectory random streams for each subject/replicate.",
        "- Population-profile selection uses a separate deterministic SHA-256-derived seed.",
        "- Full replicate trajectories are intentionally discarded after compact evaluation.",
        "",
        (
            "Synthetic values below are medians followed by the "
            f"{interval_label} predictive simulation interval."
        ),
        "",
        "## Aggregate fidelity",
        "",
    ]
    lines.extend(
        _metric_table(
            models,
            recorded_scalars,
            (
                ("sleep_fraction", "Sleep fraction"),
                ("mean_activity", "Mean activity"),
                ("nonzero_activity_fraction", "Nonzero activity fraction"),
                ("state_changes_per_hour", "State changes per hour"),
                ("wake_bout_median_seconds", "Wake bout median (s)"),
                ("sleep_bout_median_seconds", "Sleep bout median (s)"),
            ),
        )
    )

    lines.extend(
        [
            "",
            "Aggregate distribution and circadian discrepancies:",
            "",
            "| Model | Wake Wasserstein (s) | Sleep Wasserstein (s) | Sleep phase RMSE | Activity phase RMSE |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for model in models:
        lines.append(
            f"| {model['label']} | "
            f"{_interval(_aggregate_discrepancy(model, 'wake_bout_wasserstein_seconds'))} | "
            f"{_interval(_aggregate_discrepancy(model, 'sleep_bout_wasserstein_seconds'))} | "
            f"{_interval(_aggregate_discrepancy(model, 'sleep_phase_rmse'))} | "
            f"{_interval(_aggregate_discrepancy(model, 'activity_phase_rmse'))} |"
        )

    lines.extend(["", "## Between-subject heterogeneity", ""])
    lines.extend(
        [
            "The statistics below are descriptive spread across the development mice inside each weekly replicate population.",
            "",
            "| Spread metric | Recorded | Pooled | Population variation |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    between_names = (
        ("sleep_fraction_sd", "Sleep fraction SD"),
        ("sleep_fraction_range", "Sleep fraction range"),
        ("mean_activity_sd", "Mean activity SD"),
        ("state_changes_per_hour_sd", "State changes/hour SD"),
        ("wake_bout_median_seconds_sd", "Wake bout median SD (s)"),
        ("sleep_bout_median_seconds_sd", "Sleep bout median SD (s)"),
    )
    for name, label in between_names:
        lines.append(
            f"| {label} | {_number(recorded_between[name])} | "
            f"{_interval(_between_subject(pooled, name))} | "
            f"{_interval(_between_subject(population, name))} |"
        )

    lines.extend(
        [
            "",
            "See `individual-variation.svg` for per-subject sleep-fraction predictive intervals.",
            "",
            "## Individual mice",
            "",
            "| Subject | Metric | Recorded | Pooled | Population variation |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for subject_id in config.dataset.development_subjects:
        subject_recorded = recorded["subjects"][subject_id]["scalars"]
        for name, label in (
            ("sleep_fraction", "Sleep fraction"),
            ("state_changes_per_hour", "Changes/hour"),
            ("wake_bout_median_seconds", "Wake bout median (s)"),
            ("sleep_bout_median_seconds", "Sleep bout median (s)"),
        ):
            lines.append(
                f"| {subject_id} | {label} | {_number(subject_recorded[name])} | "
                f"{_interval(pooled['summary']['subjects'][subject_id]['metrics'][name])} | "
                f"{_interval(population['summary']['subjects'][subject_id]['metrics'][name])} |"
            )

    heterogeneity_declared = [name for name, _ in between_names]
    pooled_inside = sum(
        _between_subject(pooled, name)["reference_within_interval"] is True
        for name in heterogeneity_declared
    )
    population_inside = sum(
        _between_subject(population, name)["reference_within_interval"] is True
        for name in heterogeneity_declared
    )

    lines.extend(
        [
            "",
            "## Descriptive interpretation",
            "",
            (
                f"Across the {len(heterogeneity_declared)} declared heterogeneity summaries, "
                f"the recorded value fell inside the pooled predictive interval for {pooled_inside} "
                f"and inside the population-variation interval for {population_inside}. This count "
                "is a diagnostic summary, not an acceptance score or hypothesis test."
            ),
            "",
            (
                "The population mechanism should be considered useful only where it increases "
                "realistic between-subject spread without materially damaging the temporal and "
                "circadian structure already captured by the phase+duration model."
            ),
            "",
            "## Remaining limitations",
            "",
            "- The empirical population contains only the training mice in this COMPASS experiment.",
            "- Only three global state-dynamics offsets vary between synthetic individuals.",
            "- The profile distribution is resampled rather than estimated as a continuous biological population distribution.",
            "- Activity-emission parameters remain pooled and conditionally independent across epochs.",
            "- Development subjects were previously inspected and cannot support pristine confirmatory claims.",
            "- Behaviorally defined immobility is not sleep-stage physiology.",
            "",
            "## Next experiment",
            "",
            (
                "If training-population offsets improve heterogeneity while preserving temporal "
                "fidelity, the next v0.1 question is whether the effect survives prospective "
                "subject-level validation restricted to training animals and whether activity "
                "emissions require their own explicit individual or serial dynamics. If not, "
                "the failure itself should determine which minimal population mechanism is tested next."
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def write_individual_variation_svg(
    path: Path,
    *,
    config: TemporalLabConfig,
    recorded: Mapping[str, object],
    comparison: Mapping[str, object],
) -> None:
    """Plot recorded sleep fractions against pooled and population predictive intervals."""

    pooled, population = comparison["models"]
    subjects = config.dataset.development_subjects
    width = 960
    height = 150 + len(subjects) * 76
    left = 170
    right = 910
    top = 92

    def x(value: float) -> float:
        return left + value * (right - left)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        "<style>text{font-family:Arial,sans-serif;fill:#202124}.title{font-size:20px;font-weight:700}.label{font-size:12px}.subject{font-size:13px;font-weight:700}</style>",
        '<text class="title" x="24" y="32">Individual sleep-fraction variation</text>',
        '<text class="label" x="24" y="52">Recorded weekly values and predictive simulation intervals; development evidence only.</text>',
        f'<line x1="{left}" y1="{top - 12}" x2="{right}" y2="{top - 12}" stroke="#202124"/>',
    ]
    for tick in range(6):
        value = tick / 5
        position = x(value)
        parts.extend(
            [
                f'<line x1="{position:.2f}" y1="{top - 16}" x2="{position:.2f}" y2="{height - 42}" stroke="#e2e5e7"/>',
                f'<text class="label" text-anchor="middle" x="{position:.2f}" y="{top - 22}">{value:.1f}</text>',
            ]
        )

    for index, subject_id in enumerate(subjects):
        center = top + index * 76 + 22
        recorded_value = float(recorded["subjects"][subject_id]["scalars"]["sleep_fraction"])
        pooled_summary = pooled["summary"]["subjects"][subject_id]["metrics"]["sleep_fraction"]
        population_summary = population["summary"]["subjects"][subject_id]["metrics"]["sleep_fraction"]
        parts.append(
            f'<text class="subject" text-anchor="end" x="{left - 18}" y="{center + 4}">Mouse {escape(str(subject_id))}</text>'
        )
        for row_offset, summary, color in (
            (-12, pooled_summary, POOLED_COLOR),
            (12, population_summary, POPULATION_COLOR),
        ):
            y = center + row_offset
            lower = x(float(summary["lower"]))
            upper = x(float(summary["upper"]))
            median = x(float(summary["median"]))
            parts.extend(
                [
                    f'<line x1="{lower:.2f}" y1="{y}" x2="{upper:.2f}" y2="{y}" stroke="{color}" stroke-width="5" opacity="0.55"/>',
                    f'<circle cx="{median:.2f}" cy="{y}" r="5" fill="{color}"/>',
                ]
            )
        parts.extend(
            [
                f'<line x1="{x(recorded_value):.2f}" y1="{center - 23}" x2="{x(recorded_value):.2f}" y2="{center + 23}" stroke="{RECORDED_COLOR}" stroke-width="3"/>',
                f'<text class="label" x="{right + 8}" y="{center + 4}">{recorded_value:.3f}</text>',
            ]
        )

    legend_y = height - 18
    parts.extend(
        [
            f'<line x1="{left}" y1="{legend_y - 4}" x2="{left + 28}" y2="{legend_y - 4}" stroke="{POOLED_COLOR}" stroke-width="5"/><text class="label" x="{left + 36}" y="{legend_y}">Pooled phase+duration</text>',
            f'<line x1="{left + 210}" y1="{legend_y - 4}" x2="{left + 238}" y2="{legend_y - 4}" stroke="{POPULATION_COLOR}" stroke-width="5"/><text class="label" x="{left + 246}" y="{legend_y}">Population variation</text>',
            f'<line x1="{left + 440}" y1="{legend_y - 14}" x2="{left + 440}" y2="{legend_y + 4}" stroke="{RECORDED_COLOR}" stroke-width="3"/><text class="label" x="{left + 450}" y="{legend_y}">Recorded</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts) + "\n", encoding="utf-8", newline="\n")
