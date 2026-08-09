"""Markdown and dependency-free SVG reporting for temporal model comparisons."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape
from pathlib import Path

from umwelt.temporal_config import TemporalLabConfig

_COLORS = ("#006d77", "#d1495b", "#6a4c93", "#3a7d44")


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


def _median(summary: Mapping[str, object]) -> float:
    value = summary["median"]
    return float(value) if value is not None else float("nan")


def _change(first: float, second: float) -> str:
    if second < first:
        direction = "lower"
    elif second > first:
        direction = "higher"
    else:
        direction = "unchanged"
    return f"{_number(first)} to {_number(second)} ({direction})"


def _metric_table(
    models: Sequence[Mapping[str, object]],
    recorded: Mapping[str, object],
    names: Sequence[tuple[str, str]],
) -> list[str]:
    header = (
        "| Metric | Recorded | "
        + " | ".join(str(model["label"]) for model in models)
        + " |"
    )
    separator = "| --- | ---: | " + " | ".join("---:" for _ in models) + " |"
    lines = [header, separator]
    for name, label in names:
        cells = [_interval(_aggregate_metric(model, name)) for model in models]
        lines.append(
            f"| {label} | {_number(recorded[name])} | " + " | ".join(cells) + " |"
        )
    return lines


def write_temporal_report(
    path: Path,
    *,
    config: TemporalLabConfig,
    provenance: Mapping[str, object],
    recorded: Mapping[str, object],
    comparison: Mapping[str, object],
) -> None:
    """Write a restrained model-comparison report with explicit limitations."""

    models = comparison["models"]
    recorded_scalars = recorded["aggregate"]["scalars"]
    lower, upper = config.evaluation.replicate_interval
    interval_label = f"{lower * 100:g}th-{upper * 100:g}th"
    lines = [
        f"# Experiment report: {config.experiment_id}",
        "",
        "## Interpretation boundary",
        "",
        (
            "This is a descriptive computational model comparison. Synthetic states "
            "are explicit model states; they are not evidence about a real animal's "
            "subjective state, cognition, intention, emotion, or consciousness. "
            "Correspondence with recorded observations does not identify the biological "
            "mechanism that produced them."
        ),
        "",
        "## Dataset and evaluation protocol",
        "",
        "- Source: COMPASS weekly PIR activity and behaviorally defined sleep data",
        "- Dataset DOI: `10.5281/zenodo.160344`",
        "- Dataset license: CC0-1.0",
        f"- Observation interval: {recorded['epoch_seconds']} seconds",
        "- Sleep label: at least 40 seconds of immobility; not EEG-defined sleep",
        "- Training subjects: " + ", ".join(config.dataset.training_subjects),
        "- Held-out development subjects: "
        + ", ".join(config.dataset.development_subjects),
        (
            "- Exposure caveat: results for subjects 6, 12, 18, and 24 from the original "
            "baseline were inspected before this model ladder was specified. This is "
            "model-development evidence, not a pristine confirmatory test."
        ),
        (
            "- Generalization question: each pooled model is fitted without the held-out "
            "development subjects and is not calibrated to any individual development mouse."
        ),
        "",
        "## Scientific question",
        "",
        (
            "Which minimal temporal mechanisms reproduce recorded mouse sleep/activity "
            "dynamics beyond aggregate occupancy?"
        ),
        "",
        "## Model ladder",
        "",
        "All models share exactly the same fitted phase-conditioned, zero-inflated gamma activity-emission mechanism.",
        "",
        "| Model | Phase | Duration | State parameters | Hypothesis |",
        "| --- | :---: | :---: | ---: | --- |",
    ]
    for model in models:
        mechanisms = model["mechanisms"]
        lines.append(
            f"| {model['label']} (`{model['model_id']}`) | "
            f"{'yes' if mechanisms['phase_conditioning'] else 'no'} | "
            f"{'yes' if mechanisms['duration_conditioning'] else 'no'} | "
            f"{model['parameter_count']} | {model['hypothesis']} |"
        )
    lines.extend(
        [
            "",
            "Duration-aware models use fixed elapsed-bout bins of "
            + ", ".join(str(value) for value in config.model.duration_bin_edges_epochs)
            + " epochs. These bins and smoothing choices were declared in configuration and were not tuned against this run.",
            "",
            "## Reproducibility",
            "",
            f"- Git revision: `{provenance['software']['git_revision']}`",
            f"- Configuration SHA-256: `{provenance['configuration_sha256']}`",
            f"- Master seed: {config.simulation.seed}",
            f"- Replicates per model: {config.simulation.replicates}",
            "- Seed derivation: SHA-256 over master seed, model identifier, source subject, and replicate number; first 64 bits",
            "- Source file checksums: preserved in `provenance.json`",
            "- Artifact checksums: listed in `artifact-manifest.json`",
            "- Full replicate trajectories: intentionally not retained",
            "",
            (
                "Values below are synthetic medians followed by the "
                f"{interval_label} predictive simulation interval in brackets."
            ),
            "",
            "## Aggregate comparison",
            "",
        ]
    )
    lines.extend(
        _metric_table(
            models,
            recorded_scalars,
            (
                ("sleep_fraction", "Sleep fraction"),
                ("wake_fraction", "Wake fraction"),
                ("mean_activity", "Mean activity"),
                ("nonzero_activity_fraction", "Nonzero activity fraction"),
                ("state_changes_per_hour", "State changes per hour"),
            ),
        )
    )
    lines.extend(["", "## Bout-duration comparison", ""])
    lines.extend(
        _metric_table(
            models,
            recorded_scalars,
            tuple(
                (
                    f"{state}_bout_{statistic}_seconds",
                    f"{state.title()} bout {statistic} (s)",
                )
                for state in ("wake", "sleep")
                for statistic in ("mean", "median", "q90")
            ),
        )
    )
    lines.extend(
        [
            "",
            "Distribution discrepancies compare each synthetic replicate with the pooled recorded empirical bout distribution; zero is ideal but not expected.",
            "",
            "| Model | Wake Wasserstein (s) | Wake KS | Sleep Wasserstein (s) | Sleep KS |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for model in models:
        lines.append(
            f"| {model['label']} | "
            f"{_interval(_aggregate_discrepancy(model, 'wake_bout_wasserstein_seconds'))} | "
            f"{_interval(_aggregate_discrepancy(model, 'wake_bout_ks'))} | "
            f"{_interval(_aggregate_discrepancy(model, 'sleep_bout_wasserstein_seconds'))} | "
            f"{_interval(_aggregate_discrepancy(model, 'sleep_bout_ks'))} |"
        )

    lines.extend(["", "## Temporal-dependence comparison", ""])
    autocorrelation_names = tuple(
        (f"sleep_autocorrelation_lag_{lag}", f"Sleep autocorrelation, lag {lag}")
        for lag in config.evaluation.autocorrelation_lags
    )
    lines.extend(_metric_table(models, recorded_scalars, autocorrelation_names))
    lines.extend(
        [
            "",
            "The declared 10-second lags correspond to "
            + ", ".join(
                f"{lag * recorded['epoch_seconds']} s"
                for lag in config.evaluation.autocorrelation_lags
            )
            + ". See `autocorrelation.svg` for replicate bands.",
            "",
            "## Circadian comparison",
            "",
            "| Model | Sleep phase RMSE | Activity phase RMSE |",
            "| --- | ---: | ---: |",
        ]
    )
    for model in models:
        lines.append(
            f"| {model['label']} | "
            f"{_interval(_aggregate_discrepancy(model, 'sleep_phase_rmse'))} | "
            f"{_interval(_aggregate_discrepancy(model, 'activity_phase_rmse'))} |"
        )

    lines.extend(
        [
            "",
            "## Individual-level comparison",
            "",
            "Recorded values remain one observed weekly statistic per mouse; synthetic columns retain replicate variation.",
            "",
            "| Subject | Metric | Recorded | "
            + " | ".join(str(model["label"]) for model in models)
            + " |",
            "| --- | --- | ---: | " + " | ".join("---:" for _ in models) + " |",
        ]
    )
    for subject_id in config.dataset.development_subjects:
        subject_recorded = recorded["subjects"][subject_id]["scalars"]
        for name, label in (
            ("sleep_fraction", "Sleep fraction"),
            ("state_changes_per_hour", "Changes/hour"),
        ):
            cells = [
                _interval(model["summary"]["subjects"][subject_id]["metrics"][name])
                for model in models
            ]
            lines.append(
                f"| {subject_id} | {label} | {_number(subject_recorded[name])} | "
                + " | ".join(cells)
                + " |"
            )
    lines.extend(
        [
            "",
            "Subject-level bout-distribution discrepancies:",
            "",
            "| Subject | Model | Wake Wasserstein (s) | Sleep Wasserstein (s) |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for subject_id in config.dataset.development_subjects:
        for model in models:
            discrepancies = model["summary"]["subjects"][subject_id]["discrepancies"]
            lines.append(
                f"| {subject_id} | {model['label']} | "
                f"{_interval(discrepancies['wake_bout_wasserstein_seconds'])} | "
                f"{_interval(discrepancies['sleep_bout_wasserstein_seconds'])} |"
            )

    state_only, phase_only, duration_only, phase_duration = models
    lines.extend(
        [
            "",
            "## Replicate uncertainty",
            "",
            (
                f"Every model/subject combination has {config.simulation.replicates} "
                f"deterministically seeded synthetic replicates. The {interval_label} "
                "ranges are predictive simulation intervals, not confidence intervals. "
                "`replicate-metrics.jsonl` retains each compact result."
            ),
            "",
            "## Mechanistic interpretation",
            "",
            "The comparisons below describe changed discrepancies; they are not hypothesis-test results.",
            "",
            "- Adding phase to the state-only model changed median sleep-phase RMSE from "
            + _change(
                _median(_aggregate_discrepancy(state_only, "sleep_phase_rmse")),
                _median(_aggregate_discrepancy(phase_only, "sleep_phase_rmse")),
            )
            + ".",
            "- Adding duration without phase changed median wake/sleep Wasserstein distances from "
            + _change(
                _median(
                    _aggregate_discrepancy(state_only, "wake_bout_wasserstein_seconds")
                ),
                _median(
                    _aggregate_discrepancy(
                        duration_only, "wake_bout_wasserstein_seconds"
                    )
                ),
            )
            + " and "
            + _change(
                _median(
                    _aggregate_discrepancy(state_only, "sleep_bout_wasserstein_seconds")
                ),
                _median(
                    _aggregate_discrepancy(
                        duration_only, "sleep_bout_wasserstein_seconds"
                    )
                ),
            )
            + ", respectively.",
            "- Adding duration to the phase-conditioned baseline changed those distances from "
            + _change(
                _median(
                    _aggregate_discrepancy(phase_only, "wake_bout_wasserstein_seconds")
                ),
                _median(
                    _aggregate_discrepancy(
                        phase_duration, "wake_bout_wasserstein_seconds"
                    )
                ),
            )
            + " and "
            + _change(
                _median(
                    _aggregate_discrepancy(phase_only, "sleep_bout_wasserstein_seconds")
                ),
                _median(
                    _aggregate_discrepancy(
                        phase_duration, "sleep_bout_wasserstein_seconds"
                    )
                ),
            )
            + ".",
            "",
            "## Failures",
            "",
        ]
    )
    declared = (
        "sleep_fraction",
        "mean_activity",
        "nonzero_activity_fraction",
        "state_changes_per_hour",
        "wake_bout_mean_seconds",
        "wake_bout_median_seconds",
        "wake_bout_q90_seconds",
        "sleep_bout_mean_seconds",
        "sleep_bout_median_seconds",
        "sleep_bout_q90_seconds",
        *(
            f"sleep_autocorrelation_lag_{lag}"
            for lag in config.evaluation.autocorrelation_lags
        ),
    )
    for model in models:
        outside = sum(
            _aggregate_metric(model, name)["reference_within_interval"] is False
            for name in declared
        )
        lines.append(
            f"- {model['label']}: the recorded aggregate statistic lay outside the "
            f"predictive interval for {outside} of {len(declared)} declared direct metrics. "
            f"Median wake/sleep Wasserstein distances were "
            f"{_number(_median(_aggregate_discrepancy(model, 'wake_bout_wasserstein_seconds')))} s and "
            f"{_number(_median(_aggregate_discrepancy(model, 'sleep_bout_wasserstein_seconds')))} s."
        )
    lines.extend(
        [
            "- The pooled models do not represent stable individual differences.",
            "- Activity emissions remain conditionally independent across epochs and were not revised in this comparison.",
            "- Behaviorally defined immobility labels are not sleep-stage physiology.",
            "- No pass/fail threshold or confirmatory statistical test was preregistered.",
            "",
            "## Scientific conclusion",
            "",
            (
                "This factorial comparison identifies whether explicit phase and elapsed-duration memory reduce specific descriptive discrepancies. "
                "It does not establish that either conditioning variable is a biological mechanism. "
                "The simplest adequate model cannot be declared without prospective acceptance criteria; the tables show the tradeoffs that should define such criteria before another evaluation."
            ),
            "",
            "## Next experiment",
            "",
            (
                "Remain within v0.1. Define acceptance criteria and assess fixed duration-bin and smoothing choices through subject-level validation restricted to the training animals. "
                "Only then should a new untouched data source or prospectively declared split be used for stronger generalization claims. EEG validation and spatial behavior remain separate future questions."
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def write_model_comparison_svg(path: Path, comparison: Mapping[str, object]) -> None:
    """Write a four-panel SVG without collapsing dimensions into one score."""

    models = comparison["models"]
    panels = (
        ("Sleep phase RMSE", "sleep_phase_rmse"),
        ("Wake bout Wasserstein (s)", "wake_bout_wasserstein_seconds"),
        ("Sleep bout Wasserstein (s)", "sleep_bout_wasserstein_seconds"),
        ("Activity phase RMSE", "activity_phase_rmse"),
    )
    width = 960
    height = 520
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        "<style>text{font-family:Arial,sans-serif;fill:#202124;letter-spacing:0}.title{font-size:20px;font-weight:700}.label{font-size:12px}.value{font-size:11px}</style>",
        '<text class="title" x="24" y="32">Temporal model comparison</text>',
        '<text class="label" x="24" y="52">Median discrepancy across synthetic replicates; each panel has its own scale and zero is left.</text>',
    ]
    for panel_index, (title, metric) in enumerate(panels):
        column = panel_index % 2
        row = panel_index // 2
        origin_x = 24 + column * 468
        origin_y = 82 + row * 214
        values = [_median(_aggregate_discrepancy(model, metric)) for model in models]
        maximum = max(values) if max(values) > 0 else 1.0
        parts.append(
            f'<text class="label" font-weight="700" x="{origin_x}" y="{origin_y}">{escape(title)}</text>'
        )
        for index, (model, value, color) in enumerate(
            zip(models, values, _COLORS, strict=True)
        ):
            y = origin_y + 24 + index * 38
            bar_width = 260 * value / maximum
            parts.extend(
                [
                    f'<text class="label" x="{origin_x}" y="{y + 13}">{escape(str(model["label"]))}</text>',
                    f'<rect x="{origin_x + 112}" y="{y}" width="260" height="18" fill="#edf0f2"/>',
                    f'<rect x="{origin_x + 112}" y="{y}" width="{bar_width:.2f}" height="18" fill="{color}"/>',
                    f'<text class="value" x="{origin_x + 380}" y="{y + 13}">{value:.4f}</text>',
                ]
            )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8", newline="\n")


def write_autocorrelation_svg(
    path: Path,
    *,
    comparison: Mapping[str, object],
    recorded: Mapping[str, object],
    lags: Sequence[int],
) -> None:
    """Write recorded autocorrelation and model replicate bands as deterministic SVG."""

    models = comparison["models"]
    recorded_values = [
        float(recorded["aggregate"]["scalars"][f"sleep_autocorrelation_lag_{lag}"])
        for lag in lags
    ]
    all_values = list(recorded_values)
    for model in models:
        for lag in lags:
            summary = _aggregate_metric(model, f"sleep_autocorrelation_lag_{lag}")
            all_values.extend(float(summary[key]) for key in ("lower", "upper"))
    minimum = min(-0.05, min(all_values))
    maximum = max(0.05, max(all_values))
    padding = max((maximum - minimum) * 0.08, 0.02)
    minimum -= padding
    maximum += padding
    width = 960
    height = 500
    left = 76
    right = 930
    top = 70
    bottom = 410

    def x(index: int) -> float:
        return left + index * (right - left) / max(len(lags) - 1, 1)

    def y(value: float) -> float:
        return bottom - (value - minimum) * (bottom - top) / (maximum - minimum)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        "<style>text{font-family:Arial,sans-serif;fill:#202124;letter-spacing:0}.title{font-size:20px;font-weight:700}.label{font-size:12px}</style>",
        '<text class="title" x="24" y="32">Sleep-state autocorrelation</text>',
        '<text class="label" x="24" y="52">Lines are replicate medians; translucent regions span the configured predictive simulation interval.</text>',
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#202124"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#202124"/>',
    ]
    for index, lag in enumerate(lags):
        parts.extend(
            [
                f'<line x1="{x(index):.2f}" y1="{bottom}" x2="{x(index):.2f}" y2="{bottom + 5}" stroke="#202124"/>',
                f'<text class="label" text-anchor="middle" x="{x(index):.2f}" y="{bottom + 22}">{lag}</text>',
            ]
        )
    for tick in range(6):
        value = minimum + tick * (maximum - minimum) / 5
        position = y(value)
        parts.extend(
            [
                f'<line x1="{left}" y1="{position:.2f}" x2="{right}" y2="{position:.2f}" stroke="#e2e5e7"/>',
                f'<text class="label" text-anchor="end" x="{left - 8}" y="{position + 4:.2f}">{value:.2f}</text>',
            ]
        )
    for model, color in zip(models, _COLORS, strict=True):
        summaries = [
            _aggregate_metric(model, f"sleep_autocorrelation_lag_{lag}") for lag in lags
        ]
        upper_points = " ".join(
            f"{x(index):.2f},{y(float(summary['upper'])):.2f}"
            for index, summary in enumerate(summaries)
        )
        lower_points = " ".join(
            f"{x(index):.2f},{y(float(summary['lower'])):.2f}"
            for index, summary in reversed(list(enumerate(summaries)))
        )
        median_points = " ".join(
            f"{x(index):.2f},{y(float(summary['median'])):.2f}"
            for index, summary in enumerate(summaries)
        )
        parts.extend(
            [
                f'<polygon points="{upper_points} {lower_points}" fill="{color}" opacity="0.12"/>',
                f'<polyline points="{median_points}" fill="none" stroke="{color}" stroke-width="2"/>',
            ]
        )
    recorded_points = " ".join(
        f"{x(index):.2f},{y(value):.2f}" for index, value in enumerate(recorded_values)
    )
    parts.append(
        f'<polyline points="{recorded_points}" fill="none" stroke="#111111" stroke-width="3" stroke-dasharray="7 4"/>'
    )
    legend_y = 466
    parts.append(
        f'<line x1="70" y1="{legend_y - 4}" x2="100" y2="{legend_y - 4}" stroke="#111111" stroke-width="3" stroke-dasharray="7 4"/><text class="label" x="108" y="{legend_y}">Recorded</text>'
    )
    for index, (model, color) in enumerate(zip(models, _COLORS, strict=True)):
        legend_x = 205 + index * 175
        parts.append(
            f'<line x1="{legend_x}" y1="{legend_y - 4}" x2="{legend_x + 28}" y2="{legend_y - 4}" stroke="{color}" stroke-width="3"/><text class="label" x="{legend_x + 35}" y="{legend_y}">{escape(str(model["label"]))}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8", newline="\n")
