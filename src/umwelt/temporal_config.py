"""Strict configuration for replicated Umwelt temporal model comparisons."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from umwelt.config import (
    _exact_keys,
    _integer,
    _integer_tuple,
    _mapping,
    _number,
    _string,
    _string_tuple,
)
from umwelt.datasets.compass import COMPASS_DATASET_ID, COMPASS_SUBJECT_IDS
from umwelt.errors import ConfigurationError

TEMPORAL_CONFIG_SCHEMA_VERSION = "umwelt.temporal-lab.v1"


def _number_pair(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ConfigurationError(f"{label} must contain exactly two numbers.")
    return (_number(value[0], f"{label}[0]"), _number(value[1], f"{label}[1]"))


@dataclass(frozen=True, slots=True)
class TemporalDatasetConfig:
    """Recorded data selection with an explicit development-only holdout role."""

    dataset_id: str
    directory: Path
    training_subjects: tuple[str, ...]
    development_subjects: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TemporalModelConfig:
    """Shared fixed choices for the four-model factorial ladder."""

    phase_bins: int
    duration_bin_edges_epochs: tuple[int, ...]
    transition_prior: float
    duration_prior_strength: float
    emission_prior: float


@dataclass(frozen=True, slots=True)
class TemporalSimulationConfig:
    """Replicate count and deterministic master seed."""

    seed: int
    replicates: int


@dataclass(frozen=True, slots=True)
class TemporalEvaluationConfig:
    """Declared metrics and predictive replicate interval."""

    phase_bins: int
    autocorrelation_lags: tuple[int, ...]
    replicate_interval: tuple[float, float]


@dataclass(frozen=True, slots=True)
class TemporalLabConfig:
    """Complete configuration for one temporal model-comparison run."""

    experiment_id: str
    dataset: TemporalDatasetConfig
    model: TemporalModelConfig
    simulation: TemporalSimulationConfig
    evaluation: TemporalEvaluationConfig
    output_directory: Path
    source_path: Path | None = None

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", self.experiment_id):
            raise ConfigurationError(
                "experiment_id must contain lowercase letters, digits, dots, "
                "underscores, or hyphens."
            )
        if self.dataset.dataset_id != COMPASS_DATASET_ID:
            raise ConfigurationError(
                f"The temporal lab requires dataset id {COMPASS_DATASET_ID}."
            )
        available = set(COMPASS_SUBJECT_IDS)
        selected = set(self.dataset.training_subjects) | set(
            self.dataset.development_subjects
        )
        unknown = sorted(selected - available, key=int)
        if unknown:
            raise ConfigurationError(
                "Unknown COMPASS subject identifiers: " + ", ".join(unknown)
            )
        overlap = set(self.dataset.training_subjects) & set(
            self.dataset.development_subjects
        )
        if overlap:
            raise ConfigurationError(
                "Training and development subjects must be disjoint: "
                + ", ".join(sorted(overlap, key=int))
            )
        if not 1 <= self.model.phase_bins <= 1_440:
            raise ConfigurationError("model.phase_bins must be between 1 and 1440.")
        edges = self.model.duration_bin_edges_epochs
        if any(edge < 1 for edge in edges) or tuple(sorted(set(edges))) != edges:
            raise ConfigurationError(
                "model.duration_bin_edges_epochs must be strictly increasing and positive."
            )
        if (
            min(
                self.model.transition_prior,
                self.model.duration_prior_strength,
                self.model.emission_prior,
            )
            <= 0
        ):
            raise ConfigurationError("Model smoothing parameters must be positive.")
        if not 0 <= self.simulation.seed <= 2**64 - 1:
            raise ConfigurationError(
                "simulation.seed must be a 64-bit unsigned integer."
            )
        if not 1 <= self.simulation.replicates <= 100:
            raise ConfigurationError("simulation.replicates must be between 1 and 100.")
        if not 1 <= self.evaluation.phase_bins <= 1_440:
            raise ConfigurationError(
                "evaluation.phase_bins must be between 1 and 1440."
            )
        if any(lag <= 0 for lag in self.evaluation.autocorrelation_lags):
            raise ConfigurationError(
                "evaluation.autocorrelation_lags must contain positive integers."
            )
        lower, upper = self.evaluation.replicate_interval
        if not 0 <= lower < upper <= 1:
            raise ConfigurationError(
                "evaluation.replicate_interval must satisfy 0 <= lower < upper <= 1."
            )

    def to_dict(self) -> dict[str, object]:
        """Return a canonical JSON-compatible representation."""

        return {
            "schema_version": TEMPORAL_CONFIG_SCHEMA_VERSION,
            "experiment_id": self.experiment_id,
            "dataset": {
                "id": self.dataset.dataset_id,
                "directory": str(self.dataset.directory),
                "training_subjects": list(self.dataset.training_subjects),
                "development_subjects": list(self.dataset.development_subjects),
            },
            "model": {
                "phase_bins": self.model.phase_bins,
                "duration_bin_edges_epochs": list(self.model.duration_bin_edges_epochs),
                "transition_prior": self.model.transition_prior,
                "duration_prior_strength": self.model.duration_prior_strength,
                "emission_prior": self.model.emission_prior,
            },
            "simulation": {
                "seed": self.simulation.seed,
                "replicates": self.simulation.replicates,
            },
            "evaluation": {
                "phase_bins": self.evaluation.phase_bins,
                "autocorrelation_lags": list(self.evaluation.autocorrelation_lags),
                "replicate_interval": list(self.evaluation.replicate_interval),
            },
            "output": {"directory": str(self.output_directory)},
        }


def load_temporal_lab_config(path: str | Path) -> TemporalLabConfig:
    """Load and strictly validate one temporal-lab JSON configuration."""

    source_path = Path(path).resolve()
    try:
        with source_path.open(encoding="utf-8") as handle:
            raw = json.load(handle)
    except OSError as error:
        raise ConfigurationError(f"Could not read configuration: {error}") from error
    except json.JSONDecodeError as error:
        raise ConfigurationError(
            f"Invalid JSON configuration at line {error.lineno}, column {error.colno}."
        ) from error

    root = _mapping(raw, "configuration")
    _exact_keys(
        root,
        required={
            "schema_version",
            "experiment_id",
            "dataset",
            "model",
            "simulation",
            "evaluation",
            "output",
        },
        optional=set(),
        label="configuration",
    )
    if root["schema_version"] != TEMPORAL_CONFIG_SCHEMA_VERSION:
        raise ConfigurationError(
            f"Unsupported temporal-lab schema: {root['schema_version']!r}"
        )

    dataset = _mapping(root["dataset"], "dataset")
    _exact_keys(
        dataset,
        required={"id", "directory", "training_subjects", "development_subjects"},
        optional=set(),
        label="dataset",
    )
    model = _mapping(root["model"], "model")
    _exact_keys(
        model,
        required={
            "phase_bins",
            "duration_bin_edges_epochs",
            "transition_prior",
            "duration_prior_strength",
            "emission_prior",
        },
        optional=set(),
        label="model",
    )
    simulation = _mapping(root["simulation"], "simulation")
    _exact_keys(
        simulation,
        required={"seed", "replicates"},
        optional=set(),
        label="simulation",
    )
    evaluation = _mapping(root["evaluation"], "evaluation")
    _exact_keys(
        evaluation,
        required={"phase_bins", "autocorrelation_lags", "replicate_interval"},
        optional=set(),
        label="evaluation",
    )
    output = _mapping(root["output"], "output")
    _exact_keys(output, required={"directory"}, optional=set(), label="output")

    base = source_path.parent
    dataset_directory = Path(_string(dataset["directory"], "dataset.directory"))
    output_directory = Path(_string(output["directory"], "output.directory"))
    if not dataset_directory.is_absolute():
        dataset_directory = (base / dataset_directory).resolve()
    if not output_directory.is_absolute():
        output_directory = (base / output_directory).resolve()

    return TemporalLabConfig(
        experiment_id=_string(root["experiment_id"], "experiment_id"),
        dataset=TemporalDatasetConfig(
            dataset_id=_string(dataset["id"], "dataset.id"),
            directory=dataset_directory,
            training_subjects=_string_tuple(
                dataset["training_subjects"], "dataset.training_subjects"
            ),
            development_subjects=_string_tuple(
                dataset["development_subjects"], "dataset.development_subjects"
            ),
        ),
        model=TemporalModelConfig(
            phase_bins=_integer(model["phase_bins"], "model.phase_bins"),
            duration_bin_edges_epochs=_integer_tuple(
                model["duration_bin_edges_epochs"],
                "model.duration_bin_edges_epochs",
            ),
            transition_prior=_number(
                model["transition_prior"], "model.transition_prior"
            ),
            duration_prior_strength=_number(
                model["duration_prior_strength"], "model.duration_prior_strength"
            ),
            emission_prior=_number(model["emission_prior"], "model.emission_prior"),
        ),
        simulation=TemporalSimulationConfig(
            seed=_integer(simulation["seed"], "simulation.seed"),
            replicates=_integer(simulation["replicates"], "simulation.replicates"),
        ),
        evaluation=TemporalEvaluationConfig(
            phase_bins=_integer(evaluation["phase_bins"], "evaluation.phase_bins"),
            autocorrelation_lags=_integer_tuple(
                evaluation["autocorrelation_lags"],
                "evaluation.autocorrelation_lags",
            ),
            replicate_interval=_number_pair(
                evaluation["replicate_interval"],
                "evaluation.replicate_interval",
            ),
        ),
        output_directory=output_directory,
        source_path=source_path,
    )
