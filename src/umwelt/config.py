"""Validated JSON configuration for reproducible Umwelt experiments."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from umwelt.datasets.compass import COMPASS_DATASET_ID, COMPASS_SUBJECT_IDS
from umwelt.errors import ConfigurationError
from umwelt.models.circadian_markov import MODEL_TYPE

CONFIG_SCHEMA_VERSION = "umwelt.experiment.v1"


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{label} must be a JSON object.")
    return value


def _exact_keys(
    value: Mapping[str, Any], *, required: set[str], optional: set[str], label: str
) -> None:
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required - optional)
    if missing:
        raise ConfigurationError(f"{label} is missing keys: {', '.join(missing)}")
    if unknown:
        raise ConfigurationError(f"{label} has unknown keys: {', '.join(unknown)}")


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{label} must be a non-empty string.")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{label} must be an integer.")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{label} must be a number.")
    return float(value)


def _string_tuple(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigurationError(f"{label} must be a non-empty JSON array.")
    result = tuple(_string(item, f"{label} item") for item in value)
    if len(result) != len(set(result)):
        raise ConfigurationError(f"{label} must not contain duplicates.")
    return result


def _integer_tuple(value: Any, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigurationError(f"{label} must be a non-empty JSON array.")
    result = tuple(_integer(item, f"{label} item") for item in value)
    if len(result) != len(set(result)):
        raise ConfigurationError(f"{label} must not contain duplicates.")
    return result


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Recorded dataset selection and held-out subject split."""

    dataset_id: str
    directory: Path
    training_subjects: tuple[str, ...]
    evaluation_subjects: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Configuration for the explicit v0.1 baseline model."""

    model_type: str
    phase_bins: int
    transition_prior: float
    emission_prior: float


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Seed and replication settings for synthetic generation."""

    seed: int
    replicates: int


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Configuration for descriptive comparison metrics."""

    phase_bins: int
    autocorrelation_lags: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Complete and validated v0.1 experiment configuration."""

    experiment_id: str
    dataset: DatasetConfig
    model: ModelConfig
    simulation: SimulationConfig
    evaluation: EvaluationConfig
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
                f"v0.1 requires dataset id {COMPASS_DATASET_ID}."
            )
        available = set(COMPASS_SUBJECT_IDS)
        selected = set(self.dataset.training_subjects) | set(
            self.dataset.evaluation_subjects
        )
        unknown = sorted(selected - available, key=int)
        if unknown:
            raise ConfigurationError(
                "Unknown COMPASS subject identifiers: " + ", ".join(unknown)
            )
        overlap = set(self.dataset.training_subjects) & set(
            self.dataset.evaluation_subjects
        )
        if overlap:
            raise ConfigurationError(
                "Training and evaluation subjects must be disjoint: "
                + ", ".join(sorted(overlap, key=int))
            )
        if self.model.model_type != MODEL_TYPE:
            raise ConfigurationError(f"v0.1 requires model type {MODEL_TYPE}.")
        if not 1 <= self.model.phase_bins <= 1_440:
            raise ConfigurationError("model.phase_bins must be between 1 and 1440.")
        if self.model.transition_prior <= 0 or self.model.emission_prior <= 0:
            raise ConfigurationError("Model priors must be positive.")
        if not 0 <= self.simulation.seed <= 2**64 - 1:
            raise ConfigurationError("simulation.seed must be a 64-bit unsigned integer.")
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

    def to_dict(self) -> dict[str, object]:
        """Return a canonical JSON-compatible representation."""

        return {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "experiment_id": self.experiment_id,
            "dataset": {
                "id": self.dataset.dataset_id,
                "directory": str(self.dataset.directory),
                "training_subjects": list(self.dataset.training_subjects),
                "evaluation_subjects": list(self.dataset.evaluation_subjects),
            },
            "model": {
                "type": self.model.model_type,
                "phase_bins": self.model.phase_bins,
                "transition_prior": self.model.transition_prior,
                "emission_prior": self.model.emission_prior,
            },
            "simulation": {
                "seed": self.simulation.seed,
                "replicates": self.simulation.replicates,
            },
            "evaluation": {
                "phase_bins": self.evaluation.phase_bins,
                "autocorrelation_lags": list(
                    self.evaluation.autocorrelation_lags
                ),
            },
            "output": {"directory": str(self.output_directory)},
        }


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load, strictly validate, and resolve one JSON experiment configuration."""

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
    if root["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise ConfigurationError(
            f"Unsupported configuration schema: {root['schema_version']!r}"
        )

    dataset = _mapping(root["dataset"], "dataset")
    _exact_keys(
        dataset,
        required={"id", "directory", "training_subjects", "evaluation_subjects"},
        optional=set(),
        label="dataset",
    )
    model = _mapping(root["model"], "model")
    _exact_keys(
        model,
        required={"type", "phase_bins", "transition_prior", "emission_prior"},
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
        required={"phase_bins", "autocorrelation_lags"},
        optional=set(),
        label="evaluation",
    )
    output = _mapping(root["output"], "output")
    _exact_keys(
        output,
        required={"directory"},
        optional=set(),
        label="output",
    )

    base = source_path.parent
    dataset_directory = Path(_string(dataset["directory"], "dataset.directory"))
    output_directory = Path(_string(output["directory"], "output.directory"))
    if not dataset_directory.is_absolute():
        dataset_directory = (base / dataset_directory).resolve()
    if not output_directory.is_absolute():
        output_directory = (base / output_directory).resolve()

    return ExperimentConfig(
        experiment_id=_string(root["experiment_id"], "experiment_id"),
        dataset=DatasetConfig(
            dataset_id=_string(dataset["id"], "dataset.id"),
            directory=dataset_directory,
            training_subjects=_string_tuple(
                dataset["training_subjects"], "dataset.training_subjects"
            ),
            evaluation_subjects=_string_tuple(
                dataset["evaluation_subjects"], "dataset.evaluation_subjects"
            ),
        ),
        model=ModelConfig(
            model_type=_string(model["type"], "model.type"),
            phase_bins=_integer(model["phase_bins"], "model.phase_bins"),
            transition_prior=_number(
                model["transition_prior"], "model.transition_prior"
            ),
            emission_prior=_number(model["emission_prior"], "model.emission_prior"),
        ),
        simulation=SimulationConfig(
            seed=_integer(simulation["seed"], "simulation.seed"),
            replicates=_integer(simulation["replicates"], "simulation.replicates"),
        ),
        evaluation=EvaluationConfig(
            phase_bins=_integer(evaluation["phase_bins"], "evaluation.phase_bins"),
            autocorrelation_lags=_integer_tuple(
                evaluation["autocorrelation_lags"],
                "evaluation.autocorrelation_lags",
            ),
        ),
        output_directory=output_directory,
        source_path=source_path,
    )
