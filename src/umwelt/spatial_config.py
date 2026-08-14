"""Configuration for the first Umwelt v0.2 spatial laboratory."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from umwelt.errors import ConfigurationError

SPATIAL_CONFIG_SCHEMA_VERSION = "umwelt.spatial-lab.v1"


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{label} must be a JSON object.")
    return value


def _exact_keys(value: Mapping[str, Any], *, required: set[str], label: str) -> None:
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required)
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


@dataclass(frozen=True, slots=True)
class SpatialLabConfig:
    experiment_id: str
    dataset_directory: Path
    seed: int
    replicates: int
    output_directory: Path

    def __post_init__(self) -> None:
        if not isinstance(self.experiment_id, str) or not re.fullmatch(
            r"[a-z0-9][a-z0-9._-]*", self.experiment_id
        ):
            raise ConfigurationError("Invalid spatial experiment_id.")
        if not isinstance(self.dataset_directory, Path) or not isinstance(
            self.output_directory, Path
        ):
            raise ConfigurationError("Spatial directories must be Path values.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ConfigurationError("Spatial seed must be an integer.")
        if not 0 <= self.seed <= 2**64 - 1:
            raise ConfigurationError("Spatial seed must be an unsigned 64-bit integer.")
        if isinstance(self.replicates, bool) or not isinstance(self.replicates, int):
            raise ConfigurationError("Spatial replicates must be an integer.")
        if not 1 <= self.replicates <= 100:
            raise ConfigurationError("Spatial replicates must be between 1 and 100.")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SPATIAL_CONFIG_SCHEMA_VERSION,
            "experiment_id": self.experiment_id,
            "dataset": {"directory": str(self.dataset_directory)},
            "simulation": {"seed": self.seed, "replicates": self.replicates},
            "output": {"directory": str(self.output_directory)},
        }


def load_spatial_lab_config(path: str | Path) -> SpatialLabConfig:
    """Load a strict JSON contract and resolve paths relative to its file."""

    source = Path(path).resolve()
    try:
        decoded: Any = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigurationError(f"Invalid spatial configuration: {error}") from error
    raw = _mapping(decoded, "spatial configuration")
    _exact_keys(
        raw,
        required={
            "schema_version",
            "experiment_id",
            "dataset",
            "simulation",
            "output",
        },
        label="spatial configuration",
    )
    if (
        _string(raw["schema_version"], "schema_version")
        != SPATIAL_CONFIG_SCHEMA_VERSION
    ):
        raise ConfigurationError("Unsupported spatial configuration schema.")
    dataset = _mapping(raw["dataset"], "dataset")
    simulation = _mapping(raw["simulation"], "simulation")
    output = _mapping(raw["output"], "output")
    _exact_keys(dataset, required={"directory"}, label="dataset")
    _exact_keys(
        simulation,
        required={"seed", "replicates"},
        label="simulation",
    )
    _exact_keys(output, required={"directory"}, label="output")
    base = source.parent
    data_path = Path(_string(dataset["directory"], "dataset.directory"))
    output_path = Path(_string(output["directory"], "output.directory"))
    if not data_path.is_absolute():
        data_path = (base / data_path).resolve()
    if not output_path.is_absolute():
        output_path = (base / output_path).resolve()
    return SpatialLabConfig(
        experiment_id=_string(raw["experiment_id"], "experiment_id"),
        dataset_directory=data_path,
        seed=_integer(simulation["seed"], "simulation.seed"),
        replicates=_integer(simulation["replicates"], "simulation.replicates"),
        output_directory=output_path,
    )
