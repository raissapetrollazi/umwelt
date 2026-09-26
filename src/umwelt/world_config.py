"""Strict configuration for the first Umwelt v0.3 behavioral-world run."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from umwelt.errors import ConfigurationError


WORLD_CONFIG_SCHEMA_VERSION = "umwelt.world-lab.v1"


def _object(value: Any, *, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{label} must be a JSON object.")
    missing = sorted(keys - set(value))
    unknown = sorted(set(value) - keys)
    if missing or unknown:
        raise ConfigurationError(
            f"{label} keys do not match the schema: missing={missing}, unknown={unknown}."
        )
    return value


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{label} must be a non-empty string.")
    return value


@dataclass(frozen=True, slots=True)
class WorldLabConfig:
    experiment_id: str
    dataset_directory: Path
    seed: int
    replicates: int
    output_directory: Path

    def __post_init__(self) -> None:
        if not isinstance(self.experiment_id, str) or not re.fullmatch(
            r"[a-z0-9][a-z0-9._-]*", self.experiment_id
        ):
            raise ConfigurationError("Invalid world experiment_id.")
        if not isinstance(self.dataset_directory, Path) or not isinstance(
            self.output_directory, Path
        ):
            raise ConfigurationError("World directories must be Path values.")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 2**64
        ):
            raise ConfigurationError("World seed must be an unsigned 64-bit integer.")
        if (
            isinstance(self.replicates, bool)
            or not isinstance(self.replicates, int)
            or not 1 <= self.replicates <= 100
        ):
            raise ConfigurationError("World replicates must be between 1 and 100.")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": WORLD_CONFIG_SCHEMA_VERSION,
            "experiment_id": self.experiment_id,
            "dataset": {"directory": str(self.dataset_directory)},
            "simulation": {"seed": self.seed, "replicates": self.replicates},
            "output": {"directory": str(self.output_directory)},
        }


def load_world_lab_config(path: str | Path) -> WorldLabConfig:
    """Resolve only the user-facing paths, seed, and replicate count."""

    source = Path(path).resolve()
    try:
        decoded: Any = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigurationError(f"Invalid world configuration: {error}") from error
    raw = _object(
        decoded,
        keys={"schema_version", "experiment_id", "dataset", "simulation", "output"},
        label="world configuration",
    )
    if (
        _nonempty(raw["schema_version"], "schema_version")
        != WORLD_CONFIG_SCHEMA_VERSION
    ):
        raise ConfigurationError("Unsupported world configuration schema.")
    dataset = _object(raw["dataset"], keys={"directory"}, label="dataset")
    simulation = _object(
        raw["simulation"], keys={"seed", "replicates"}, label="simulation"
    )
    output = _object(raw["output"], keys={"directory"}, label="output")
    dataset_directory = Path(_nonempty(dataset["directory"], "dataset.directory"))
    output_directory = Path(_nonempty(output["directory"], "output.directory"))
    if not dataset_directory.is_absolute():
        dataset_directory = (source.parent / dataset_directory).resolve()
    if not output_directory.is_absolute():
        output_directory = (source.parent / output_directory).resolve()
    return WorldLabConfig(
        experiment_id=_nonempty(raw["experiment_id"], "experiment_id"),
        dataset_directory=dataset_directory,
        seed=simulation["seed"],
        replicates=simulation["replicates"],
        output_directory=output_directory,
    )
