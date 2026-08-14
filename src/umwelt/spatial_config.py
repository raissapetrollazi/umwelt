"""Configuration for the first Umwelt v0.2 spatial laboratory."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from umwelt.errors import ConfigurationError

SPATIAL_CONFIG_SCHEMA_VERSION = "umwelt.spatial-lab.v1"


@dataclass(frozen=True, slots=True)
class SpatialLabConfig:
    experiment_id: str
    dataset_directory: Path
    seed: int
    replicates: int
    output_directory: Path

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", self.experiment_id):
            raise ConfigurationError("Invalid spatial experiment_id.")
        if not 0 <= self.seed <= 2**64 - 1:
            raise ConfigurationError("Spatial seed must be an unsigned 64-bit integer.")
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
    source = Path(path).resolve()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigurationError(f"Invalid spatial configuration: {error}") from error
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version", "experiment_id", "dataset", "simulation", "output"
    }:
        raise ConfigurationError("Spatial configuration has unexpected keys.")
    if raw["schema_version"] != SPATIAL_CONFIG_SCHEMA_VERSION:
        raise ConfigurationError("Unsupported spatial configuration schema.")
    dataset = raw["dataset"]
    simulation = raw["simulation"]
    output = raw["output"]
    if not isinstance(dataset, dict) or set(dataset) != {"directory"}:
        raise ConfigurationError("Spatial dataset config must contain directory.")
    if not isinstance(simulation, dict) or set(simulation) != {"seed", "replicates"}:
        raise ConfigurationError("Spatial simulation config must contain seed and replicates.")
    if not isinstance(output, dict) or set(output) != {"directory"}:
        raise ConfigurationError("Spatial output config must contain directory.")
    base = source.parent
    data_path = Path(dataset["directory"])
    output_path = Path(output["directory"])
    if not data_path.is_absolute():
        data_path = (base / data_path).resolve()
    if not output_path.is_absolute():
        output_path = (base / output_path).resolve()
    return SpatialLabConfig(
        experiment_id=raw["experiment_id"],
        dataset_directory=data_path,
        seed=simulation["seed"],
        replicates=simulation["replicates"],
        output_directory=output_path,
    )
