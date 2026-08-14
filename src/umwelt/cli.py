"""Headless command-line interface for Umwelt experiments."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import TextIO

from umwelt import __version__
from umwelt.config import load_experiment_config
from umwelt.datasets.compass import download_compass, load_compass_week, verify_compass
from umwelt.errors import ConfigurationError, UmweltError
from umwelt.experiment import run_experiment
from umwelt.individual_lab import run_individual_variation_lab
from umwelt.spatial_config import load_spatial_lab_config
from umwelt.spatial_lab import run_spatial_lab
from umwelt.temporal_config import load_temporal_lab_config
from umwelt.temporal_lab import run_temporal_lab


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="umwelt",
        description="Run explicit, reproducible animal-behavior experiments.",
    )
    parser.add_argument("--version", action="version", version=f"umwelt {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    data = commands.add_parser("data", help="Manage recorded source datasets.")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    download = data_commands.add_parser("download", help="Download pinned COMPASS files.")
    download.add_argument("--directory", type=Path, default=Path("data/raw/compass"))
    download.add_argument("--all", action="store_true")
    download.add_argument("--repair", action="store_true")
    verify = data_commands.add_parser("verify", help="Verify local COMPASS files.")
    verify.add_argument("--directory", type=Path, default=Path("data/raw/compass"))
    verify.add_argument("--all", action="store_true")

    replay = commands.add_parser("replay", help="Stream fixed recorded observations.")
    replay.add_argument("--config", type=Path, required=True)
    replay.add_argument("--subject", required=True)
    replay.add_argument("--start", type=int, default=0)
    replay.add_argument("--limit", type=int, default=20)
    replay.add_argument("--format", choices=("jsonl", "csv"), default="jsonl")
    replay.add_argument("--omit-gaps", action="store_true")

    run = commands.add_parser("run", help="Run one v0.1 experiment.")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path)

    compare = commands.add_parser("compare", help="Run replicated v0.1 temporal comparison.")
    compare.add_argument("--config", type=Path, required=True)
    compare.add_argument("--output", type=Path)

    individual = commands.add_parser("individual", help="Run v0.1 population-variation comparison.")
    individual.add_argument("--config", type=Path, required=True)
    individual.add_argument("--output", type=Path, required=True)

    spatial = commands.add_parser("spatial", help="Run the replicated v0.2 spatial experiment.")
    spatial.add_argument("--config", type=Path, required=True)
    spatial.add_argument("--output", type=Path)
    return parser


def _print_json(value: object, stream: TextIO) -> None:
    json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=True)
    stream.write("\n")


def _data_download(arguments: argparse.Namespace, output: TextIO) -> int:
    results = download_compass(arguments.directory, include_all=arguments.all, repair=arguments.repair)
    _print_json({"dataset": "compass-zenodo-160344-v1", "directory": str(arguments.directory.resolve()), "files": [asdict(result) for result in results]}, output)
    return 0


def _data_verify(arguments: argparse.Namespace, output: TextIO) -> int:
    results = verify_compass(arguments.directory, include_all=arguments.all)
    valid = all(result.valid for result in results)
    _print_json({"dataset": "compass-zenodo-160344-v1", "directory": str(arguments.directory.resolve()), "valid": valid, "files": [asdict(result) for result in results]}, output)
    return 0 if valid else 1


def _replay_rows(arguments: argparse.Namespace):
    if arguments.start < 0:
        raise ConfigurationError("replay --start must be non-negative.")
    if arguments.limit <= 0:
        raise ConfigurationError("replay --limit must be positive.")
    config = load_experiment_config(arguments.config)
    selected = set(config.dataset.training_subjects) | set(config.dataset.evaluation_subjects)
    if arguments.subject not in selected:
        raise ConfigurationError(f"Subject {arguments.subject} is not selected by the experiment configuration.")
    dataset = load_compass_week(config.dataset.directory, subject_ids=[arguments.subject])
    series = dataset.subject(arguments.subject)
    stop = min(arguments.start + arguments.limit, len(series.timestamps))
    for index in range(arguments.start, stop):
        state = series.states[index]
        activity = series.activities[index]
        if state is None and arguments.omit_gaps:
            continue
        yield {"epoch_index": index, "timestamp": series.timestamps[index].isoformat(sep=" "), "subject_id": series.subject_id, "source": series.source.value, "available": state is not None, "state": state.value if state is not None else None, "activity": activity}


def _replay(arguments: argparse.Namespace, output: TextIO) -> int:
    rows = _replay_rows(arguments)
    if arguments.format == "jsonl":
        output.writelines(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n" for row in rows)
        return 0
    fieldnames = ["epoch_index", "timestamp", "subject_id", "source", "available", "state", "activity"]
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return 0


def _run(arguments: argparse.Namespace, output: TextIO) -> int:
    result = run_experiment(load_experiment_config(arguments.config), output_directory=arguments.output)
    _print_json({"experiment_id": result.experiment_id, "output_directory": str(result.output_directory), "configuration_sha256": result.configuration_sha256, "synthetic_series_count": result.synthetic_series_count, "artifact_count": result.artifact_count}, output)
    return 0


def _compare(arguments: argparse.Namespace, output: TextIO) -> int:
    result = run_temporal_lab(load_temporal_lab_config(arguments.config), output_directory=arguments.output)
    _print_json({"experiment_id": result.experiment_id, "output_directory": str(result.output_directory), "configuration_sha256": result.configuration_sha256, "model_count": result.model_count, "replicate_records": result.replicate_records, "generated_series": result.generated_series, "artifact_count": result.artifact_count}, output)
    return 0


def _individual(arguments: argparse.Namespace, output: TextIO) -> int:
    result = run_individual_variation_lab(load_temporal_lab_config(arguments.config), output_directory=arguments.output)
    _print_json({"experiment_id": result.experiment_id, "output_directory": str(result.output_directory), "configuration_sha256": result.configuration_sha256, "model_count": result.model_count, "replicate_records": result.replicate_records, "generated_series": result.generated_series, "artifact_count": result.artifact_count}, output)
    return 0


def _spatial(arguments: argparse.Namespace, output: TextIO) -> int:
    result = run_spatial_lab(load_spatial_lab_config(arguments.config), output_directory=arguments.output)
    _print_json({"experiment_id": result.experiment_id, "output_directory": str(result.output_directory), "configuration_sha256": result.configuration_sha256, "replicate_records": result.replicate_records, "generated_series": result.generated_series, "artifact_count": result.artifact_count}, output)
    return 0


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None, stderr: TextIO | None = None) -> int:
    output = stdout or sys.stdout
    errors = stderr or sys.stderr
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "data" and arguments.data_command == "download": return _data_download(arguments, output)
        if arguments.command == "data" and arguments.data_command == "verify": return _data_verify(arguments, output)
        if arguments.command == "replay": return _replay(arguments, output)
        if arguments.command == "run": return _run(arguments, output)
        if arguments.command == "compare": return _compare(arguments, output)
        if arguments.command == "individual": return _individual(arguments, output)
        if arguments.command == "spatial": return _spatial(arguments, output)
    except UmweltError as error:
        errors.write(f"error: {error}\n")
        return 2
    raise AssertionError("Unhandled command")
