"""Two-stage, reproducible v0.4 social-model laboratory."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from umwelt import __version__
from umwelt.calms21 import (
    CALMS21_ARCHIVE_MD5,
    Calms21Task1Sequence,
    iter_calms21_task1_sequences,
    scan_calms21_task1_sequence_ids,
)
from umwelt.calms21_audit import verify_calms21_task1_archive
from umwelt.dyad import DyadFrame, measure_dyad_frames
from umwelt.errors import DataError
from umwelt.social_evaluation import (
    SOCIAL_MASTER_SEED,
    SOCIAL_REPLICATES,
    SocialReplicateScore,
    aggregate_social_scores,
    compare_social_measurements,
)
from umwelt.social_model import (
    SOCIAL_MODEL_IDS,
    SocialMovementFit,
    fit_social_movement_model,
    generate_social_dyad,
    social_role_seed,
)
from umwelt.spatial_reporting import file_sha256, write_json


TRAIN_SEQUENCE_COUNT = 70
FITTING_SEQUENCE_COUNT = 56
DEVELOPMENT_SEQUENCE_COUNT = 14
TEST_SEQUENCE_COUNT = 19
SPLIT_SALT = "umwelt-v0.4-development-v1"
PROTOCOL_DOCUMENT = "docs/v0.4-frozen-experiment.md"


@dataclass(frozen=True, slots=True)
class SocialLabResult:
    phase: str
    output_directory: Path
    eligible_pair_count: int
    excluded_pair_count: int
    candidate_preferred: bool | None
    artifact_count: int


def partition_calms21_training_ids(
    sequence_ids: Iterable[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Freeze an outcome-blind 56/14 split from source names and MD5."""

    ids = tuple(sequence_ids)
    if len(ids) != TRAIN_SEQUENCE_COUNT or len(ids) != len(set(ids)):
        raise DataError("CalMS21 training split needs 70 distinct sequences.")
    if any(not value.startswith("task1/train/") for value in ids):
        raise DataError("CalMS21 training split contains a non-training identifier.")
    ordered = tuple(
        sorted(
            ids,
            key=lambda value: hashlib.sha256(
                f"{SPLIT_SALT}|{CALMS21_ARCHIVE_MD5}|{value}".encode("utf-8")
            ).hexdigest(),
        )
    )
    return ordered[:FITTING_SEQUENCE_COUNT], ordered[FITTING_SEQUENCE_COUNT:]


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _software_provenance() -> dict[str, object]:
    repository = _repository_root()
    revision: str | None = None
    dirty: bool | None = None
    if (repository / ".git").exists():
        try:
            revision_result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=3,
                cwd=repository,
            )
            status_result = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                check=True,
                capture_output=True,
                text=True,
                timeout=3,
                cwd=repository,
            )
            revision = revision_result.stdout.strip() or None
            dirty = bool(status_result.stdout.strip())
        except (OSError, subprocess.SubprocessError):
            pass
    return {
        "name": "umwelt",
        "version": __version__,
        "git_revision": revision,
        "git_tracked_worktree_dirty": dirty,
    }


def _protocol_hash() -> str:
    path = _repository_root() / PROTOCOL_DOCUMENT
    if not path.is_file():
        raise DataError("Frozen social protocol document is missing.")
    return file_sha256(path)


@contextmanager
def _new_output_directory(target: Path) -> Iterator[Path]:
    if target.exists():
        raise DataError(f"Social output directory already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        yield temporary
        if target.exists():
            raise DataError(f"Social output directory already exists: {target}")
        temporary.rename(target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _manifest(directory: Path) -> int:
    artifacts = [
        {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(directory.iterdir())
        if path.is_file() and path.name != "artifact-manifest.json"
    ]
    write_json(
        directory / "artifact-manifest.json",
        {"schema_version": "umwelt.artifacts.v1", "artifacts": artifacts},
    )
    return len(artifacts) + 1


def _verify_development_artifacts(directory: Path) -> dict[str, object]:
    manifest_path = directory / "artifact-manifest.json"
    if not manifest_path.is_file():
        raise DataError("Social development artifact manifest is missing.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("artifacts")
    if not isinstance(entries, list):
        raise DataError("Social development artifact manifest is invalid.")
    for entry in entries:
        if not isinstance(entry, dict):
            raise DataError("Social development artifact entry is invalid.")
        name, expected_hash = entry.get("name"), entry.get("sha256")
        if not isinstance(name, str) or not isinstance(expected_hash, str):
            raise DataError("Social development artifact entry is invalid.")
        if name != Path(name).name:
            raise DataError("Social development artifact path is invalid.")
        path = directory / name
        if not path.is_file() or file_sha256(path) != expected_hash:
            raise DataError("Social development artifact hash differs.")
    required = {"model-fit.json", "provenance.json", "split.json"}
    if not required.issubset({entry["name"] for entry in entries}):
        raise DataError("Social development artifacts are incomplete.")
    provenance = json.loads((directory / "provenance.json").read_text("utf-8"))
    if not isinstance(provenance, dict):
        raise DataError("Social development provenance is invalid.")
    return cast(dict[str, object], provenance)


def _fitting_frames(
    archive_path: Path, fitting_ids: frozenset[str]
) -> Iterator[Iterator[DyadFrame]]:
    seen: set[str] = set()
    for sequence in iter_calms21_task1_sequences(archive_path):
        if sequence.sequence_id in fitting_ids:
            seen.add(sequence.sequence_id)
            yield sequence.frames()
    if seen != fitting_ids:
        raise DataError("Some selected fitting sequences are absent from CalMS21.")


def _score_split(
    sequences: Iterable[Calms21Task1Sequence],
    *,
    selected_ids: frozenset[str] | None,
    fit: SocialMovementFit,
) -> tuple[
    tuple[dict[str, object], ...],
    tuple[SocialReplicateScore, ...],
    tuple[dict[str, str], ...],
    tuple[str, ...],
]:
    recorded_summaries: list[dict[str, object]] = []
    scores: list[SocialReplicateScore] = []
    exclusions: list[dict[str, str]] = []
    eligible_ids: list[str] = []
    seen: set[str] = set()

    for sequence in sequences:
        pair_id = sequence.sequence_id
        if selected_ids is not None and pair_id not in selected_ids:
            continue
        if pair_id in seen:
            raise DataError("Social evaluation contains a duplicate pair.")
        seen.add(pair_id)
        frames = tuple(sequence.frames())
        if any(position is None for position in frames[0].neck_positions()):
            exclusions.append(
                {"pair_id": pair_id, "reason": "first pair position unavailable"}
            )
            continue
        reference = measure_dyad_frames(frames)
        recorded_summaries.append(reference.compact_dict())
        eligible_ids.append(pair_id)
        for replicate in range(SOCIAL_REPLICATES):
            resident_seed = social_role_seed(
                SOCIAL_MASTER_SEED, pair_id, replicate, "resident"
            )
            intruder_seed = social_role_seed(
                SOCIAL_MASTER_SEED, pair_id, replicate, "intruder"
            )
            for model_id in SOCIAL_MODEL_IDS:
                generated = generate_social_dyad(
                    fit,
                    frames,
                    model_id=model_id,
                    master_seed=SOCIAL_MASTER_SEED,
                    replicate=replicate,
                )
                synthetic = measure_dyad_frames(generated)
                scores.append(
                    compare_social_measurements(
                        reference,
                        synthetic,
                        model_id=model_id,
                        replicate=replicate,
                        resident_seed=resident_seed,
                        intruder_seed=intruder_seed,
                    )
                )
    if selected_ids is not None and seen != selected_ids:
        raise DataError("Some selected development sequences are absent.")
    if not eligible_ids:
        raise DataError("Social evaluation has no eligible pair sequences.")
    return (
        tuple(recorded_summaries),
        tuple(scores),
        tuple(exclusions),
        tuple(eligible_ids),
    )


def _write_split_results(
    directory: Path,
    *,
    phase: str,
    recorded: tuple[dict[str, object], ...],
    scores: tuple[SocialReplicateScore, ...],
    excluded: tuple[dict[str, str], ...],
    eligible_ids: tuple[str, ...],
) -> dict[str, object]:
    with (directory / "recorded-metrics.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        for item in recorded:
            stream.write(json.dumps(item, sort_keys=True) + "\n")
    with (directory / "replicate-metrics.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        for score in scores:
            stream.write(json.dumps(score.to_dict(), sort_keys=True) + "\n")
    comparison = aggregate_social_scores(scores, eligible_pair_ids=eligible_ids)
    write_json(directory / "comparison.json", comparison)
    write_json(directory / "excluded-pairs.json", list(excluded))
    write_json(directory / "eligible-pairs.json", list(eligible_ids))
    write_json(
        directory / "protocol.json",
        {
            "phase": phase,
            "document": PROTOCOL_DOCUMENT,
            "document_sha256": _protocol_hash(),
            "master_seed": SOCIAL_MASTER_SEED,
            "replicates_per_model_pair": SOCIAL_REPLICATES,
            "models": list(SOCIAL_MODEL_IDS),
            "unit_of_evaluation": "recorded-pair-sequence",
            "metric_ids": [
                "pair-distance-empirical-1-wasserstein-px",
                "adjacent-signed-pair-distance-change-empirical-1-wasserstein-px",
            ],
            "preference_rule": {
                "distance_mean_max_ratio": 0.95,
                "signed_change_mean_max_ratio": 1.05,
                "minimum_pair_win_fraction": 0.60,
            },
        },
    )
    return comparison


def _report_markdown(
    phase: str,
    comparison: dict[str, object],
    excluded: tuple[dict[str, str], ...],
) -> str:
    means = cast(
        dict[str, float | None], comparison["equal_pair_mean_distance_wasserstein_px"]
    )
    change = cast(
        dict[str, float | None],
        comparison["equal_pair_mean_signed_change_wasserstein_px"],
    )
    criteria = cast(dict[str, bool | None], comparison["preference_criteria"])
    lines = [
        f"# v0.4 {phase} social comparison",
        "",
        "The unit is a recorded pair sequence; eight synthetic replicates are averaged within each pair.",
        "This describes image-space distance, not social intention or biological validity.",
        "",
        f"Eligible pairs: {comparison['eligible_pair_count']}; excluded pairs: {len(excluded)}.",
        "",
        "| Model | Mean pair-distance W1 (px) | Mean signed-change W1 (px) |",
        "| --- | ---: | ---: |",
    ]
    for model_id in SOCIAL_MODEL_IDS:
        lines.append(f"| {model_id} | {means[model_id]} | {change[model_id]} |")
    lines.extend(
        [
            "",
            f"S1 distance wins: {comparison['candidate_distance_win_count']} / {comparison['eligible_pair_count']} (required {comparison['required_distance_win_count']}).",
            "",
            f"Distance criterion: {criteria['distance_at_least_5_percent_better']}.",
            f"Signed-change criterion: {criteria['signed_change_no_more_than_5_percent_worse']}.",
            f"Pair-win criterion: {criteria['distance_better_on_at_least_60_percent_of_pairs']}.",
            f"S1 preferred: {comparison['candidate_preferred']}.",
            "",
            "The image boundaries are computational reflection limits, not measured cage walls.",
            "The source does not expose global intruder identity across recordings.",
            "See comparison.json and replicate-metrics.jsonl for all pair results and exposures.",
            "",
        ]
    )
    return "\n".join(lines)


def run_social_development(
    archive_path: str | Path, *, output_directory: str | Path
) -> SocialLabResult:
    """Fit on 56 source-train sequences and score 14 development pairs."""

    target = Path(output_directory).resolve()
    if target.exists():
        raise DataError(f"Social output directory already exists: {target}")
    archive = Path(archive_path)
    source = verify_calms21_task1_archive(archive)
    software = _software_provenance()
    if (
        software["git_revision"] is None
        or software["git_tracked_worktree_dirty"] is True
    ):
        raise DataError("Commit tracked social code before the canonical run.")
    train_ids = scan_calms21_task1_sequence_ids(archive, split="train")
    fitting_ids, development_ids = partition_calms21_training_ids(train_ids)
    fit = fit_social_movement_model(_fitting_frames(archive, frozenset(fitting_ids)))
    recorded, scores, excluded, eligible = _score_split(
        iter_calms21_task1_sequences(archive),
        selected_ids=frozenset(development_ids),
        fit=fit,
    )
    with _new_output_directory(target) as temporary:
        write_json(temporary / "model-fit.json", fit.to_dict())
        write_json(
            temporary / "split.json",
            {
                "split_salt": SPLIT_SALT,
                "fitting_sequence_ids": list(fitting_ids),
                "development_sequence_ids": list(development_ids),
            },
        )
        write_json(
            temporary / "provenance.json",
            {
                "source": source,
                "source_license": "CC-BY-NC-SA-2.0 as linked by dataset paper",
                "software": software,
                "frozen_protocol_document_sha256": _protocol_hash(),
                "test_pose_outcomes_used": False,
            },
        )
        comparison = _write_split_results(
            temporary,
            phase="development",
            recorded=recorded,
            scores=scores,
            excluded=excluded,
            eligible_ids=eligible,
        )
        (temporary / "report.md").write_text(
            _report_markdown("development", comparison, excluded), encoding="utf-8"
        )
        artifact_count = _manifest(temporary)
    return SocialLabResult(
        "development",
        target,
        len(eligible),
        len(excluded),
        cast(bool | None, comparison["candidate_preferred"]),
        artifact_count,
    )


def run_social_test(
    archive_path: str | Path,
    *,
    development_directory: str | Path,
    output_directory: str | Path,
) -> SocialLabResult:
    """Apply the unchanged development fit to held-out source-test pairs once."""

    target = Path(output_directory).resolve()
    if target.exists():
        raise DataError(f"Social output directory already exists: {target}")
    archive = Path(archive_path)
    source = verify_calms21_task1_archive(archive)
    development = Path(development_directory).resolve()
    previous_provenance = _verify_development_artifacts(development)
    previous_source = previous_provenance.get("source")
    if (
        not isinstance(previous_source, dict)
        or previous_source.get("archive_sha256") != source["archive_sha256"]
    ):
        raise DataError("Held-out social run requires the same verified archive.")
    if previous_provenance.get("frozen_protocol_document_sha256") != _protocol_hash():
        raise DataError("Frozen social protocol changed after development.")
    software = _software_provenance()
    previous_software = previous_provenance.get("software")
    if not isinstance(previous_software, dict) or (
        software["git_revision"] != previous_software.get("git_revision")
        or software["git_tracked_worktree_dirty"] is True
        or previous_software.get("git_tracked_worktree_dirty") is True
    ):
        raise DataError("Held-out social run requires the frozen code revision.")
    fit_payload = json.loads((development / "model-fit.json").read_text("utf-8"))
    if not isinstance(fit_payload, dict):
        raise DataError("Social development model artifact is invalid.")
    fit = SocialMovementFit.from_dict(fit_payload)
    recorded, scores, excluded, eligible = _score_split(
        iter_calms21_task1_sequences(archive, split="test"),
        selected_ids=None,
        fit=fit,
    )
    if len(eligible) + len(excluded) != TEST_SEQUENCE_COUNT:
        raise DataError("Held-out CalMS21 Task 1 does not have 19 sequences.")
    with _new_output_directory(target) as temporary:
        write_json(
            temporary / "provenance.json",
            {
                "source": source,
                "source_license": "CC-BY-NC-SA-2.0 as linked by dataset paper",
                "software": software,
                "frozen_protocol_document_sha256": _protocol_hash(),
                "development_directory": str(development),
                "development_manifest_sha256": file_sha256(
                    development / "artifact-manifest.json"
                ),
                "held_out_pose_outcomes_used_for_fitting": False,
                "held_out_pose_outcomes_used_for_scoring": True,
            },
        )
        comparison = _write_split_results(
            temporary,
            phase="held-out-test",
            recorded=recorded,
            scores=scores,
            excluded=excluded,
            eligible_ids=eligible,
        )
        (temporary / "report.md").write_text(
            _report_markdown("held-out test", comparison, excluded),
            encoding="utf-8",
        )
        artifact_count = _manifest(temporary)
    return SocialLabResult(
        "held-out-test",
        target,
        len(eligible),
        len(excluded),
        cast(bool | None, comparison["candidate_preferred"]),
        artifact_count,
    )
