"""Execution core for the Umwelt v0.2 spatial laboratory."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from umwelt import __version__
from umwelt.datasets.roche_open_field import RocheRecording
from umwelt.errors import DataError
from umwelt.observations import ObservationSource
from umwelt.spatial import Keypoint2D, Pose2D, SpatialFrame
from umwelt.spatial_config import SpatialLabConfig
from umwelt.spatial_evaluation import (
    ROCHE_SPATIAL_EVALUATION_PROTOCOL,
    SpatialComparison,
    compare_spatial_measurements,
    derive_roche_recording_arena,
    measure_spatial_trajectory,
)
from umwelt.spatial_model import fit_spatial_movement_model, generate_spatial_trajectory
from umwelt.spatial_protocol import (
    ROCHE_CONTROL_DEVELOPMENT_SUBJECTS,
    ROCHE_CONTROL_FITTING_SUBJECTS,
    ROCHE_RECORDED_TRAJECTORY_PROTOCOL,
    SpatialTrajectoryProtocol,
)
from umwelt.spatial_reporting import file_sha256, render_spatial_report, write_json


def _git_revision() -> str | None:
    repository = Path(__file__).resolve().parents[2]
    if not (repository / ".git").exists():
        return None
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
            cwd=repository,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _git_dirty() -> bool | None:
    repository = Path(__file__).resolve().parents[2]
    if not (repository / ".git").exists():
        return None
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
            cwd=repository,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(completed.stdout.strip())


def _seed(master: int, subject_id: str, replicate: int) -> int:
    payload = f"{master}|v0.2-spatial|{subject_id}|{replicate}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _apply_recorded_availability_mask(
    recorded_frames: Sequence[SpatialFrame],
    synthetic_frames: Sequence[SpatialFrame],
) -> tuple[SpatialFrame, ...]:
    """Copy only observed bodycentre availability, never recorded coordinates."""

    if len(recorded_frames) != len(synthetic_frames):
        raise DataError(
            "Synthetic and recorded spatial frame grids must have equal size."
        )
    masked: list[SpatialFrame] = []
    for recorded, synthetic in zip(recorded_frames, synthetic_frames, strict=True):
        if recorded.frame_index != synthetic.frame_index:
            raise DataError("Synthetic and recorded spatial frame indices must match.")
        if recorded.pose is None:
            pose = None
        elif recorded.pose.keypoint("bodycentre").point is None:
            pose = Pose2D((Keypoint2D("bodycentre", None, None),))
        else:
            pose = synthetic.pose
        masked.append(
            SpatialFrame(
                context=synthetic.context,
                frame_index=synthetic.frame_index,
                pose=pose,
            )
        )
    return tuple(masked)


def execute_spatial_lab(
    config: SpatialLabConfig,
    recordings: Mapping[str, RocheRecording],
    target: Path,
    *,
    source_provenance: Mapping[str, object],
) -> tuple[str, int, int, int]:
    """Fit, generate, compare, and atomically write compact run artifacts."""

    fitting_steps = []
    fitting_arenas = []
    for subject_id in ROCHE_CONTROL_FITTING_SUBJECTS:
        recording = recordings[subject_id]
        frames = tuple(recording.frames())
        arena = derive_roche_recording_arena(frames)
        fitting_steps.append(
            tuple(ROCHE_RECORDED_TRAJECTORY_PROTOCOL.trajectory_steps(frames))
        )
        fitting_arenas.append(arena)
    model = fit_spatial_movement_model(tuple(fitting_steps), tuple(fitting_arenas))

    synthetic_protocol = SpatialTrajectoryProtocol(
        keypoint_name="bodycentre",
        minimum_likelihood=None,
        required_source=ObservationSource.SYNTHETIC,
        required_coordinate_frame=(
            ROCHE_RECORDED_TRAJECTORY_PROTOCOL.required_coordinate_frame
        ),
    )
    records: list[dict[str, object]] = []
    comparisons: list[tuple[str, SpatialComparison]] = []
    seeds: list[dict[str, object]] = []
    generated = 0
    for subject_id in ROCHE_CONTROL_DEVELOPMENT_SUBJECTS:
        recording = recordings[subject_id]
        frames = tuple(recording.frames())
        arena = derive_roche_recording_arena(frames)
        recorded = measure_spatial_trajectory(
            frames,
            trajectory_protocol=ROCHE_RECORDED_TRAJECTORY_PROTOCOL,
        )
        frame_indices = tuple(frame.frame_index for frame in frames)
        for replicate in range(config.replicates):
            seed = _seed(config.seed, subject_id, replicate)
            generated_frames = generate_spatial_trajectory(
                model,
                arena=arena,
                frame_indices=frame_indices,
                subject_id=subject_id,
                recording_id=f"synthetic:{subject_id}:{replicate}",
                seed=seed,
            )
            synthetic_frames = _apply_recorded_availability_mask(
                frames, generated_frames
            )
            synthetic = measure_spatial_trajectory(
                synthetic_frames,
                trajectory_protocol=synthetic_protocol,
            )
            comparison = compare_spatial_measurements(recorded, synthetic)
            records.append(
                {
                    "subject_id": subject_id,
                    "replicate": replicate,
                    "recorded": recorded.compact_dict(),
                    "synthetic": synthetic.compact_dict(),
                    "comparison": comparison.to_dict(),
                }
            )
            comparisons.append((subject_id, comparison))
            seeds.append(
                {"subject_id": subject_id, "replicate": replicate, "seed": seed}
            )
            generated += 1

    config_payload = json.dumps(
        config.to_dict(), sort_keys=True, separators=(",", ":")
    ).encode()
    config_hash = hashlib.sha256(config_payload).hexdigest()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise DataError(f"Spatial output directory already exists: {target}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        write_json(temporary / "resolved-config.json", config.to_dict())
        write_json(
            temporary / "protocol.json",
            {
                "trajectory": ROCHE_RECORDED_TRAJECTORY_PROTOCOL.to_dict(),
                "evaluation": ROCHE_SPATIAL_EVALUATION_PROTOCOL.to_dict(),
                "fitting_subjects": list(ROCHE_CONTROL_FITTING_SUBJECTS),
                "development_subjects": list(ROCHE_CONTROL_DEVELOPMENT_SUBJECTS),
            },
        )
        write_json(temporary / "model.json", model.to_dict())
        write_json(
            temporary / "seeds.json",
            {"master_seed": config.seed, "derived": seeds},
        )
        with (temporary / "replicate-metrics.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        write_json(
            temporary / "provenance.json",
            {
                "dataset": dict(source_provenance),
                "configuration_sha256": config_hash,
                "software": {
                    "name": "umwelt",
                    "version": __version__,
                    "git_revision": _git_revision(),
                    "git_tracked_worktree_dirty": _git_dirty(),
                },
                "information_categories": [
                    "recorded",
                    "derived",
                    "model",
                    "synthetic",
                ],
                "development_subjects_used_for_fitting": False,
                "synthetic_trajectories_retained": False,
            },
        )
        (temporary / "report.md").write_text(
            render_spatial_report(
                config=config,
                model=model.to_dict(),
                comparisons=comparisons,
            ),
            encoding="utf-8",
        )
        manifest = {}
        for path in sorted(temporary.iterdir()):
            if path.is_file():
                manifest[path.name] = {
                    "size_bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
        write_json(temporary / "artifact-manifest.json", manifest)
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return config_hash, len(records), generated, len(list(target.iterdir()))
