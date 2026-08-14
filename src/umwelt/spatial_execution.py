"""Execution core for the Umwelt v0.2 spatial laboratory."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from umwelt.observations import ObservationSource
from umwelt.spatial_config import SpatialLabConfig
from umwelt.spatial_evaluation import (
    ROCHE_SPATIAL_EVALUATION_PROTOCOL,
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


def _seed(master: int, subject_id: str, replicate: int) -> int:
    payload = f"{master}|v0.2-spatial|{subject_id}|{replicate}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def execute_spatial_lab(config: SpatialLabConfig, recordings: dict[str, object], target: Path) -> tuple[str, int, int, int]:
    fitting_steps = []
    fitting_arenas = []
    for subject_id in ROCHE_CONTROL_FITTING_SUBJECTS:
        recording = recordings[subject_id]
        frames = tuple(recording.frames())
        arena = derive_roche_recording_arena(frames)
        fitting_steps.append(tuple(ROCHE_RECORDED_TRAJECTORY_PROTOCOL.trajectory_steps(frames)))
        fitting_arenas.append(arena)
    model = fit_spatial_movement_model(tuple(fitting_steps), tuple(fitting_arenas))

    synthetic_protocol = SpatialTrajectoryProtocol(
        keypoint_name="bodycentre",
        minimum_likelihood=None,
        required_source=ObservationSource.SYNTHETIC,
        required_coordinate_frame=ROCHE_RECORDED_TRAJECTORY_PROTOCOL.required_coordinate_frame,
    )
    records: list[dict[str, object]] = []
    seeds: list[dict[str, object]] = []
    generated = 0
    for subject_id in ROCHE_CONTROL_DEVELOPMENT_SUBJECTS:
        recording = recordings[subject_id]
        frames = tuple(recording.frames())
        arena = derive_roche_recording_arena(frames)
        recorded = measure_spatial_trajectory(
            frames,
            arena=arena,
            trajectory_protocol=ROCHE_RECORDED_TRAJECTORY_PROTOCOL,
        )
        frame_indices = tuple(frame.frame_index for frame in frames)
        for replicate in range(config.replicates):
            seed = _seed(config.seed, subject_id, replicate)
            synthetic_frames = generate_spatial_trajectory(
                model,
                arena=arena,
                frame_indices=frame_indices,
                subject_id=subject_id,
                recording_id=f"synthetic:{subject_id}:{replicate}",
                seed=seed,
            )
            synthetic = measure_spatial_trajectory(
                synthetic_frames,
                arena=arena,
                trajectory_protocol=synthetic_protocol,
            )
            records.append({
                "subject_id": subject_id,
                "replicate": replicate,
                "recorded": recorded.compact_dict(),
                "synthetic": synthetic.compact_dict(),
                "comparison": compare_spatial_measurements(recorded, synthetic),
            })
            seeds.append({"subject_id": subject_id, "replicate": replicate, "seed": seed})
            generated += 1

    config_payload = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    config_hash = hashlib.sha256(config_payload).hexdigest()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Spatial output directory already exists: {target}")
    temp = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        write_json(temp / "resolved-config.json", config.to_dict())
        write_json(temp / "protocol.json", {
            "trajectory": ROCHE_RECORDED_TRAJECTORY_PROTOCOL.to_dict(),
            "evaluation": ROCHE_SPATIAL_EVALUATION_PROTOCOL.to_dict(),
            "fitting_subjects": list(ROCHE_CONTROL_FITTING_SUBJECTS),
            "development_subjects": list(ROCHE_CONTROL_DEVELOPMENT_SUBJECTS),
        })
        write_json(temp / "model.json", model.to_dict())
        write_json(temp / "seeds.json", {"master_seed": config.seed, "derived": seeds})
        with (temp / "replicate-metrics.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        write_json(temp / "provenance.json", {
            "dataset_id": "roche-open-field-zenodo-8188683-v1",
            "configuration_sha256": config_hash,
            "information_categories": ["recorded", "derived", "model", "synthetic"],
            "synthetic_trajectories_retained": False,
        })
        (temp / "report.md").write_text(render_spatial_report(config=config, model=model.to_dict(), records=records), encoding="utf-8")
        manifest = {}
        for path in sorted(temp.iterdir()):
            if path.is_file():
                manifest[path.name] = {"size_bytes": path.stat().st_size, "sha256": file_sha256(path)}
        write_json(temp / "artifact-manifest.json", manifest)
        os.replace(temp, target)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return config_hash, len(records), generated, len(list(target.iterdir()))
