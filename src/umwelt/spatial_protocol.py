"""Pre-result protocol facts for the first Umwelt v0.2 spatial experiment."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from math import isfinite

from umwelt.datasets.roche_open_field import ROCHE_COORDINATE_FRAME, ROCHE_METADATA_MD5
from umwelt.errors import DataError
from umwelt.observations import ObservationSource
from umwelt.spatial import CoordinateFrame, SpatialContext, SpatialFrame
from umwelt.trajectory import (
    PositionRule,
    PositionSample,
    TrajectoryStep,
    iter_trajectory_steps,
    keypoint_position_samples,
)

ROCHE_CONTROL_SUBJECT_IDS = (
    "16459-67049",
    "16459-67053",
    "16459-67060",
    "16459-67064",
    "16459-67067",
    "16459-67071",
    "16459-67074",
    "16459-67078",
)
ROCHE_CONTROL_SPLIT_METHOD = "sha256-pinned-metadata-ranking-v1"
ROCHE_CONTROL_SPLIT_HASH_PREFIX = f"umwelt-v0.2-control-split-v1|{ROCHE_METADATA_MD5}"
ROCHE_CONTROL_DEVELOPMENT_COUNT = 2

ADJACENT_ACCEPTED_DISPLACEMENT_POLICY = "adjacent-frames-with-accepted-endpoints"
CONSECUTIVE_NONSTATIONARY_TURNING_POLICY = (
    "two-consecutive-nonstationary-valid-movements"
)
RETAIN_ZERO_DISPLACEMENT_POLICY = "retain-and-break-turning"


def _split_digest(subject_id: str) -> str:
    payload = f"{ROCHE_CONTROL_SPLIT_HASH_PREFIX}|{subject_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def derive_roche_control_split() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the outcome-blind fitting/development assignment.

    The two smallest SHA-256 digests become development animals. Role tuples
    retain source-metadata order so later iteration order is reproducible.
    """

    ranked = sorted(
        ROCHE_CONTROL_SUBJECT_IDS,
        key=lambda subject_id: (_split_digest(subject_id), subject_id),
    )
    development_set = set(ranked[:ROCHE_CONTROL_DEVELOPMENT_COUNT])
    fitting = tuple(
        subject_id
        for subject_id in ROCHE_CONTROL_SUBJECT_IDS
        if subject_id not in development_set
    )
    development = tuple(
        subject_id
        for subject_id in ROCHE_CONTROL_SUBJECT_IDS
        if subject_id in development_set
    )
    return fitting, development


ROCHE_CONTROL_FITTING_SUBJECTS, ROCHE_CONTROL_DEVELOPMENT_SUBJECTS = (
    derive_roche_control_split()
)


@dataclass(frozen=True, slots=True)
class SpatialTrajectoryProtocol:
    """Executable position and continuity rules declared before model results."""

    keypoint_name: str
    minimum_likelihood: float | None
    required_source: ObservationSource
    required_coordinate_frame: CoordinateFrame

    def __post_init__(self) -> None:
        if not self.keypoint_name.strip():
            raise DataError("A spatial protocol keypoint name must be non-empty.")
        if self.minimum_likelihood is not None and (
            isinstance(self.minimum_likelihood, bool)
            or not isfinite(self.minimum_likelihood)
            or not 0.0 <= self.minimum_likelihood <= 1.0
        ):
            raise DataError(
                "A spatial protocol likelihood threshold must be between 0 and 1."
            )
        if not isinstance(self.required_source, ObservationSource):
            raise DataError("A spatial protocol must declare an observation source.")
        if not isinstance(self.required_coordinate_frame, CoordinateFrame):
            raise DataError("A spatial protocol must declare a coordinate frame.")

    def _validate_context(self, context: SpatialContext) -> None:
        if context.source is not self.required_source:
            raise DataError(
                "Spatial protocol input does not match the required observation source."
            )
        if context.coordinate_frame != self.required_coordinate_frame:
            raise DataError(
                "Spatial protocol input does not match the required coordinate frame."
            )

    def position_rule(self, context: SpatialContext) -> PositionRule:
        """Bind the declared position choice to one source context."""

        self._validate_context(context)
        return PositionRule(
            context=context,
            keypoint_name=self.keypoint_name,
            minimum_likelihood=self.minimum_likelihood,
        )

    def position_samples(
        self, frames: Iterable[SpatialFrame]
    ) -> Iterator[PositionSample]:
        """Extract positions without adding interpolation or smoothing."""

        def validated_frames() -> Iterator[SpatialFrame]:
            for frame in frames:
                self._validate_context(frame.context)
                yield frame

        yield from keypoint_position_samples(
            validated_frames(),
            keypoint_name=self.keypoint_name,
            minimum_likelihood=self.minimum_likelihood,
        )

    def trajectory_steps(
        self, frames: Iterable[SpatialFrame]
    ) -> Iterator[TrajectoryStep]:
        """Apply the declared extraction and gap-aware continuity rules."""

        yield from iter_trajectory_steps(self.position_samples(frames))

    def to_dict(self) -> dict[str, object]:
        """Return the policy in a provenance-ready representation."""

        return {
            "input": {
                "source_category": self.required_source.value,
                "coordinate_frame": {
                    "frame_id": self.required_coordinate_frame.frame_id,
                    "unit": self.required_coordinate_frame.unit,
                    "axis_orientation": (
                        self.required_coordinate_frame.axis_orientation.value
                    ),
                },
            },
            "position": {
                "keypoint_name": self.keypoint_name,
                "minimum_likelihood": self.minimum_likelihood,
            },
            "continuity": {
                "interpolation_applied_by_protocol": False,
                "smoothing_applied_by_protocol": False,
                "valid_displacement": ADJACENT_ACCEPTED_DISPLACEMENT_POLICY,
                "turning": CONSECUTIVE_NONSTATIONARY_TURNING_POLICY,
                "zero_displacement": RETAIN_ZERO_DISPLACEMENT_POLICY,
            },
        }


ROCHE_RECORDED_TRAJECTORY_PROTOCOL = SpatialTrajectoryProtocol(
    keypoint_name="bodycentre",
    minimum_likelihood=None,
    required_source=ObservationSource.RECORDED,
    required_coordinate_frame=ROCHE_COORDINATE_FRAME,
)
