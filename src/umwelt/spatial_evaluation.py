"""Pre-declared spatial measurements and comparison for Umwelt v0.2."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import fmean, median, pstdev

from umwelt.arena import RectangularArena
from umwelt.errors import DataError
from umwelt.observations import ObservationSource
from umwelt.spatial import Point2D, SpatialContext, SpatialFrame
from umwelt.spatial_protocol import SpatialTrajectoryProtocol
from umwelt.trajectory import PositionStatus, TrajectoryStep


REGISTERED_SPATIAL_METRIC_IDS = (
    "path-length-px",
    "adjacent-displacement-px-distribution",
    "absolute-turning-radians-distribution",
)
SPATIAL_QUALITY_CONTROL_IDS = (
    "position-status-counts",
    "source-likelihood-summary",
    "valid-transition-coverage",
)
DISTRIBUTION_SUMMARY_FIELDS = (
    "count",
    "mean",
    "population-standard-deviation",
    "minimum",
    "q25",
    "median",
    "q75",
    "q90",
    "maximum",
)


@dataclass(frozen=True, slots=True)
class SpatialEvaluationProtocol:
    """Measurement definitions frozen before inspecting baseline results."""

    registered_metric_ids: tuple[str, ...] = REGISTERED_SPATIAL_METRIC_IDS
    quality_control_ids: tuple[str, ...] = SPATIAL_QUALITY_CONTROL_IDS
    distribution_summary_fields: tuple[str, ...] = DISTRIBUTION_SUMMARY_FIELDS

    def __post_init__(self) -> None:
        if self.registered_metric_ids != REGISTERED_SPATIAL_METRIC_IDS:
            raise DataError(
                "Registered spatial metrics must match the frozen protocol."
            )
        if self.quality_control_ids != SPATIAL_QUALITY_CONTROL_IDS:
            raise DataError("Spatial quality control must match the frozen protocol.")
        if self.distribution_summary_fields != DISTRIBUTION_SUMMARY_FIELDS:
            raise DataError(
                "Spatial distribution summaries must match the frozen protocol."
            )

    def to_dict(self) -> dict[str, object]:
        """Return exact definitions, exposures, and explicit exclusions."""

        return {
            "unit_of_description": "animal-recording",
            "registered_metrics": [
                {
                    "id": "path-length-px",
                    "definition": (
                        "sum of displacement between adjacent accepted positions"
                    ),
                    "unit": "px",
                    "required_exposure": "valid-transition-count",
                },
                {
                    "id": "adjacent-displacement-px-distribution",
                    "definition": (
                        "displacement between adjacent accepted positions, including zero"
                    ),
                    "unit": "px-per-adjacent-frame-interval",
                    "comparison": "empirical-1-wasserstein",
                },
                {
                    "id": "absolute-turning-radians-distribution",
                    "definition": (
                        "absolute turning magnitude across two consecutive "
                        "nonstationary valid movements"
                    ),
                    "unit": "radian",
                    "comparison": "empirical-1-wasserstein",
                },
            ],
            "distribution_summary_fields": list(self.distribution_summary_fields),
            "quality_control": [
                {
                    "id": "position-status-counts",
                    "definition": "per-status counts for every derived position sample",
                },
                {
                    "id": "source-likelihood-summary",
                    "definition": (
                        "present and missing counts plus the fixed distribution "
                        "summary over available source likelihoods"
                    ),
                },
                {
                    "id": "valid-transition-coverage",
                    "definition": (
                        "adjacent frame-pair opportunities, valid and zero "
                        "displacements, valid turns, and valid displacement "
                        "fraction over opportunities"
                    ),
                },
            ],
            "synthetic_observation_mask": (
                "copy development bodycentre availability only; never coordinate values"
            ),
            "excluded_due_to_missing_source_support": [
                "speed",
                "duration",
                "acceleration",
                "time-lag-autocorrelation",
                "physical-distance",
            ],
            "deferred_pending_validated_arena_bounds": [
                "boundary-distance",
                "center-periphery",
                "spatial-occupancy",
                "outside-arena-fraction",
            ],
            "not_registered_in_first_baseline": ["absolute-heading"],
            "acceptance_threshold": None,
        }


ROCHE_SPATIAL_EVALUATION_PROTOCOL = SpatialEvaluationProtocol()


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return float(ordered[lower] + fraction * (ordered[upper] - ordered[lower]))


def distribution_summary(
    values: Sequence[float],
) -> dict[str, float | int | None]:
    """Summarize one finite one-dimensional sample deterministically."""

    if not values:
        return {
            "count": 0,
            "mean": None,
            "population-standard-deviation": None,
            "minimum": None,
            "q25": None,
            "median": None,
            "q75": None,
            "q90": None,
            "maximum": None,
        }
    return {
        "count": len(values),
        "mean": fmean(values),
        "population-standard-deviation": pstdev(values),
        "minimum": min(values),
        "q25": _quantile(values, 0.25),
        "median": _quantile(values, 0.5),
        "q75": _quantile(values, 0.75),
        "q90": _quantile(values, 0.9),
        "maximum": max(values),
    }


@dataclass(frozen=True, slots=True)
class SpatialQualityControl:
    """Observation and transition exposure kept separate from model metrics."""

    total_samples: int
    position_status_counts: tuple[tuple[str, int], ...]
    source_likelihoods: tuple[float, ...]
    source_likelihood_missing_count: int
    adjacent_transition_opportunities: int
    valid_transition_count: int
    zero_transition_count: int
    valid_turn_count: int

    @property
    def accepted_positions(self) -> int:
        return dict(self.position_status_counts)[PositionStatus.ACCEPTED.value]

    @property
    def accepted_fraction(self) -> float:
        return (
            self.accepted_positions / self.total_samples if self.total_samples else 0.0
        )

    @property
    def valid_transition_fraction(self) -> float | None:
        if not self.adjacent_transition_opportunities:
            return None
        return self.valid_transition_count / self.adjacent_transition_opportunities

    def to_dict(self) -> dict[str, object]:
        return {
            "total_samples": self.total_samples,
            "position_status_counts": dict(self.position_status_counts),
            "accepted_positions": self.accepted_positions,
            "accepted_fraction": self.accepted_fraction,
            "source_likelihood": {
                "present_count": len(self.source_likelihoods),
                "missing_count": self.source_likelihood_missing_count,
                "summary": distribution_summary(self.source_likelihoods),
            },
            "adjacent_transition_opportunities": (
                self.adjacent_transition_opportunities
            ),
            "valid_transition_count": self.valid_transition_count,
            "valid_transition_fraction": self.valid_transition_fraction,
            "zero_transition_count": self.zero_transition_count,
            "valid_turn_count": self.valid_turn_count,
        }


@dataclass(frozen=True, slots=True)
class SpatialMeasurements:
    """Registered measurements for one fully attributed trajectory."""

    context: SpatialContext
    path_length_px: float
    displacements_px: tuple[float, ...]
    absolute_turnings_radians: tuple[float, ...]
    quality_control: SpatialQualityControl

    def compact_dict(self) -> dict[str, object]:
        return {
            "subject_id": self.context.subject_id,
            "recording_id": self.context.recording_id,
            "source_category": self.context.source.value,
            "coordinate_unit": self.context.coordinate_frame.unit,
            "quality_control": self.quality_control.to_dict(),
            "registered_metrics": {
                "path-length-px": {
                    "value": self.path_length_px,
                    "valid-transition-count": (
                        self.quality_control.valid_transition_count
                    ),
                },
                "adjacent-displacement-px-distribution": distribution_summary(
                    self.displacements_px
                ),
                "absolute-turning-radians-distribution": distribution_summary(
                    self.absolute_turnings_radians
                ),
            },
        }


@dataclass(frozen=True, slots=True)
class SpatialComparison:
    """Descriptive recorded-versus-synthetic differences for one replicate."""

    path_length_signed_difference_px: float
    path_length_absolute_difference_px: float
    path_length_signed_relative_difference: float | None
    path_length_relative_difference: float | None
    adjacent_displacement_wasserstein_px: float | None
    absolute_turning_wasserstein_radians: float | None

    def to_dict(self) -> dict[str, float | None]:
        return {
            "path_length_signed_difference_px": self.path_length_signed_difference_px,
            "path_length_absolute_difference_px": (
                self.path_length_absolute_difference_px
            ),
            "path_length_signed_relative_difference": (
                self.path_length_signed_relative_difference
            ),
            "path_length_relative_difference": self.path_length_relative_difference,
            "adjacent_displacement_wasserstein_px": (
                self.adjacent_displacement_wasserstein_px
            ),
            "absolute_turning_wasserstein_radians": (
                self.absolute_turning_wasserstein_radians
            ),
        }


def _median(values: Sequence[float], label: str) -> float:
    if not values:
        raise DataError(
            f"Cannot derive Roche arena geometry without {label} landmarks."
        )
    return float(median(values))


def derive_roche_recording_arena(
    frames: Iterable[SpatialFrame], *, arena_id: str | None = None
) -> RectangularArena:
    """Derive model bounds from source landmarks, not trajectory coordinates."""

    context: SpatialContext | None = None
    left_x: list[float] = []
    right_x: list[float] = []
    top_y: list[float] = []
    bottom_y: list[float] = []

    for frame in frames:
        if context is None:
            context = frame.context
        elif frame.context != context:
            raise DataError("Arena derivation cannot mix spatial contexts.")
        if frame.landmarks is None:
            continue
        for name in ("tl", "bl"):
            point = frame.landmarks.landmark(name).point
            if point is not None:
                left_x.append(point.x)
        for name in ("tr", "br"):
            point = frame.landmarks.landmark(name).point
            if point is not None:
                right_x.append(point.x)
        for name in ("tl", "tr"):
            point = frame.landmarks.landmark(name).point
            if point is not None:
                top_y.append(point.y)
        for name in ("bl", "br"):
            point = frame.landmarks.landmark(name).point
            if point is not None:
                bottom_y.append(point.y)

    if context is None:
        raise DataError("Arena derivation requires at least one spatial frame.")

    return RectangularArena(
        arena_id=arena_id or f"{context.recording_id}:arena",
        coordinate_frame=context.coordinate_frame,
        minimum=Point2D(_median(left_x, "left"), _median(top_y, "top")),
        maximum=Point2D(_median(right_x, "right"), _median(bottom_y, "bottom")),
    )


def measure_trajectory_steps(
    steps: Iterable[TrajectoryStep],
) -> SpatialMeasurements:
    """Measure the frozen contract from already derived trajectory steps."""

    materialized = tuple(steps)
    if not materialized:
        raise DataError("Spatial evaluation requires at least one trajectory sample.")
    context = materialized[0].context
    statuses: Counter[PositionStatus] = Counter()
    confidences: list[float] = []
    missing_confidences = 0
    opportunities = 0
    displacements: list[float] = []
    absolute_turnings: list[float] = []
    zero_transitions = 0
    previous_index: int | None = None

    for step in materialized:
        if step.context != context:
            raise DataError("Spatial evaluation cannot mix trajectory contexts.")
        statuses[step.status] += 1
        if step.confidence is None:
            missing_confidences += 1
        else:
            confidences.append(step.confidence)
        if previous_index is not None and step.frame_index == previous_index + 1:
            opportunities += 1
        if step.displacement is not None:
            displacements.append(step.displacement)
            zero_transitions += step.displacement == 0
        if step.turning_radians is not None:
            absolute_turnings.append(abs(step.turning_radians))
        previous_index = step.frame_index

    ordered_statuses = tuple(
        (status.value, statuses[status]) for status in PositionStatus
    )
    quality_control = SpatialQualityControl(
        total_samples=len(materialized),
        position_status_counts=ordered_statuses,
        source_likelihoods=tuple(confidences),
        source_likelihood_missing_count=missing_confidences,
        adjacent_transition_opportunities=opportunities,
        valid_transition_count=len(displacements),
        zero_transition_count=zero_transitions,
        valid_turn_count=len(absolute_turnings),
    )
    return SpatialMeasurements(
        context=context,
        path_length_px=sum(displacements),
        displacements_px=tuple(displacements),
        absolute_turnings_radians=tuple(absolute_turnings),
        quality_control=quality_control,
    )


def measure_spatial_trajectory(
    frames: Iterable[SpatialFrame],
    *,
    trajectory_protocol: SpatialTrajectoryProtocol,
) -> SpatialMeasurements:
    """Apply the trajectory rule and frozen metric contract without gap filling."""

    return measure_trajectory_steps(trajectory_protocol.trajectory_steps(frames))


def empirical_wasserstein(
    left: Sequence[float], right: Sequence[float]
) -> float | None:
    """Return exact empirical 1-Wasserstein distance in one dimension."""

    if not left or not right:
        return None
    a = sorted(left)
    b = sorted(right)
    points = sorted(set(a) | set(b))
    ai = bi = 0
    distance = 0.0
    for index, point in enumerate(points[:-1]):
        while ai < len(a) and a[ai] <= point:
            ai += 1
        while bi < len(b) and b[bi] <= point:
            bi += 1
        distance += abs(ai / len(a) - bi / len(b)) * (points[index + 1] - point)
    return distance


def compare_spatial_measurements(
    recorded: SpatialMeasurements, synthetic: SpatialMeasurements
) -> SpatialComparison:
    """Compare registered metrics without treating frames as independent animals."""

    if recorded.context.source is not ObservationSource.RECORDED:
        raise DataError("The recorded side must retain the recorded source category.")
    if synthetic.context.source is not ObservationSource.SYNTHETIC:
        raise DataError("The synthetic side must retain the synthetic source category.")
    if recorded.context.subject_id != synthetic.context.subject_id:
        raise DataError("Spatial comparison requires the same development animal.")
    if recorded.context.coordinate_frame != synthetic.context.coordinate_frame:
        raise DataError(
            "Recorded and synthetic spatial comparisons require one coordinate frame."
        )
    if (
        recorded.quality_control.valid_transition_count
        != synthetic.quality_control.valid_transition_count
    ):
        raise DataError(
            "Path-length comparison requires equal valid-transition exposure."
        )

    signed_path = synthetic.path_length_px - recorded.path_length_px
    signed_relative_path = (
        signed_path / recorded.path_length_px if recorded.path_length_px > 0 else None
    )
    return SpatialComparison(
        path_length_signed_difference_px=signed_path,
        path_length_absolute_difference_px=abs(signed_path),
        path_length_signed_relative_difference=signed_relative_path,
        path_length_relative_difference=(
            abs(signed_relative_path) if signed_relative_path is not None else None
        ),
        adjacent_displacement_wasserstein_px=empirical_wasserstein(
            recorded.displacements_px, synthetic.displacements_px
        ),
        absolute_turning_wasserstein_radians=empirical_wasserstein(
            recorded.absolute_turnings_radians,
            synthetic.absolute_turnings_radians,
        ),
    )
