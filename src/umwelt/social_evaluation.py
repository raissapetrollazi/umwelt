"""Per-pair metrics and frozen preference rule for the v0.4 dyad experiment."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import ceil
from statistics import fmean

from umwelt.dyad import DyadMeasurements
from umwelt.errors import DataError
from umwelt.observations import ObservationSource
from umwelt.social_model import SOCIAL_MODEL_IDS
from umwelt.spatial_evaluation import empirical_wasserstein


SOCIAL_REPLICATES = 8
SOCIAL_MASTER_SEED = 4042026
DISTANCE_IMPROVEMENT_FRACTION = 0.05
SIGNED_CHANGE_TOLERANCE_FRACTION = 0.05
PAIR_WIN_FRACTION = 0.60


@dataclass(frozen=True, slots=True)
class SocialReplicateScore:
    pair_id: str
    model_id: str
    replicate: int
    resident_seed: int
    intruder_seed: int
    distance_wasserstein_px: float | None
    signed_change_wasserstein_px: float | None
    recorded_pair_position_count: int
    synthetic_pair_position_count: int
    recorded_pair_transition_count: int
    synthetic_pair_transition_count: int
    synthetic_measurement_summary: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "pair_id": self.pair_id,
            "model_id": self.model_id,
            "replicate": self.replicate,
            "resident_seed": self.resident_seed,
            "intruder_seed": self.intruder_seed,
            "distance_wasserstein_px": self.distance_wasserstein_px,
            "signed_change_wasserstein_px": self.signed_change_wasserstein_px,
            "recorded_pair_position_count": self.recorded_pair_position_count,
            "synthetic_pair_position_count": self.synthetic_pair_position_count,
            "recorded_pair_transition_count": self.recorded_pair_transition_count,
            "synthetic_pair_transition_count": self.synthetic_pair_transition_count,
            "synthetic_measurement_summary": self.synthetic_measurement_summary,
        }


def compare_social_measurements(
    recorded: DyadMeasurements,
    synthetic: DyadMeasurements,
    *,
    model_id: str,
    replicate: int,
    resident_seed: int,
    intruder_seed: int,
) -> SocialReplicateScore:
    """Compare two equally exposed pair trajectories without pooling pairs."""

    if model_id not in SOCIAL_MODEL_IDS or not 0 <= replicate < SOCIAL_REPLICATES:
        raise DataError("Social comparison has an invalid model or replicate.")
    if recorded.source is not ObservationSource.RECORDED:
        raise DataError("Social reference must be recorded.")
    if synthetic.source is not ObservationSource.SYNTHETIC:
        raise DataError("Social candidate must be synthetic.")
    if (
        recorded.pair_id != synthetic.pair_id
        or recorded.recording_id != synthetic.recording_id
        or recorded.resident_id != synthetic.resident_id
        or recorded.intruder_id != synthetic.intruder_id
        or recorded.coordinate_frame != synthetic.coordinate_frame
    ):
        raise DataError("Social comparison requires matching pair identities.")
    if (
        recorded.frame_count != synthetic.frame_count
        or recorded.valid_pair_position_count != synthetic.valid_pair_position_count
        or recorded.valid_pair_transition_count != synthetic.valid_pair_transition_count
    ):
        raise DataError("Social comparison requires equal valid exposure.")
    return SocialReplicateScore(
        pair_id=recorded.pair_id,
        model_id=model_id,
        replicate=replicate,
        resident_seed=resident_seed,
        intruder_seed=intruder_seed,
        distance_wasserstein_px=empirical_wasserstein(
            recorded.pair_distances_px, synthetic.pair_distances_px
        ),
        signed_change_wasserstein_px=empirical_wasserstein(
            recorded.signed_distance_changes_px,
            synthetic.signed_distance_changes_px,
        ),
        recorded_pair_position_count=recorded.valid_pair_position_count,
        synthetic_pair_position_count=synthetic.valid_pair_position_count,
        recorded_pair_transition_count=recorded.valid_pair_transition_count,
        synthetic_pair_transition_count=synthetic.valid_pair_transition_count,
        synthetic_measurement_summary=synthetic.compact_dict(),
    )


def aggregate_social_scores(
    scores: tuple[SocialReplicateScore, ...],
    *,
    eligible_pair_ids: tuple[str, ...],
) -> dict[str, object]:
    """Apply the same three-part preference rule to either source split."""

    if not eligible_pair_ids or len(eligible_pair_ids) != len(set(eligible_pair_ids)):
        raise DataError("Social aggregation needs distinct eligible pairs.")
    grouped: dict[tuple[str, str], list[SocialReplicateScore]] = defaultdict(list)
    for score in scores:
        if score.model_id not in SOCIAL_MODEL_IDS:
            raise DataError("Social score has an unknown model identifier.")
        if score.pair_id not in eligible_pair_ids:
            raise DataError("Social score refers to an ineligible pair.")
        grouped[(score.pair_id, score.model_id)].append(score)

    pair_summaries: list[dict[str, object]] = []
    means_by_model: dict[str, dict[str, list[float]]] = {
        model_id: {"distance": [], "signed_change": []} for model_id in SOCIAL_MODEL_IDS
    }
    distance_wins = 0
    complete = True

    for pair_id in eligible_pair_ids:
        model_means: dict[str, dict[str, float | None]] = {}
        for model_id in SOCIAL_MODEL_IDS:
            replicates = grouped[(pair_id, model_id)]
            if sorted(item.replicate for item in replicates) != list(
                range(SOCIAL_REPLICATES)
            ):
                raise DataError("Social pair needs exactly eight distinct replicates.")
            distance_values = [item.distance_wasserstein_px for item in replicates]
            change_values = [item.signed_change_wasserstein_px for item in replicates]
            distance = (
                fmean(value for value in distance_values if value is not None)
                if all(value is not None for value in distance_values)
                else None
            )
            signed_change = (
                fmean(value for value in change_values if value is not None)
                if all(value is not None for value in change_values)
                else None
            )
            model_means[model_id] = {
                "distance_wasserstein_px": distance,
                "signed_change_wasserstein_px": signed_change,
            }
            if distance is None or signed_change is None:
                complete = False
            else:
                means_by_model[model_id]["distance"].append(distance)
                means_by_model[model_id]["signed_change"].append(signed_change)
        baseline_distance = model_means[SOCIAL_MODEL_IDS[0]]["distance_wasserstein_px"]
        candidate_distance = model_means[SOCIAL_MODEL_IDS[1]]["distance_wasserstein_px"]
        pair_win = (
            candidate_distance < baseline_distance
            if candidate_distance is not None and baseline_distance is not None
            else None
        )
        distance_wins += pair_win is True
        pair_summaries.append(
            {
                "pair_id": pair_id,
                "models": model_means,
                "candidate_distance_win": pair_win,
            }
        )

    mean_distance = {
        model_id: fmean(means_by_model[model_id]["distance"]) if complete else None
        for model_id in SOCIAL_MODEL_IDS
    }
    mean_change = {
        model_id: fmean(means_by_model[model_id]["signed_change"]) if complete else None
        for model_id in SOCIAL_MODEL_IDS
    }
    needed_wins = ceil(PAIR_WIN_FRACTION * len(eligible_pair_ids))
    distance_criterion: bool | None = None
    change_criterion: bool | None = None
    win_criterion: bool | None = None
    preferred: bool | None = None
    if complete:
        baseline_distance_mean = mean_distance[SOCIAL_MODEL_IDS[0]]
        candidate_distance_mean = mean_distance[SOCIAL_MODEL_IDS[1]]
        baseline_change_mean = mean_change[SOCIAL_MODEL_IDS[0]]
        candidate_change_mean = mean_change[SOCIAL_MODEL_IDS[1]]
        if (
            baseline_distance_mean is None
            or candidate_distance_mean is None
            or baseline_change_mean is None
            or candidate_change_mean is None
        ):
            raise DataError("Complete social split lacks aggregate metrics.")
        distance_criterion = (
            candidate_distance_mean
            <= (1 - DISTANCE_IMPROVEMENT_FRACTION) * baseline_distance_mean
        )
        change_criterion = (
            candidate_change_mean
            <= (1 + SIGNED_CHANGE_TOLERANCE_FRACTION) * baseline_change_mean
        )
        win_criterion = distance_wins >= needed_wins
        preferred = distance_criterion and change_criterion and win_criterion
    return {
        "eligible_pair_count": len(eligible_pair_ids),
        "replicates_per_model_pair": SOCIAL_REPLICATES,
        "per_pair": pair_summaries,
        "equal_pair_mean_distance_wasserstein_px": mean_distance,
        "equal_pair_mean_signed_change_wasserstein_px": mean_change,
        "candidate_distance_win_count": distance_wins,
        "required_distance_win_count": needed_wins,
        "preference_criteria": {
            "distance_at_least_5_percent_better": distance_criterion,
            "signed_change_no_more_than_5_percent_worse": change_criterion,
            "distance_better_on_at_least_60_percent_of_pairs": win_criterion,
        },
        "candidate_preferred": preferred,
    }
