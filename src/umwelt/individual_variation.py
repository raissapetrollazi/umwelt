"""Explicit training-population individual variation for Umwelt v0.1."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from datetime import datetime
from math import exp, isclose, log

from umwelt.errors import DataError
from umwelt.models.temporal_hazard import (
    PHASE_DURATION_MODEL,
    TemporalHazardModel,
    next_bout_age,
)
from umwelt.observations import (
    BehavioralState,
    ObservationDataset,
    ObservationSource,
    TimeSeries,
)

POPULATION_MODEL_ID = "phase-duration-population-v1"
PROBABILITY_EPSILON = 1e-9


def _smoothed_probability(successes: int, total: int, prior: float) -> float:
    return (successes + prior) / (total + 2 * prior)


def _clamp_probability(probability: float) -> float:
    return min(max(probability, PROBABILITY_EPSILON), 1.0 - PROBABILITY_EPSILON)


def _logit(probability: float) -> float:
    value = _clamp_probability(probability)
    return log(value / (1.0 - value))


def _logistic(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + exp(-value))
    exponential = exp(value)
    return exponential / (1.0 + exponential)


def _offset_probability(probability: float, logit_offset: float) -> float:
    return _logistic(_logit(probability) + logit_offset)


@dataclass(frozen=True, slots=True)
class IndividualVariationProfile:
    """Training-subject offsets used as an empirical population profile."""

    source_subject_id: str
    sleep_occupancy_logit_offset: float
    wake_leave_logit_offset: float
    sleep_leave_logit_offset: float
    observations: int
    transitions: int

    def leave_offset(self, state: BehavioralState) -> float:
        """Return the fitted logit offset for leaving one explicit state."""

        if state is BehavioralState.WAKE:
            return self.wake_leave_logit_offset
        return self.sleep_leave_logit_offset

    def to_dict(self) -> dict[str, object]:
        """Return an inspectable representation of one training profile."""

        return {
            "source_subject_id": self.source_subject_id,
            "sleep_occupancy_logit_offset": self.sleep_occupancy_logit_offset,
            "wake_leave_logit_offset": self.wake_leave_logit_offset,
            "sleep_leave_logit_offset": self.sleep_leave_logit_offset,
            "observations": self.observations,
            "transitions": self.transitions,
        }


@dataclass(frozen=True, slots=True)
class _SubjectStatistics:
    subject_id: str
    observations: int
    sleep_observations: int
    exposures: dict[BehavioralState, int]
    leaves: dict[BehavioralState, int]

    @property
    def transitions(self) -> int:
        return sum(self.exposures.values())


@dataclass(frozen=True, slots=True)
class PopulationTemporalModel:
    """Phase+duration dynamics plus training-derived between-subject variation."""

    base_model: TemporalHazardModel
    profiles: tuple[IndividualVariationProfile, ...]

    @property
    def model_id(self) -> str:
        return POPULATION_MODEL_ID

    @property
    def label(self) -> str:
        return "Phase + duration + population variation"

    @property
    def hypothesis(self) -> str:
        return (
            "Tests whether stable between-subject offsets estimated only from training "
            "animals are sufficient to reproduce observed individual heterogeneity."
        )

    @property
    def parameter_count(self) -> int:
        """Count pooled state parameters plus three stored offsets per training mouse."""

        return self.base_model.parameter_count + 3 * len(self.profiles)

    @property
    def training_subject_ids(self) -> tuple[str, ...]:
        return self.base_model.training_subject_ids

    def profile_for_seed(self, seed: int) -> IndividualVariationProfile:
        """Sample one empirical training profile deterministically."""

        if not self.profiles:
            raise DataError("Population model requires at least one training profile.")
        generator = random.Random(seed)
        return self.profiles[generator.randrange(len(self.profiles))]

    def initial_sleep_probability(
        self, timestamp: datetime, profile: IndividualVariationProfile
    ) -> float:
        """Apply one subject-level occupancy offset to the pooled phase probability."""

        pooled = self.base_model.initial_sleep_probabilities[
            self.base_model.phase_for(timestamp)
        ]
        return _offset_probability(pooled, profile.sleep_occupancy_logit_offset)

    def leave_probability(
        self,
        state: BehavioralState,
        timestamp: datetime,
        bout_age_epochs: int,
        profile: IndividualVariationProfile,
    ) -> float:
        """Apply one subject-level state-specific offset to the pooled hazard."""

        pooled = self.base_model.leave_probability(state, timestamp, bout_age_epochs)
        return _offset_probability(pooled, profile.leave_offset(state))

    def to_dict(self) -> dict[str, object]:
        """Return the empirical population mechanism and its scientific assumptions."""

        return {
            "schema_version": "umwelt.population-temporal-model.v1",
            "model_id": self.model_id,
            "label": self.label,
            "hypothesis": self.hypothesis,
            "base_model_id": self.base_model.spec.model_id,
            "parameter_count": self.parameter_count,
            "training_subject_ids": list(self.training_subject_ids),
            "profile_count": len(self.profiles),
            "profile_sampling": "uniform empirical resampling with replacement",
            "profile_effects": (
                "global logit offsets for sleep occupancy, wake-leaving hazard, "
                "and sleep-leaving hazard"
            ),
            "profiles": [profile.to_dict() for profile in self.profiles],
            "assumptions": [
                "Only training animals contribute individual-variation profiles.",
                "Development animals are never used to calibrate profile values or selection.",
                "A synthetic individual samples one fixed profile for its full generated series.",
                "The profile modifies pooled phase+duration state dynamics but not activity emissions.",
                "Profile offsets are computational population variation, not inferred personality, physiology, or mental state.",
                "The empirical profile distribution is a minimal baseline, not a claim of a biological random-effects distribution.",
            ],
        }


def _subject_statistics(series: TimeSeries, epoch_seconds: int) -> _SubjectStatistics:
    observations = 0
    sleep_observations = 0
    exposures = {state: 0 for state in BehavioralState}
    leaves = {state: 0 for state in BehavioralState}
    previous_state: BehavioralState | None = None
    previous_timestamp: datetime | None = None

    for timestamp, state in zip(series.timestamps, series.states, strict=True):
        if state is None:
            previous_state = None
            previous_timestamp = None
            continue
        observations += 1
        sleep_observations += state is BehavioralState.SLEEP
        contiguous = previous_timestamp is not None and isclose(
            (timestamp - previous_timestamp).total_seconds(), epoch_seconds
        )
        if previous_state is not None and contiguous:
            exposures[previous_state] += 1
            if state is not previous_state:
                leaves[previous_state] += 1
        previous_state = state
        previous_timestamp = timestamp

    if not observations:
        raise DataError(f"Training subject {series.subject_id} has no observations.")
    return _SubjectStatistics(
        subject_id=series.subject_id,
        observations=observations,
        sleep_observations=sleep_observations,
        exposures=exposures,
        leaves=leaves,
    )


def fit_population_temporal_model(
    dataset: ObservationDataset,
    base_model: TemporalHazardModel,
) -> PopulationTemporalModel:
    """Fit a minimal empirical distribution of stable training-subject offsets."""

    if base_model.spec.model_id != PHASE_DURATION_MODEL:
        raise DataError(
            "Population variation currently requires the phase+duration temporal model."
        )
    if any(
        series.source is not ObservationSource.RECORDED for series in dataset.series
    ):
        raise DataError("Individual variation may only be fit to recorded observations.")
    subject_ids = tuple(series.subject_id for series in dataset.series)
    if subject_ids != base_model.training_subject_ids:
        raise DataError(
            "Population-variation training subjects must exactly match the base model."
        )
    if not dataset.series:
        raise DataError("Population variation requires at least one training subject.")

    statistics = tuple(
        _subject_statistics(series, base_model.epoch_seconds) for series in dataset.series
    )
    total_observations = sum(item.observations for item in statistics)
    total_sleep = sum(item.sleep_observations for item in statistics)
    pooled_sleep = _smoothed_probability(
        total_sleep, total_observations, base_model.transition_prior
    )
    pooled_leave = {
        state: _smoothed_probability(
            sum(item.leaves[state] for item in statistics),
            sum(item.exposures[state] for item in statistics),
            base_model.transition_prior,
        )
        for state in BehavioralState
    }

    profiles = []
    for item in statistics:
        subject_sleep = _smoothed_probability(
            item.sleep_observations, item.observations, base_model.transition_prior
        )
        subject_leave = {
            state: _smoothed_probability(
                item.leaves[state], item.exposures[state], base_model.transition_prior
            )
            for state in BehavioralState
        }
        profiles.append(
            IndividualVariationProfile(
                source_subject_id=item.subject_id,
                sleep_occupancy_logit_offset=_logit(subject_sleep)
                - _logit(pooled_sleep),
                wake_leave_logit_offset=_logit(subject_leave[BehavioralState.WAKE])
                - _logit(pooled_leave[BehavioralState.WAKE]),
                sleep_leave_logit_offset=_logit(subject_leave[BehavioralState.SLEEP])
                - _logit(pooled_leave[BehavioralState.SLEEP]),
                observations=item.observations,
                transitions=item.transitions,
            )
        )

    return PopulationTemporalModel(base_model=base_model, profiles=tuple(profiles))


def derive_profile_seed(
    master_seed: int, subject_id: str, replicate: int
) -> int:
    """Derive a stable seed for empirical population-profile selection."""

    if replicate < 1:
        raise DataError("Replicate numbers begin at one.")
    payload = f"{master_seed}|individual-profile|{subject_id}|{replicate}".encode(
        "ascii"
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def simulate_pooled_from_template(
    model: TemporalHazardModel,
    template: TimeSeries,
    *,
    subject_id: str,
    state_seed: int,
    activity_seed: int,
) -> TimeSeries:
    """Generate the pooled control with state and activity randomness isolated."""

    if model.spec.model_id != PHASE_DURATION_MODEL:
        raise DataError("Pooled individuality control requires phase+duration dynamics.")
    state_generator = random.Random(state_seed)
    activity_generator = random.Random(activity_seed)
    states: list[BehavioralState | None] = []
    activities: list[float | None] = []
    previous_state: BehavioralState | None = None
    previous_timestamp: datetime | None = None
    bout_age = 0

    for timestamp, available in zip(
        template.timestamps,
        (state is not None for state in template.states),
        strict=True,
    ):
        if not available:
            states.append(None)
            activities.append(None)
            previous_state = None
            previous_timestamp = None
            bout_age = 0
            continue

        contiguous = previous_timestamp is not None and isclose(
            (timestamp - previous_timestamp).total_seconds(), model.epoch_seconds
        )
        if previous_state is None or not contiguous:
            state = model.initial_state(timestamp, state_generator)
            bout_age = 1
        else:
            state = model.next_state(
                previous_state, previous_timestamp, bout_age, state_generator
            )
            bout_age = next_bout_age(previous_state, state, bout_age)
        activity = model.activity_model.activity_for(
            state, timestamp, activity_generator
        )
        states.append(state)
        activities.append(activity)
        previous_state = state
        previous_timestamp = timestamp

    return TimeSeries(
        subject_id=subject_id,
        source=ObservationSource.SYNTHETIC,
        timestamps=template.timestamps,
        states=tuple(states),
        activities=tuple(activities),
    )


def simulate_population_from_template(
    model: PopulationTemporalModel,
    template: TimeSeries,
    *,
    subject_id: str,
    state_seed: int,
    activity_seed: int,
    profile_seed: int,
) -> tuple[TimeSeries, IndividualVariationProfile]:
    """Generate one synthetic individual with isolated state/activity randomness."""

    profile = model.profile_for_seed(profile_seed)
    state_generator = random.Random(state_seed)
    activity_generator = random.Random(activity_seed)
    states: list[BehavioralState | None] = []
    activities: list[float | None] = []
    previous_state: BehavioralState | None = None
    previous_timestamp: datetime | None = None
    bout_age = 0

    for timestamp, available in zip(
        template.timestamps,
        (state is not None for state in template.states),
        strict=True,
    ):
        if not available:
            states.append(None)
            activities.append(None)
            previous_state = None
            previous_timestamp = None
            bout_age = 0
            continue

        contiguous = previous_timestamp is not None and isclose(
            (timestamp - previous_timestamp).total_seconds(),
            model.base_model.epoch_seconds,
        )
        if previous_state is None or not contiguous:
            probability = model.initial_sleep_probability(timestamp, profile)
            state = (
                BehavioralState.SLEEP
                if state_generator.random() < probability
                else BehavioralState.WAKE
            )
            bout_age = 1
        else:
            probability = model.leave_probability(
                previous_state, previous_timestamp, bout_age, profile
            )
            if state_generator.random() < probability:
                state = (
                    BehavioralState.WAKE
                    if previous_state is BehavioralState.SLEEP
                    else BehavioralState.SLEEP
                )
            else:
                state = previous_state
            bout_age = next_bout_age(previous_state, state, bout_age)

        activity = model.base_model.activity_model.activity_for(
            state, timestamp, activity_generator
        )
        states.append(state)
        activities.append(activity)
        previous_state = state
        previous_timestamp = timestamp

    return (
        TimeSeries(
            subject_id=subject_id,
            source=ObservationSource.SYNTHETIC,
            timestamps=template.timestamps,
            states=tuple(states),
            activities=tuple(activities),
        ),
        profile,
    )
