"""A compact phase-conditioned Markov baseline for sleep/activity dynamics."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from math import isclose
from typing import Iterable

from umwelt.errors import DataError
from umwelt.observations import (
    BehavioralState,
    ObservationDataset,
    ObservationSource,
    TimeSeries,
)

MODEL_TYPE = "phase-conditioned-markov-v1"


@dataclass(slots=True)
class _Moments:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    maximum: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        self.maximum = max(self.maximum, value)

    @property
    def variance(self) -> float:
        return self.m2 / (self.count - 1) if self.count > 1 else 0.0


@dataclass(slots=True)
class _EmissionAccumulator:
    total: int = 0
    zero: int = 0
    positive: _Moments | None = None

    def __post_init__(self) -> None:
        if self.positive is None:
            self.positive = _Moments()

    def update(self, activity: float) -> None:
        self.total += 1
        if activity == 0:
            self.zero += 1
        else:
            assert self.positive is not None
            self.positive.update(activity)


@dataclass(frozen=True, slots=True)
class ActivityEmission:
    """A zero-inflated gamma description of observable activity."""

    zero_probability: float
    positive_mean: float
    positive_variance: float
    positive_cap: float
    observation_count: int
    positive_count: int

    def sample(self, generator: random.Random) -> float:
        """Draw one non-negative activity value."""

        if self.positive_count == 0 or generator.random() < self.zero_probability:
            return 0.0
        if self.positive_variance <= 0 or self.positive_mean <= 0:
            return self.positive_mean
        shape = self.positive_mean**2 / self.positive_variance
        scale = self.positive_variance / self.positive_mean
        sampled = generator.gammavariate(shape, scale)
        return round(min(sampled, self.positive_cap), 6)


@dataclass(frozen=True, slots=True)
class PhaseParameters:
    """Transition, occupancy, and emission parameters for one daily phase bin."""

    phase: int
    initial_sleep_probability: float
    wake_to_sleep_probability: float
    sleep_to_wake_probability: float
    wake_activity: ActivityEmission
    sleep_activity: ActivityEmission
    occupancy_observations: int
    transition_observations: int


@dataclass(frozen=True, slots=True)
class PhaseConditionedMarkovModel:
    """A pooled first-order model whose parameters vary by daily phase."""

    phase_bins: int
    epoch_seconds: int
    transition_prior: float
    emission_prior: float
    training_subject_ids: tuple[str, ...]
    phases: tuple[PhaseParameters, ...]
    training_observations: int
    training_transitions: int

    def __post_init__(self) -> None:
        if self.phase_bins < 1 or self.phase_bins > 1_440:
            raise DataError("Model phase bins must be between 1 and 1440.")
        if len(self.phases) != self.phase_bins:
            raise DataError("Model phase parameters do not match phase-bin count.")
        if self.epoch_seconds <= 0:
            raise DataError("Model epoch duration must be positive.")

    def phase_for(self, timestamp: datetime) -> int:
        """Map a source timestamp to a daily phase bin."""

        seconds = (
            timestamp.hour * 3_600
            + timestamp.minute * 60
            + timestamp.second
            + timestamp.microsecond / 1_000_000
        )
        return min(int(seconds * self.phase_bins / 86_400), self.phase_bins - 1)

    def initial_state(
        self, timestamp: datetime, generator: random.Random
    ) -> BehavioralState:
        """Draw a state at the beginning of a contiguous simulated segment."""

        parameters = self.phases[self.phase_for(timestamp)]
        if generator.random() < parameters.initial_sleep_probability:
            return BehavioralState.SLEEP
        return BehavioralState.WAKE

    def next_state(
        self,
        state: BehavioralState,
        timestamp: datetime,
        generator: random.Random,
    ) -> BehavioralState:
        """Advance the explicit synthetic state by one epoch."""

        parameters = self.phases[self.phase_for(timestamp)]
        draw = generator.random()
        if state is BehavioralState.WAKE:
            return (
                BehavioralState.SLEEP
                if draw < parameters.wake_to_sleep_probability
                else BehavioralState.WAKE
            )
        return (
            BehavioralState.WAKE
            if draw < parameters.sleep_to_wake_probability
            else BehavioralState.SLEEP
        )

    def activity_for(
        self,
        state: BehavioralState,
        timestamp: datetime,
        generator: random.Random,
    ) -> float:
        """Draw observable activity conditional on synthetic state and phase."""

        parameters = self.phases[self.phase_for(timestamp)]
        emission = (
            parameters.sleep_activity
            if state is BehavioralState.SLEEP
            else parameters.wake_activity
        )
        return emission.sample(generator)

    def to_dict(self) -> dict[str, object]:
        """Return a complete, JSON-compatible model artifact."""

        return {
            "schema_version": "umwelt.model.v1",
            "model_type": MODEL_TYPE,
            "phase_bins": self.phase_bins,
            "epoch_seconds": self.epoch_seconds,
            "transition_prior": self.transition_prior,
            "emission_prior": self.emission_prior,
            "training_subject_ids": list(self.training_subject_ids),
            "training_observations": self.training_observations,
            "training_transitions": self.training_transitions,
            "assumptions": [
                "The observable state space contains only wake and sleep labels.",
                "The next state depends only on the current state and daily phase.",
                "Parameters are pooled across the selected training subjects.",
                "Activity is conditionally independent given state and daily phase.",
                "Positive activity follows a gamma distribution truncated at the observed training maximum.",
                "Synthetic model state is not an inferred mental state of a real animal.",
            ],
            "phases": [asdict(phase) for phase in self.phases],
        }


def _smoothed_probability(successes: int, total: int, prior: float) -> float:
    return (successes + prior) / (total + 2 * prior)


def _epoch_seconds(dataset: ObservationDataset) -> int:
    intervals: Counter[int] = Counter()
    for series in dataset.series:
        for previous, current in zip(series.timestamps, series.timestamps[1:]):
            delta = (current - previous).total_seconds()
            if not delta.is_integer() or delta <= 0:
                raise DataError("Model fitting requires positive whole-second epochs.")
            intervals[int(delta)] += 1
    if not intervals:
        raise DataError("Model fitting requires at least two timestamps.")
    if len(intervals) != 1:
        raise DataError("Model fitting requires a regular observation interval.")
    return next(iter(intervals))


def _build_emission(
    accumulator: _EmissionAccumulator,
    fallback: _EmissionAccumulator,
    prior: float,
) -> ActivityEmission:
    selected = accumulator if accumulator.total else fallback
    assert selected.positive is not None
    positive = selected.positive
    if positive.count == 0:
        zero_probability = 1.0
    else:
        zero_probability = _smoothed_probability(selected.zero, selected.total, prior)
    return ActivityEmission(
        zero_probability=zero_probability,
        positive_mean=positive.mean,
        positive_variance=positive.variance,
        positive_cap=positive.maximum,
        observation_count=selected.total,
        positive_count=positive.count,
    )


def fit_phase_conditioned_markov(
    dataset: ObservationDataset,
    *,
    phase_bins: int = 24,
    transition_prior: float = 0.5,
    emission_prior: float = 0.5,
) -> PhaseConditionedMarkovModel:
    """Fit the explicit baseline model to recorded observation series."""

    if not dataset.series:
        raise DataError("Model fitting requires at least one time series.")
    if any(series.source is not ObservationSource.RECORDED for series in dataset.series):
        raise DataError("The v0.1 model may only be fit to recorded observations.")
    if phase_bins < 1 or phase_bins > 1_440:
        raise DataError("Model phase bins must be between 1 and 1440.")
    if transition_prior <= 0 or emission_prior <= 0:
        raise DataError("Model smoothing priors must be positive.")

    epoch_seconds = _epoch_seconds(dataset)
    occupancy_total = [0] * phase_bins
    occupancy_sleep = [0] * phase_bins
    transitions_total = [0] * phase_bins
    wake_total = [0] * phase_bins
    wake_to_sleep = [0] * phase_bins
    sleep_total = [0] * phase_bins
    sleep_to_wake = [0] * phase_bins
    emissions = [
        {
            BehavioralState.WAKE: _EmissionAccumulator(),
            BehavioralState.SLEEP: _EmissionAccumulator(),
        }
        for _ in range(phase_bins)
    ]
    global_emissions = {
        BehavioralState.WAKE: _EmissionAccumulator(),
        BehavioralState.SLEEP: _EmissionAccumulator(),
    }

    global_wake_total = 0
    global_wake_to_sleep = 0
    global_sleep_total = 0
    global_sleep_to_wake = 0
    global_occupancy = 0
    global_sleep = 0

    for series in dataset.series:
        previous_state: BehavioralState | None = None
        previous_timestamp: datetime | None = None
        for timestamp, state, activity in zip(
            series.timestamps, series.states, series.activities, strict=True
        ):
            if state is None or activity is None:
                previous_state = None
                previous_timestamp = None
                continue
            phase = int(
                (
                    timestamp.hour * 3_600
                    + timestamp.minute * 60
                    + timestamp.second
                    + timestamp.microsecond / 1_000_000
                )
                * phase_bins
                / 86_400
            )
            phase = min(phase, phase_bins - 1)
            occupancy_total[phase] += 1
            global_occupancy += 1
            if state is BehavioralState.SLEEP:
                occupancy_sleep[phase] += 1
                global_sleep += 1
            emissions[phase][state].update(activity)
            global_emissions[state].update(activity)

            if previous_state is not None and previous_timestamp is not None:
                delta = (timestamp - previous_timestamp).total_seconds()
                if isclose(delta, epoch_seconds):
                    transition_phase = int(
                        (
                            previous_timestamp.hour * 3_600
                            + previous_timestamp.minute * 60
                            + previous_timestamp.second
                            + previous_timestamp.microsecond / 1_000_000
                        )
                        * phase_bins
                        / 86_400
                    )
                    transition_phase = min(transition_phase, phase_bins - 1)
                    transitions_total[transition_phase] += 1
                    if previous_state is BehavioralState.WAKE:
                        wake_total[transition_phase] += 1
                        global_wake_total += 1
                        if state is BehavioralState.SLEEP:
                            wake_to_sleep[transition_phase] += 1
                            global_wake_to_sleep += 1
                    else:
                        sleep_total[transition_phase] += 1
                        global_sleep_total += 1
                        if state is BehavioralState.WAKE:
                            sleep_to_wake[transition_phase] += 1
                            global_sleep_to_wake += 1
            previous_state = state
            previous_timestamp = timestamp

    global_initial_sleep = _smoothed_probability(
        global_sleep, global_occupancy, transition_prior
    )
    global_wake_switch = _smoothed_probability(
        global_wake_to_sleep, global_wake_total, transition_prior
    )
    global_sleep_switch = _smoothed_probability(
        global_sleep_to_wake, global_sleep_total, transition_prior
    )

    phases: list[PhaseParameters] = []
    for phase in range(phase_bins):
        initial_sleep = (
            _smoothed_probability(
                occupancy_sleep[phase], occupancy_total[phase], transition_prior
            )
            if occupancy_total[phase]
            else global_initial_sleep
        )
        wake_switch = (
            _smoothed_probability(
                wake_to_sleep[phase], wake_total[phase], transition_prior
            )
            if wake_total[phase]
            else global_wake_switch
        )
        sleep_switch = (
            _smoothed_probability(
                sleep_to_wake[phase], sleep_total[phase], transition_prior
            )
            if sleep_total[phase]
            else global_sleep_switch
        )
        phases.append(
            PhaseParameters(
                phase=phase,
                initial_sleep_probability=initial_sleep,
                wake_to_sleep_probability=wake_switch,
                sleep_to_wake_probability=sleep_switch,
                wake_activity=_build_emission(
                    emissions[phase][BehavioralState.WAKE],
                    global_emissions[BehavioralState.WAKE],
                    emission_prior,
                ),
                sleep_activity=_build_emission(
                    emissions[phase][BehavioralState.SLEEP],
                    global_emissions[BehavioralState.SLEEP],
                    emission_prior,
                ),
                occupancy_observations=occupancy_total[phase],
                transition_observations=transitions_total[phase],
            )
        )

    return PhaseConditionedMarkovModel(
        phase_bins=phase_bins,
        epoch_seconds=epoch_seconds,
        transition_prior=transition_prior,
        emission_prior=emission_prior,
        training_subject_ids=tuple(series.subject_id for series in dataset.series),
        phases=tuple(phases),
        training_observations=global_occupancy,
        training_transitions=global_wake_total + global_sleep_total,
    )


def simulate_from_template(
    model: PhaseConditionedMarkovModel,
    template: TimeSeries,
    *,
    subject_id: str,
    seed: int,
) -> TimeSeries:
    """Generate a synthetic series using only a template's time grid and gaps."""

    generator = random.Random(seed)
    states: list[BehavioralState | None] = []
    activities: list[float | None] = []
    previous_state: BehavioralState | None = None
    previous_timestamp: datetime | None = None

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
            continue

        contiguous = (
            previous_timestamp is not None
            and isclose(
                (timestamp - previous_timestamp).total_seconds(), model.epoch_seconds
            )
        )
        if previous_state is None or not contiguous:
            state = model.initial_state(timestamp, generator)
        else:
            state = model.next_state(previous_state, previous_timestamp, generator)
        activity = model.activity_for(state, timestamp, generator)
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
