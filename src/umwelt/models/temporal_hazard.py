"""Interpretable phase and bout-duration hazard models for Umwelt v0.1."""

from __future__ import annotations

import random
from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime
from math import isclose

from umwelt.errors import DataError
from umwelt.models.circadian_markov import (
    MODEL_TYPE,
    PhaseConditionedMarkovModel,
    fit_phase_conditioned_markov,
)
from umwelt.observations import (
    BehavioralState,
    ObservationDataset,
    ObservationSource,
    TimeSeries,
)

SIMPLE_STATE_MODEL = "state-markov-v1"
DURATION_MODEL = "duration-hazard-v1"
PHASE_DURATION_MODEL = "phase-duration-hazard-v1"


@dataclass(frozen=True, slots=True)
class TemporalModelSpec:
    """One factorial combination of phase and duration dependence."""

    model_id: str
    label: str
    use_phase: bool
    use_duration: bool
    hypothesis: str


MODEL_LADDER = (
    TemporalModelSpec(
        SIMPLE_STATE_MODEL,
        "State only",
        False,
        False,
        "Tests how much structure follows from pooled first-order state persistence.",
    ),
    TemporalModelSpec(
        MODEL_TYPE,
        "Phase only",
        True,
        False,
        "Tests the additional contribution of daily phase conditioning.",
    ),
    TemporalModelSpec(
        DURATION_MODEL,
        "Duration only",
        False,
        True,
        "Tests whether elapsed bout duration explains structure beyond first-order persistence.",
    ),
    TemporalModelSpec(
        PHASE_DURATION_MODEL,
        "Phase and duration",
        True,
        True,
        "Tests whether daily phase and elapsed bout duration contribute complementary structure.",
    ),
)


@dataclass(frozen=True, slots=True)
class HazardCell:
    """Estimated probability of leaving a state in one conditioning cell."""

    state: BehavioralState
    phase: int
    duration_bin: int
    exposures: int
    leaves: int
    prior_probability: float
    leave_probability: float


@dataclass(frozen=True, slots=True)
class TemporalHazardModel:
    """A two-state model with optional phase and elapsed-duration memory."""

    spec: TemporalModelSpec
    epoch_seconds: int
    phase_bins: int
    duration_bin_edges_epochs: tuple[int, ...]
    transition_prior: float
    duration_prior_strength: float
    initial_sleep_probabilities: tuple[float, ...]
    hazards: dict[tuple[BehavioralState, int, int], HazardCell]
    activity_model: PhaseConditionedMarkovModel
    training_subject_ids: tuple[str, ...]
    training_observations: int
    training_transitions: int

    def phase_for(self, timestamp: datetime) -> int:
        """Return the transition phase bin, or zero when phase is disabled."""

        if not self.spec.use_phase:
            return 0
        seconds = (
            timestamp.hour * 3_600
            + timestamp.minute * 60
            + timestamp.second
            + timestamp.microsecond / 1_000_000
        )
        return min(int(seconds * self.phase_bins / 86_400), self.phase_bins - 1)

    def duration_bin_for(self, bout_age_epochs: int) -> int:
        """Map positive elapsed bout age to a compact logarithmic-style bin."""

        if bout_age_epochs < 1:
            raise DataError("Bout age must be at least one epoch.")
        if not self.spec.use_duration:
            return 0
        return bisect_left(self.duration_bin_edges_epochs, bout_age_epochs)

    def initial_state(
        self, timestamp: datetime, generator: random.Random
    ) -> BehavioralState:
        """Draw the first state in a contiguous synthetic segment."""

        probability = self.initial_sleep_probabilities[self.phase_for(timestamp)]
        if generator.random() < probability:
            return BehavioralState.SLEEP
        return BehavioralState.WAKE

    def leave_probability(
        self,
        state: BehavioralState,
        timestamp: datetime,
        bout_age_epochs: int,
    ) -> float:
        """Return the fitted discrete hazard for the current model state."""

        key = (
            state,
            self.phase_for(timestamp),
            self.duration_bin_for(bout_age_epochs),
        )
        return self.hazards[key].leave_probability

    def next_state(
        self,
        state: BehavioralState,
        timestamp: datetime,
        bout_age_epochs: int,
        generator: random.Random,
    ) -> BehavioralState:
        """Advance the explicit state using its fitted discrete hazard."""

        if generator.random() >= self.leave_probability(
            state, timestamp, bout_age_epochs
        ):
            return state
        if state is BehavioralState.SLEEP:
            return BehavioralState.WAKE
        return BehavioralState.SLEEP

    @property
    def parameter_count(self) -> int:
        """Return the explicit state-dynamics parameter count."""

        return len(self.initial_sleep_probabilities) + len(self.hazards)

    def to_dict(self) -> dict[str, object]:
        """Return inspectable state-dynamics parameters without duplicating emissions."""

        cells = sorted(
            self.hazards.values(),
            key=lambda cell: (cell.state.value, cell.phase, cell.duration_bin),
        )
        return {
            "schema_version": "umwelt.temporal-model.v1",
            "model_id": self.spec.model_id,
            "label": self.spec.label,
            "hypothesis": self.spec.hypothesis,
            "mechanisms": {
                "phase_conditioning": self.spec.use_phase,
                "duration_conditioning": self.spec.use_duration,
            },
            "epoch_seconds": self.epoch_seconds,
            "phase_bins": self.phase_bins if self.spec.use_phase else 1,
            "duration_bin_edges_epochs": (
                list(self.duration_bin_edges_epochs) if self.spec.use_duration else []
            ),
            "transition_prior": self.transition_prior,
            "duration_prior_strength": (
                self.duration_prior_strength if self.spec.use_duration else None
            ),
            "parameter_count": self.parameter_count,
            "training_subject_ids": list(self.training_subject_ids),
            "training_observations": self.training_observations,
            "training_transitions": self.training_transitions,
            "initial_sleep_probabilities": list(self.initial_sleep_probabilities),
            "hazards": [
                {
                    "state": cell.state.value,
                    "phase": cell.phase,
                    "duration_bin": cell.duration_bin,
                    "exposures": cell.exposures,
                    "leaves": cell.leaves,
                    "prior_probability": cell.prior_probability,
                    "leave_probability": cell.leave_probability,
                }
                for cell in cells
            ],
            "shared_activity_emission": "phase-conditioned-zero-inflated-gamma-v1",
            "assumptions": [
                "The observable state space contains only wake and behaviorally defined sleep.",
                "State transitions depend only on the enabled phase and duration mechanisms.",
                "Duration is elapsed time in the current synthetic state, not a mental variable.",
                "Parameters are pooled across training subjects.",
                "All ladder models use the same fitted activity-emission mechanism.",
            ],
        }


def next_bout_age(
    previous_state: BehavioralState,
    next_state: BehavioralState,
    previous_age_epochs: int,
) -> int:
    """Increment elapsed state duration or reset it after a transition."""

    if previous_age_epochs < 1:
        raise DataError("Bout age must be at least one epoch.")
    return previous_age_epochs + 1 if next_state is previous_state else 1


def _probability(successes: int, total: int, prior: float) -> float:
    return (successes + prior) / (total + 2 * prior)


def _fit_one_model(
    dataset: ObservationDataset,
    activity_model: PhaseConditionedMarkovModel,
    spec: TemporalModelSpec,
    *,
    phase_bins: int,
    duration_bin_edges_epochs: tuple[int, ...],
    transition_prior: float,
    duration_prior_strength: float,
) -> TemporalHazardModel:
    model_phase_bins = phase_bins if spec.use_phase else 1
    duration_bins = len(duration_bin_edges_epochs) + 1 if spec.use_duration else 1
    occupancy_total = [0] * model_phase_bins
    occupancy_sleep = [0] * model_phase_bins
    global_exposure = {state: 0 for state in BehavioralState}
    global_leaves = {state: 0 for state in BehavioralState}
    phase_exposure = {
        (state, phase): 0
        for state in BehavioralState
        for phase in range(model_phase_bins)
    }
    phase_leaves = dict(phase_exposure)
    cell_exposure = {
        (state, phase, duration): 0
        for state in BehavioralState
        for phase in range(model_phase_bins)
        for duration in range(duration_bins)
    }
    cell_leaves = dict(cell_exposure)
    observations = 0
    transitions = 0

    def phase_for(timestamp: datetime) -> int:
        if not spec.use_phase:
            return 0
        seconds = (
            timestamp.hour * 3_600
            + timestamp.minute * 60
            + timestamp.second
            + timestamp.microsecond / 1_000_000
        )
        return min(int(seconds * model_phase_bins / 86_400), model_phase_bins - 1)

    def duration_for(age: int) -> int:
        return bisect_left(duration_bin_edges_epochs, age) if spec.use_duration else 0

    for series in dataset.series:
        previous_state: BehavioralState | None = None
        previous_timestamp: datetime | None = None
        previous_age = 0
        for timestamp, state in zip(series.timestamps, series.states, strict=True):
            if state is None:
                previous_state = None
                previous_timestamp = None
                previous_age = 0
                continue
            phase = phase_for(timestamp)
            occupancy_total[phase] += 1
            occupancy_sleep[phase] += state is BehavioralState.SLEEP
            observations += 1

            contiguous = previous_timestamp is not None and isclose(
                (timestamp - previous_timestamp).total_seconds(),
                activity_model.epoch_seconds,
            )
            if previous_state is not None and contiguous:
                transition_phase = phase_for(previous_timestamp)
                duration_bin = duration_for(previous_age)
                key = (previous_state, transition_phase, duration_bin)
                phase_key = (previous_state, transition_phase)
                left = state is not previous_state
                cell_exposure[key] += 1
                phase_exposure[phase_key] += 1
                global_exposure[previous_state] += 1
                transitions += 1
                if left:
                    cell_leaves[key] += 1
                    phase_leaves[phase_key] += 1
                    global_leaves[previous_state] += 1
                current_age = next_bout_age(previous_state, state, previous_age)
            else:
                current_age = 1
            previous_state = state
            previous_timestamp = timestamp
            previous_age = current_age

    global_sleep_probability = _probability(
        sum(occupancy_sleep), sum(occupancy_total), transition_prior
    )
    initial_sleep = tuple(
        _probability(occupancy_sleep[phase], occupancy_total[phase], transition_prior)
        if occupancy_total[phase]
        else global_sleep_probability
        for phase in range(model_phase_bins)
    )
    global_hazard = {
        state: _probability(
            global_leaves[state], global_exposure[state], transition_prior
        )
        for state in BehavioralState
    }
    phase_hazard = {
        (state, phase): _probability(
            phase_leaves[(state, phase)],
            phase_exposure[(state, phase)],
            transition_prior,
        )
        if phase_exposure[(state, phase)]
        else global_hazard[state]
        for state in BehavioralState
        for phase in range(model_phase_bins)
    }

    hazards = {}
    for key, exposure in cell_exposure.items():
        state, phase, _ = key
        if spec.use_duration:
            prior_probability = phase_hazard[(state, phase)]
            probability = (
                cell_leaves[key] + duration_prior_strength * prior_probability
            ) / (exposure + duration_prior_strength)
        else:
            prior_probability = 0.5
            probability = _probability(cell_leaves[key], exposure, transition_prior)
        hazards[key] = HazardCell(
            state=state,
            phase=phase,
            duration_bin=key[2],
            exposures=exposure,
            leaves=cell_leaves[key],
            prior_probability=prior_probability,
            leave_probability=probability,
        )

    return TemporalHazardModel(
        spec=spec,
        epoch_seconds=activity_model.epoch_seconds,
        phase_bins=model_phase_bins,
        duration_bin_edges_epochs=duration_bin_edges_epochs,
        transition_prior=transition_prior,
        duration_prior_strength=duration_prior_strength,
        initial_sleep_probabilities=initial_sleep,
        hazards=hazards,
        activity_model=activity_model,
        training_subject_ids=tuple(series.subject_id for series in dataset.series),
        training_observations=observations,
        training_transitions=transitions,
    )


def fit_temporal_model_ladder(
    dataset: ObservationDataset,
    *,
    phase_bins: int = 24,
    duration_bin_edges_epochs: tuple[int, ...] = (
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1_024,
    ),
    transition_prior: float = 0.5,
    duration_prior_strength: float = 8.0,
    emission_prior: float = 0.5,
) -> tuple[TemporalHazardModel, ...]:
    """Fit the four-model factorial ladder with shared activity emissions."""

    if any(
        series.source is not ObservationSource.RECORDED for series in dataset.series
    ):
        raise DataError("Temporal models may only be fit to recorded observations.")
    if phase_bins < 1 or phase_bins > 1_440:
        raise DataError("Phase bins must be between 1 and 1440.")
    if transition_prior <= 0 or duration_prior_strength <= 0:
        raise DataError("Temporal-model smoothing parameters must be positive.")
    if not duration_bin_edges_epochs or any(
        edge < 1 for edge in duration_bin_edges_epochs
    ):
        raise DataError("Duration-bin edges must contain positive epochs.")
    if tuple(sorted(set(duration_bin_edges_epochs))) != duration_bin_edges_epochs:
        raise DataError("Duration-bin edges must be strictly increasing.")

    activity_model = fit_phase_conditioned_markov(
        dataset,
        phase_bins=phase_bins,
        transition_prior=transition_prior,
        emission_prior=emission_prior,
    )
    return tuple(
        _fit_one_model(
            dataset,
            activity_model,
            spec,
            phase_bins=phase_bins,
            duration_bin_edges_epochs=duration_bin_edges_epochs,
            transition_prior=transition_prior,
            duration_prior_strength=duration_prior_strength,
        )
        for spec in MODEL_LADDER
    )


def simulate_temporal_from_template(
    model: TemporalHazardModel,
    template: TimeSeries,
    *,
    subject_id: str,
    seed: int,
) -> TimeSeries:
    """Generate synthetic behavior using only a template's timestamps and gaps."""

    generator = random.Random(seed)
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
            state = model.initial_state(timestamp, generator)
            bout_age = 1
        else:
            state = model.next_state(
                previous_state, previous_timestamp, bout_age, generator
            )
            bout_age = next_bout_age(previous_state, state, bout_age)
        activity = model.activity_model.activity_for(state, timestamp, generator)
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
