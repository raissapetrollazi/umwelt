"""Tests for the factorial temporal state-model ladder."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta

from umwelt.models.temporal_hazard import (
    DURATION_MODEL,
    MODEL_LADDER,
    PHASE_DURATION_MODEL,
    SIMPLE_STATE_MODEL,
    fit_temporal_model_ladder,
    next_bout_age,
    simulate_temporal_from_template,
)
from umwelt.observations import (
    BehavioralState,
    ObservationDataset,
    ObservationSource,
    TimeSeries,
)


def _series(subject_id: str, *, alternate_values: bool = False) -> TimeSeries:
    start = datetime.fromisoformat("2020-01-01 23:58:00")
    timestamps = tuple(start + timedelta(seconds=10 * index) for index in range(48))
    states = []
    activities = []
    for index in range(48):
        if index == 17:
            states.append(None)
            activities.append(None)
            continue
        state = BehavioralState.SLEEP if (index // 4) % 2 else BehavioralState.WAKE
        if alternate_values:
            state = (
                BehavioralState.WAKE
                if state is BehavioralState.SLEEP
                else BehavioralState.SLEEP
            )
        states.append(state)
        activities.append(0.0 if state is BehavioralState.SLEEP else float(index % 7))
    return TimeSeries(
        subject_id=subject_id,
        source=ObservationSource.RECORDED,
        timestamps=timestamps,
        states=tuple(states),
        activities=tuple(activities),
    )


class TemporalHazardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.recorded = ObservationDataset(
            dataset_id="recorded-fixture",
            series=(_series("1"), _series("2")),
            provenance={"source_category": "recorded"},
        )
        self.models = fit_temporal_model_ladder(
            self.recorded,
            phase_bins=24,
            duration_bin_edges_epochs=(1, 2, 4, 8),
        )

    def test_factorial_ladder_shares_one_activity_emission(self) -> None:
        self.assertEqual(
            [model.spec.model_id for model in self.models],
            [
                SIMPLE_STATE_MODEL,
                "phase-conditioned-markov-v1",
                DURATION_MODEL,
                PHASE_DURATION_MODEL,
            ],
        )
        self.assertEqual(len(MODEL_LADDER), 4)
        self.assertTrue(
            all(
                model.activity_model is self.models[0].activity_model
                for model in self.models
            )
        )

    def test_smoothing_keeps_every_hazard_strictly_valid(self) -> None:
        for model in self.models:
            for cell in model.hazards.values():
                self.assertGreater(cell.leave_probability, 0.0)
                self.assertLess(cell.leave_probability, 1.0)

    def test_duration_bins_and_counter_are_explicit(self) -> None:
        duration_model = self.models[2]
        self.assertEqual(
            [duration_model.duration_bin_for(age) for age in (1, 2, 3, 4, 9)],
            [0, 1, 2, 2, 4],
        )
        self.assertEqual(
            next_bout_age(BehavioralState.WAKE, BehavioralState.WAKE, 3), 4
        )
        self.assertEqual(
            next_bout_age(BehavioralState.WAKE, BehavioralState.SLEEP, 3), 1
        )

    def test_phase_wraps_across_midnight(self) -> None:
        phase_model = self.models[1]
        self.assertEqual(
            phase_model.phase_for(datetime.fromisoformat("2020-01-01 23:59:59")), 23
        )
        self.assertEqual(
            phase_model.phase_for(datetime.fromisoformat("2020-01-02 00:00:00")), 0
        )

    def test_generation_is_deterministic_and_preserves_missingness(self) -> None:
        template = self.recorded.subject("1")
        model = self.models[3]
        first = simulate_temporal_from_template(
            model, template, subject_id="synthetic", seed=1729
        )
        second = simulate_temporal_from_template(
            model, template, subject_id="synthetic", seed=1729
        )
        different = simulate_temporal_from_template(
            model, template, subject_id="synthetic", seed=1730
        )

        self.assertEqual(first, second)
        self.assertNotEqual(
            (first.states, first.activities), (different.states, different.activities)
        )
        self.assertEqual(
            tuple(state is None for state in first.states),
            tuple(state is None for state in template.states),
        )
        self.assertEqual(first.source, ObservationSource.SYNTHETIC)

    def test_template_behavior_values_do_not_leak_into_generation(self) -> None:
        original = self.recorded.subject("1")
        altered = _series("altered", alternate_values=True)
        model = self.models[3]

        first = simulate_temporal_from_template(
            model, original, subject_id="synthetic", seed=99
        )
        second = simulate_temporal_from_template(
            model, altered, subject_id="synthetic", seed=99
        )

        self.assertEqual(first.timestamps, second.timestamps)
        self.assertEqual(first.states, second.states)
        self.assertEqual(first.activities, second.activities)


if __name__ == "__main__":
    unittest.main()
