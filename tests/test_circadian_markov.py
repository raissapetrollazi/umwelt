"""Tests for the explicit v0.1 generative model."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta

from umwelt.errors import DataError
from umwelt.models.circadian_markov import (
    MODEL_TYPE,
    fit_phase_conditioned_markov,
    simulate_from_template,
)
from umwelt.observations import (
    BehavioralState,
    ObservationDataset,
    ObservationSource,
    TimeSeries,
)


def _series(subject_id: str, source: ObservationSource) -> TimeSeries:
    start = datetime.fromisoformat("2020-01-01 00:00:00")
    timestamps = tuple(start + timedelta(seconds=10 * index) for index in range(12))
    states = (
        BehavioralState.WAKE,
        BehavioralState.WAKE,
        BehavioralState.SLEEP,
        BehavioralState.SLEEP,
        BehavioralState.SLEEP,
        BehavioralState.WAKE,
        None,
        BehavioralState.WAKE,
        BehavioralState.SLEEP,
        BehavioralState.SLEEP,
        BehavioralState.WAKE,
        BehavioralState.WAKE,
    )
    activities = (8.0, 2.0, 0.0, 0.0, 0.0, 4.0, None, 0.0, 0.0, 0.0, 5.0, 9.0)
    return TimeSeries(subject_id, source, timestamps, states, activities)


class CircadianMarkovTests(unittest.TestCase):
    def setUp(self) -> None:
        self.recorded = ObservationDataset(
            dataset_id="fixture",
            series=(
                _series("1", ObservationSource.RECORDED),
                _series("2", ObservationSource.RECORDED),
            ),
            provenance={"source_category": "recorded"},
        )

    def test_fit_produces_inspectable_phase_parameters(self) -> None:
        model = fit_phase_conditioned_markov(self.recorded, phase_bins=2)

        artifact = model.to_dict()
        self.assertEqual(artifact["model_type"], MODEL_TYPE)
        self.assertEqual(model.epoch_seconds, 10)
        self.assertEqual(model.training_observations, 22)
        self.assertEqual(model.training_transitions, 18)
        self.assertEqual(len(model.phases), 2)
        self.assertTrue(0 <= model.phases[0].wake_to_sleep_probability <= 1)
        self.assertIn("assumptions", artifact)

    def test_simulation_is_reproducible_and_explicitly_synthetic(self) -> None:
        model = fit_phase_conditioned_markov(self.recorded, phase_bins=1)
        template = self.recorded.subject("1")

        first = simulate_from_template(
            model, template, subject_id="synthetic-1", seed=1729
        )
        second = simulate_from_template(
            model, template, subject_id="synthetic-1", seed=1729
        )

        self.assertEqual(first, second)
        self.assertEqual(first.source, ObservationSource.SYNTHETIC)
        self.assertIsNone(first.states[6])
        for state, activity in zip(first.states, first.activities, strict=True):
            if state is BehavioralState.SLEEP:
                self.assertEqual(activity, 0.0)

    def test_different_seeds_can_generate_different_sequences(self) -> None:
        model = fit_phase_conditioned_markov(self.recorded, phase_bins=1)
        template = self.recorded.subject("1")

        first = simulate_from_template(model, template, subject_id="a", seed=1)
        second = simulate_from_template(model, template, subject_id="b", seed=2)

        self.assertNotEqual(first.states, second.states)

    def test_fit_rejects_synthetic_training_data(self) -> None:
        dataset = ObservationDataset(
            dataset_id="synthetic-fixture",
            series=(_series("synthetic", ObservationSource.SYNTHETIC),),
            provenance={"source_category": "synthetic"},
        )
        with self.assertRaisesRegex(DataError, "recorded observations"):
            fit_phase_conditioned_markov(dataset)

    def test_fit_rejects_non_positive_priors(self) -> None:
        with self.assertRaisesRegex(DataError, "priors must be positive"):
            fit_phase_conditioned_markov(self.recorded, transition_prior=0)


if __name__ == "__main__":
    unittest.main()
