"""Tests for explicit training-population individual variation."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta

from umwelt.errors import DataError
from umwelt.individual_variation import (
    POPULATION_MODEL_ID,
    IndividualVariationProfile,
    PopulationTemporalModel,
    derive_profile_seed,
    fit_population_temporal_model,
    simulate_pooled_from_template,
    simulate_population_from_template,
)
from umwelt.models.temporal_hazard import (
    PHASE_DURATION_MODEL,
    fit_temporal_model_ladder,
)
from umwelt.observations import (
    BehavioralState,
    ObservationDataset,
    ObservationSource,
    TimeSeries,
)


def _series(
    subject_id: str,
    *,
    wake_epochs: int,
    sleep_epochs: int,
    invert_values: bool = False,
) -> TimeSeries:
    start = datetime.fromisoformat("2020-01-01 00:00:00")
    timestamps = tuple(start + timedelta(seconds=10 * index) for index in range(240))
    cycle = wake_epochs + sleep_epochs
    states = []
    activities = []
    for index in range(240):
        if index in {71, 155}:
            states.append(None)
            activities.append(None)
            continue
        sleep = index % cycle >= wake_epochs
        if invert_values:
            sleep = not sleep
        state = BehavioralState.SLEEP if sleep else BehavioralState.WAKE
        states.append(state)
        activities.append(0.0 if sleep else float(1 + index % 5))
    return TimeSeries(
        subject_id=subject_id,
        source=ObservationSource.RECORDED,
        timestamps=timestamps,
        states=tuple(states),
        activities=tuple(activities),
    )


class IndividualVariationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.training = ObservationDataset(
            dataset_id="training",
            series=(
                _series("1", wake_epochs=8, sleep_epochs=4),
                _series("2", wake_epochs=5, sleep_epochs=7),
                _series("3", wake_epochs=3, sleep_epochs=9),
            ),
            provenance={"source_category": "recorded"},
        )
        ladder = fit_temporal_model_ladder(
            self.training,
            phase_bins=2,
            duration_bin_edges_epochs=(1, 2, 4, 8, 16),
        )
        self.base = next(
            model for model in ladder if model.spec.model_id == PHASE_DURATION_MODEL
        )
        self.population = fit_population_temporal_model(self.training, self.base)

    def test_profiles_are_fit_only_for_base_training_subjects(self) -> None:
        self.assertEqual(self.population.model_id, POPULATION_MODEL_ID)
        self.assertEqual(
            tuple(profile.source_subject_id for profile in self.population.profiles),
            ("1", "2", "3"),
        )
        self.assertEqual(
            self.population.parameter_count,
            self.base.parameter_count + 2 * len(self.training.series),
        )
        offsets = [
            abs(profile.sleep_bias_logit_offset) + abs(profile.switching_logit_offset)
            for profile in self.population.profiles
        ]
        self.assertTrue(any(value > 0 for value in offsets))

    def test_profile_effects_act_throughout_state_dynamics(self) -> None:
        timestamp = self.training.subject("1").timestamps[20]
        sleep_biased = IndividualVariationProfile(
            source_subject_id="fixture",
            sleep_bias_logit_offset=0.75,
            switching_logit_offset=0.0,
            observations=100,
            transitions=99,
        )
        faster_switching = IndividualVariationProfile(
            source_subject_id="fixture",
            sleep_bias_logit_offset=0.0,
            switching_logit_offset=0.75,
            observations=100,
            transitions=99,
        )

        self.assertGreater(
            self.population.initial_sleep_probability(timestamp, sleep_biased),
            self.base.initial_sleep_probabilities[self.base.phase_for(timestamp)],
        )
        self.assertGreater(
            self.population.leave_probability(
                BehavioralState.WAKE, timestamp, 5, sleep_biased
            ),
            self.base.leave_probability(BehavioralState.WAKE, timestamp, 5),
        )
        self.assertLess(
            self.population.leave_probability(
                BehavioralState.SLEEP, timestamp, 5, sleep_biased
            ),
            self.base.leave_probability(BehavioralState.SLEEP, timestamp, 5),
        )
        for state in BehavioralState:
            self.assertGreater(
                self.population.leave_probability(
                    state, timestamp, 5, faster_switching
                ),
                self.base.leave_probability(state, timestamp, 5),
            )

    def test_profile_selection_and_generation_are_deterministic(self) -> None:
        profile_seed = derive_profile_seed(1729, "6", 1)
        self.assertEqual(profile_seed, derive_profile_seed(1729, "6", 1))
        self.assertNotEqual(profile_seed, derive_profile_seed(1729, "6", 2))
        self.assertEqual(
            self.population.profile_for_seed(profile_seed),
            self.population.profile_for_seed(profile_seed),
        )

        template = _series("6", wake_epochs=6, sleep_epochs=6)
        first, first_profile = simulate_population_from_template(
            self.population,
            template,
            subject_id="6",
            state_seed=99,
            activity_seed=199,
            profile_seed=profile_seed,
        )
        second, second_profile = simulate_population_from_template(
            self.population,
            template,
            subject_id="6",
            state_seed=99,
            activity_seed=199,
            profile_seed=profile_seed,
        )
        self.assertEqual(first, second)
        self.assertEqual(first_profile, second_profile)
        self.assertEqual(first.source, ObservationSource.SYNTHETIC)

    def test_activity_randomness_does_not_change_state_trajectory(self) -> None:
        template = _series("6", wake_epochs=6, sleep_epochs=6)
        first = simulate_pooled_from_template(
            self.base,
            template,
            subject_id="6",
            state_seed=99,
            activity_seed=100,
        )
        second = simulate_pooled_from_template(
            self.base,
            template,
            subject_id="6",
            state_seed=99,
            activity_seed=101,
        )

        self.assertEqual(first.states, second.states)
        self.assertNotEqual(first.activities, second.activities)

    def test_paired_streams_match_exactly_for_a_zero_effect_profile(self) -> None:
        template = _series("6", wake_epochs=6, sleep_epochs=6)
        profile = IndividualVariationProfile(
            source_subject_id="fixture",
            sleep_bias_logit_offset=0.0,
            switching_logit_offset=0.0,
            observations=100,
            transitions=99,
        )
        population = PopulationTemporalModel(
            base_model=self.base,
            profiles=(profile,),
        )

        pooled = simulate_pooled_from_template(
            self.base,
            template,
            subject_id="6",
            state_seed=99,
            activity_seed=199,
        )
        generated, selected_profile = simulate_population_from_template(
            population,
            template,
            subject_id="6",
            state_seed=99,
            activity_seed=199,
            profile_seed=299,
        )

        self.assertEqual(generated, pooled)
        self.assertEqual(selected_profile, profile)

    def test_template_behavior_does_not_select_or_calibrate_profile(self) -> None:
        original = _series("6", wake_epochs=6, sleep_epochs=6)
        altered = _series("6", wake_epochs=6, sleep_epochs=6, invert_values=True)
        profile_seed = derive_profile_seed(1729, "6", 1)

        first, first_profile = simulate_population_from_template(
            self.population,
            original,
            subject_id="6",
            state_seed=123,
            activity_seed=456,
            profile_seed=profile_seed,
        )
        second, second_profile = simulate_population_from_template(
            self.population,
            altered,
            subject_id="6",
            state_seed=123,
            activity_seed=456,
            profile_seed=profile_seed,
        )

        self.assertEqual(first_profile, second_profile)
        self.assertEqual(first.timestamps, second.timestamps)
        self.assertEqual(first.states, second.states)
        self.assertEqual(first.activities, second.activities)

    def test_training_mismatch_is_rejected(self) -> None:
        mismatched = ObservationDataset(
            dataset_id="mismatch",
            series=self.training.series[:2],
            provenance={"source_category": "recorded"},
        )
        with self.assertRaisesRegex(DataError, "must exactly match"):
            fit_population_temporal_model(mismatched, self.base)


if __name__ == "__main__":
    unittest.main()
