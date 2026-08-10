"""Tests for the pre-result v0.2 spatial protocol facts."""

from __future__ import annotations

import unittest

from umwelt.spatial_protocol import (
    ROCHE_CONTROL_DEVELOPMENT_SUBJECTS,
    ROCHE_CONTROL_FITTING_SUBJECTS,
    ROCHE_CONTROL_SPLIT_HASH_PREFIX,
    ROCHE_CONTROL_SUBJECT_IDS,
    derive_roche_control_split,
)


class SpatialProtocolTests(unittest.TestCase):
    def test_outcome_blind_split_is_exact_and_animal_disjoint(self) -> None:
        fitting, development = derive_roche_control_split()

        self.assertEqual(
            ROCHE_CONTROL_SPLIT_HASH_PREFIX,
            "umwelt-v0.2-control-split-v1|096e21e4d319130370aa4bb244b670f8",
        )
        self.assertEqual(
            fitting,
            (
                "16459-67049",
                "16459-67053",
                "16459-67060",
                "16459-67067",
                "16459-67071",
                "16459-67074",
            ),
        )
        self.assertEqual(development, ("16459-67064", "16459-67078"))
        self.assertEqual(fitting, ROCHE_CONTROL_FITTING_SUBJECTS)
        self.assertEqual(development, ROCHE_CONTROL_DEVELOPMENT_SUBJECTS)
        self.assertFalse(set(fitting) & set(development))
        self.assertEqual(
            set(fitting) | set(development), set(ROCHE_CONTROL_SUBJECT_IDS)
        )


if __name__ == "__main__":
    unittest.main()
