"""Tests for the explicit Umwelt v0.3 open-field world contract."""

from __future__ import annotations

import unittest

from umwelt.arena import RectangularArena
from umwelt.errors import DataError
from umwelt.spatial import AxisOrientation, CoordinateFrame, Point2D
from umwelt.world import (
    V03_ARENA_SCALE_CONDITIONS,
    V03_BOUNDARY_BAND_FRACTION,
    ArenaScaleCondition,
    OpenFieldWorld,
    WorldZone,
)


class OpenFieldWorldTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = CoordinateFrame(
            "roche-image", "px", AxisOrientation.X_RIGHT_Y_DOWN
        )
        self.recorded_arena = RectangularArena(
            arena_id="recording-1",
            coordinate_frame=self.frame,
            minimum=Point2D(10.0, 20.0),
            maximum=Point2D(110.0, 220.0),
        )

    def _world(
        self,
        condition: ArenaScaleCondition | None = None,
        boundary_band_fraction: float = V03_BOUNDARY_BAND_FRACTION,
    ) -> OpenFieldWorld:
        return OpenFieldWorld(
            world_id="roche-open-field-v1",
            source_arena=self.recorded_arena,
            condition=condition or ArenaScaleCondition("recorded", 1.0),
            boundary_band_fraction=boundary_band_fraction,
        )

    def test_declared_conditions_are_ordered_and_exact(self) -> None:
        self.assertEqual(
            tuple(condition.condition_id for condition in V03_ARENA_SCALE_CONDITIONS),
            ("compact", "recorded", "expanded"),
        )
        self.assertEqual(
            tuple(condition.scale for condition in V03_ARENA_SCALE_CONDITIONS),
            (0.75, 1.0, 1.25),
        )

    def test_condition_validation_rejects_implicit_or_invalid_values(self) -> None:
        with self.assertRaisesRegex(DataError, "non-empty identifier"):
            ArenaScaleCondition(" ", 1.0)
        for value in (True, 0.0, -1.0, float("inf"), float("nan")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(DataError, "finite and positive"):
                    ArenaScaleCondition("invalid", value)  # type: ignore[arg-type]

    def test_scaling_is_centered_isotropic_and_preserves_context(self) -> None:
        compact_world = self._world(ArenaScaleCondition("compact", 0.75))
        compact = compact_world.arena
        expanded = self._world(ArenaScaleCondition("expanded", 1.25)).arena

        self.assertIs(compact_world.arena, compact)
        self.assertEqual(compact.minimum, Point2D(22.5, 45.0))
        self.assertEqual(compact.maximum, Point2D(97.5, 195.0))
        self.assertEqual(expanded.minimum, Point2D(-2.5, -5.0))
        self.assertEqual(expanded.maximum, Point2D(122.5, 245.0))
        self.assertEqual(compact.width / compact.height, 0.5)
        self.assertEqual(expanded.width / expanded.height, 0.5)
        self.assertEqual(compact.coordinate_frame, self.frame)

    def test_zone_uses_shorter_side_and_includes_threshold(self) -> None:
        world = self._world()

        self.assertEqual(world.clearance_fraction(Point2D(20.0, 100.0)), 0.10)
        self.assertEqual(world.zone(Point2D(20.0, 100.0)), WorldZone.BOUNDARY_BAND)
        self.assertEqual(world.zone(Point2D(20.1, 100.0)), WorldZone.INTERIOR)
        self.assertEqual(world.zone(Point2D(60.0, 120.0)), WorldZone.INTERIOR)
        self.assertEqual(world.zone(Point2D(110.0, 120.0)), WorldZone.BOUNDARY_BAND)

    def test_zone_rejects_points_outside_condition_geometry(self) -> None:
        compact = self._world(ArenaScaleCondition("compact", 0.75))

        with self.assertRaisesRegex(DataError, "inside the arena"):
            compact.zone(Point2D(10.0, 20.0))

    def test_world_validation_keeps_the_boundary_partition_explicit(self) -> None:
        with self.assertRaisesRegex(DataError, "non-empty identifier"):
            OpenFieldWorld(
                " ",
                self.recorded_arena,
                ArenaScaleCondition("recorded", 1.0),
            )
        for value in (True, 0.0, 0.5, float("nan")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(DataError, "finite and in"):
                    self._world(boundary_band_fraction=value)  # type: ignore[arg-type]

    def test_world_serialization_separates_source_condition_and_geometry(self) -> None:
        serialized = self._world(ArenaScaleCondition("compact", 0.75)).to_dict()

        self.assertEqual(serialized["world_id"], "roche-open-field-v1")
        self.assertEqual(serialized["world_instance_id"], "roche-open-field-v1:compact")
        self.assertEqual(serialized["source_arena_id"], "recording-1")
        self.assertEqual(
            serialized["condition"],
            {
                "condition_id": "compact",
                "intervention": "isotropic-arena-scale",
                "scale": 0.75,
            },
        )
        self.assertEqual(
            serialized["coordinate_frame"],
            {
                "frame_id": "roche-image",
                "unit": "px",
                "axis_orientation": "x-right-y-down",
            },
        )
        self.assertEqual(
            serialized["geometry"],
            {
                "type": "axis-aligned-rectangle",
                "minimum": {"x": 22.5, "y": 45.0},
                "maximum": {"x": 97.5, "y": 195.0},
                "width": 75.0,
                "height": 150.0,
            },
        )
        self.assertEqual(
            serialized["zones"],
            {
                "boundary-band": {
                    "definition": "clearance-fraction-less-than-or-equal",
                    "threshold": 0.10,
                },
                "interior": {
                    "definition": "clearance-fraction-greater-than",
                    "threshold": 0.10,
                },
            },
        )


if __name__ == "__main__":
    unittest.main()
