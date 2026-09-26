"""Explicit headless world state for the first Umwelt v0.3 experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite

from umwelt.arena import RectangularArena
from umwelt.errors import DataError
from umwelt.spatial import Point2D


V03_BOUNDARY_BAND_FRACTION = 0.10
V03_WORLD_ID = "roche-open-field-v1"


class WorldZone(StrEnum):
    """Geometric zones declared by the first behavioral-world experiment."""

    BOUNDARY_BAND = "boundary-band"
    INTERIOR = "interior"


@dataclass(frozen=True, slots=True)
class ArenaScaleCondition:
    """One explicit isotropic arena-scale intervention condition."""

    condition_id: str
    scale: float

    def __post_init__(self) -> None:
        if not self.condition_id.strip():
            raise DataError("A world condition must have a non-empty identifier.")
        if (
            isinstance(self.scale, bool)
            or not isfinite(self.scale)
            or self.scale <= 0.0
        ):
            raise DataError("Arena scale must be finite and positive.")

    def to_dict(self) -> dict[str, object]:
        """Return the intervention in a provenance-ready representation."""

        return {
            "condition_id": self.condition_id,
            "intervention": "isotropic-arena-scale",
            "scale": self.scale,
        }


V03_ARENA_SCALE_CONDITIONS = (
    ArenaScaleCondition("compact", 0.75),
    ArenaScaleCondition("recorded", 1.00),
    ArenaScaleCondition("expanded", 1.25),
)


@dataclass(frozen=True, slots=True)
class OpenFieldWorld:
    """A rectangular world with one explicit boundary-zone dependency."""

    world_id: str
    source_arena: RectangularArena
    condition: ArenaScaleCondition
    boundary_band_fraction: float = V03_BOUNDARY_BAND_FRACTION
    _arena: RectangularArena = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.world_id.strip():
            raise DataError("A behavioral world must have a non-empty identifier.")
        if not isinstance(self.source_arena, RectangularArena):
            raise DataError("An open-field world must declare its source arena.")
        if not isinstance(self.condition, ArenaScaleCondition):
            raise DataError("An open-field world must declare its condition.")
        if (
            isinstance(self.boundary_band_fraction, bool)
            or not isfinite(self.boundary_band_fraction)
            or not 0.0 < self.boundary_band_fraction < 0.5
        ):
            raise DataError("Boundary-band fraction must be finite and in (0, 0.5).")
        x_center = self.source_arena.minimum.x + self.source_arena.width / 2.0
        y_center = self.source_arena.minimum.y + self.source_arena.height / 2.0
        x_radius = self.source_arena.width * self.condition.scale / 2.0
        y_radius = self.source_arena.height * self.condition.scale / 2.0
        object.__setattr__(
            self,
            "_arena",
            RectangularArena(
                arena_id=(
                    f"{self.source_arena.arena_id}:condition:"
                    f"{self.condition.condition_id}"
                ),
                coordinate_frame=self.source_arena.coordinate_frame,
                minimum=Point2D(x_center - x_radius, y_center - y_radius),
                maximum=Point2D(x_center + x_radius, y_center + y_radius),
            ),
        )

    @property
    def arena(self) -> RectangularArena:
        """Return geometry scaled once around the recorded arena center."""

        return self._arena

    def clearance_fraction(self, point: Point2D) -> float:
        """Return nearest-boundary clearance relative to the shorter arena side."""

        arena = self.arena
        return arena.distance_to_boundary(point) / min(arena.width, arena.height)

    def zone(self, point: Point2D) -> WorldZone:
        """Classify one in-arena point using the declared boundary fraction."""

        if self.clearance_fraction(point) <= self.boundary_band_fraction:
            return WorldZone.BOUNDARY_BAND
        return WorldZone.INTERIOR

    def to_dict(self) -> dict[str, object]:
        """Return world geometry, condition, and zones without implied semantics."""

        arena = self.arena
        return {
            "world_id": self.world_id,
            "world_instance_id": f"{self.world_id}:{self.condition.condition_id}",
            "source_arena_id": self.source_arena.arena_id,
            "condition": self.condition.to_dict(),
            "coordinate_frame": {
                "frame_id": arena.coordinate_frame.frame_id,
                "unit": arena.coordinate_frame.unit,
                "axis_orientation": arena.coordinate_frame.axis_orientation.value,
            },
            "geometry": {
                "type": "axis-aligned-rectangle",
                "minimum": {"x": arena.minimum.x, "y": arena.minimum.y},
                "maximum": {"x": arena.maximum.x, "y": arena.maximum.y},
                "width": arena.width,
                "height": arena.height,
            },
            "zones": {
                WorldZone.BOUNDARY_BAND.value: {
                    "definition": "clearance-fraction-less-than-or-equal",
                    "threshold": self.boundary_band_fraction,
                },
                WorldZone.INTERIOR.value: {
                    "definition": "clearance-fraction-greater-than",
                    "threshold": self.boundary_band_fraction,
                },
            },
        }
