"""Explicit headless arena geometry for Umwelt v0.2."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from umwelt.errors import DataError
from umwelt.spatial import CoordinateFrame, Point2D


@dataclass(frozen=True, slots=True)
class RectangularArena:
    """Axis-aligned rectangular arena in one explicit coordinate frame."""

    arena_id: str
    coordinate_frame: CoordinateFrame
    minimum: Point2D
    maximum: Point2D

    def __post_init__(self) -> None:
        if not self.arena_id.strip():
            raise DataError("An arena must have a non-empty identifier.")
        if self.maximum.x <= self.minimum.x or self.maximum.y <= self.minimum.y:
            raise DataError("Arena maximum bounds must exceed minimum bounds.")

    @property
    def width(self) -> float:
        """Return arena width in the coordinate frame's declared unit."""

        return self.maximum.x - self.minimum.x

    @property
    def height(self) -> float:
        """Return arena height in the coordinate frame's declared unit."""

        return self.maximum.y - self.minimum.y

    def contains(self, point: Point2D, *, include_boundary: bool = True) -> bool:
        """Return whether a point lies within the arena bounds."""

        if include_boundary:
            return (
                self.minimum.x <= point.x <= self.maximum.x
                and self.minimum.y <= point.y <= self.maximum.y
            )
        return (
            self.minimum.x < point.x < self.maximum.x
            and self.minimum.y < point.y < self.maximum.y
        )

    def distance_to_boundary(self, point: Point2D) -> float:
        """Return shortest in-arena distance to a rectangular boundary."""

        if not self.contains(point):
            raise DataError("Boundary distance requires a point inside the arena.")
        return min(
            point.x - self.minimum.x,
            self.maximum.x - point.x,
            point.y - self.minimum.y,
            self.maximum.y - point.y,
        )

    def center_region(self, *, margin_fraction: float) -> RectangularArena:
        """Return a concentric geometric center region defined by wall margins."""

        if not isfinite(margin_fraction) or not 0.0 <= margin_fraction < 0.5:
            raise DataError("Center-region margin fraction must be in [0, 0.5).")
        x_margin = self.width * margin_fraction
        y_margin = self.height * margin_fraction
        return RectangularArena(
            arena_id=f"{self.arena_id}:center:{margin_fraction:g}",
            coordinate_frame=self.coordinate_frame,
            minimum=Point2D(self.minimum.x + x_margin, self.minimum.y + y_margin),
            maximum=Point2D(self.maximum.x - x_margin, self.maximum.y - y_margin),
        )
