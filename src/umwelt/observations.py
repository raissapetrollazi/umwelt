"""Explicit representations of recorded and synthetic observations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from math import isfinite
from typing import Iterator

from umwelt.errors import DataError


class ObservationSource(StrEnum):
    """The origin category of an observation."""

    RECORDED = "recorded"
    SYNTHETIC = "synthetic"


class BehavioralState(StrEnum):
    """Observable sleep/activity states used by the v0.1 experiment."""

    WAKE = "wake"
    SLEEP = "sleep"


@dataclass(frozen=True, slots=True)
class Observation:
    """One time-indexed recorded or synthetic observation."""

    timestamp: datetime
    subject_id: str
    source: ObservationSource
    state: BehavioralState
    activity: float

    def __post_init__(self) -> None:
        if not self.subject_id:
            raise DataError("An observation must have a subject identifier.")
        if not isfinite(self.activity) or self.activity < 0:
            raise DataError("Observation activity must be finite and non-negative.")


@dataclass(frozen=True, slots=True)
class TimeSeries:
    """A regular observation series in which source-data gaps remain explicit."""

    subject_id: str
    source: ObservationSource
    timestamps: tuple[datetime, ...]
    states: tuple[BehavioralState | None, ...]
    activities: tuple[float | None, ...]

    def __post_init__(self) -> None:
        lengths = {len(self.timestamps), len(self.states), len(self.activities)}
        if len(lengths) != 1:
            raise DataError("Time-series fields must have equal lengths.")
        if not self.subject_id:
            raise DataError("A time series must have a subject identifier.")

        previous: datetime | None = None
        for timestamp, state, activity in zip(
            self.timestamps, self.states, self.activities, strict=True
        ):
            if previous is not None and timestamp <= previous:
                raise DataError("Time-series timestamps must be strictly increasing.")
            previous = timestamp
            if (state is None) != (activity is None):
                raise DataError("State and activity must be missing together.")
            if activity is not None and (not isfinite(activity) or activity < 0):
                raise DataError("Time-series activity must be finite and non-negative.")

    @property
    def valid_epoch_count(self) -> int:
        """Return the number of epochs containing an observation."""

        return sum(state is not None for state in self.states)

    def observations(self) -> Iterator[Observation]:
        """Yield present observations while retaining their source category."""

        for timestamp, state, activity in zip(
            self.timestamps, self.states, self.activities, strict=True
        ):
            if state is None or activity is None:
                continue
            yield Observation(
                timestamp=timestamp,
                subject_id=self.subject_id,
                source=self.source,
                state=state,
                activity=activity,
            )


@dataclass(frozen=True, slots=True)
class ObservationDataset:
    """A collection of observation series with shared provenance."""

    dataset_id: str
    series: tuple[TimeSeries, ...]
    provenance: dict[str, object]

    def __post_init__(self) -> None:
        if not self.dataset_id:
            raise DataError("A dataset must have an identifier.")
        identifiers = [item.subject_id for item in self.series]
        if len(identifiers) != len(set(identifiers)):
            raise DataError("Dataset subject identifiers must be unique.")
        sources = {item.source for item in self.series}
        if len(sources) > 1:
            raise DataError("A dataset cannot silently mix observation sources.")

    def subject(self, subject_id: str) -> TimeSeries:
        """Return one series by identifier."""

        for item in self.series:
            if item.subject_id == subject_id:
                return item
        raise DataError(f"Unknown subject identifier: {subject_id}")
