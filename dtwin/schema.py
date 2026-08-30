"""
Frozen event schema and adapter interface.

THE CONTRACT
------------
Every model in this project consumes ONLY `events` (+ optionally `tags`).
`truth` is written by the simulator and read ONLY by the scorer.
No detector, forecaster or classifier may import from a truth table.

This separation is enforced by convention here and asserted in
`dtwin.audit.assert_no_truth_leak()`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Iterator

import pandas as pd

SCHEMA_VERSION = "1.0.0"


class EventType(str, Enum):
    """The one signal every station emits, regardless of instrumentation."""

    ENTER = "enter"        # part begins service at station
    EXIT = "exit"          # part physically leaves station (post-blocking)
    INSPECT = "inspect"    # end-of-line / in-line inspection verdict
    REWORK_IN = "rework_in"
    REWORK_OUT = "rework_out"


class StationState(str, Enum):
    """Per-station timeline states. ACTIVE = station-internal constraint.

    Roser's active-period method treats WORKING and DOWN as 'active'
    (the station is constrained by itself), while BLOCKED and STARVED
    are inactive (constrained by another station).
    """

    WORKING = "working"
    DOWN = "down"          # breakdown / repair
    BLOCKED = "blocked"    # downstream buffer full
    STARVED = "starved"    # upstream buffer empty


ACTIVE_STATES = {StationState.WORKING.value, StationState.DOWN.value}


class SensorTier(str, Enum):
    RICH = "rich"          # hundreds of OPC-UA tags, every part
    SPARSE = "sparse"      # a handful of tags, irregular sampling
    BLIND = "blind"        # manual / legacy bay: timestamps only


# --------------------------------------------------------------------------
# Table shapes
# --------------------------------------------------------------------------

EVENT_COLUMNS = [
    "t",             # float seconds since line start
    "part_id",       # int
    "station_id",    # str, e.g. "S08"
    "event_type",    # EventType
    "variant",       # str, model variant
]

TAG_COLUMNS = [
    "t",
    "part_id",
    "station_id",
    "tag",           # str
    "value",         # float
]

STATE_COLUMNS = [
    "station_id",
    "state",
    "t_start",
    "t_end",
]

# truth tables -------------------------------------------------------------

TRUTH_BOTTLENECK_COLUMNS = [
    "t",
    "true_bottleneck",       # station_id with max theoretical load at t
    "true_load",             # that load value
    "runner_up",             # second-highest, for margin analysis
    "margin",                # true_load - runner_up_load
]

TRUTH_DEFECT_COLUMNS = [
    "part_id",
    "is_defective",
    "cause_station",
    "cause_mechanism",       # e.g. "fixture_twist"
    "severity",
]

TRUTH_DRIFT_COLUMNS = [
    "station_id",
    "t_onset",
    "t_full",                # ramp completion
    "mechanism",
    "affected_tags",
]


# --------------------------------------------------------------------------
# Adapter interface
# --------------------------------------------------------------------------

@dataclass
class LineSnapshot:
    """Everything a substrate must be able to hand over."""

    events: pd.DataFrame
    tags: pd.DataFrame
    states: pd.DataFrame | None = None      # simulator-only; None for real logs
    meta: dict | None = None


class LineAdapter(ABC):
    """One interface, several substrates.

    Implementations
    ---------------
    SimAdapter    -- our SimPy line (has ground truth, primary substrate)
    BoschAdapter  -- Bosch Production Line Performance (real, no flow truth)
    """

    name: str

    @abstractmethod
    def load(self) -> LineSnapshot:
        ...

    @abstractmethod
    def stations(self) -> list[str]:
        ...

    def iter_events(self) -> Iterator[dict]:
        """Replay in timestamp order — used by the live twin."""
        snap = self.load()
        for row in snap.events.sort_values("t").itertuples(index=False):
            yield row._asdict()
