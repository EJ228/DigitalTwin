"""
DigitalTwin.ai -- a live digital twin of a mixed-model vehicle assembly line.

Accenture Innovation Challenge 2026, Round 2, Track 4. Team ACE, IIT Guwahati.

Two engines on one twin, both driven by the same event log:

    FLOW     BottleneckWalkDetector -- where the constraint is, right now
    QUALITY  T2CUSUM                -- a fixture drifting before defects exist

Both work from the one signal every station emits: the timestamp of a part
entering and leaving. That is what lets the twin cover the 37% of stations
with sparse or no instrumentation.
"""

from .adapters import BoschAdapter, SimAdapter
from .detectors import (
    ActivePeriodDetector,
    BottleneckWalkDetector,
    QueueLengthDetector,
    UtilisationDetector,
)
from .line_config import TAKT_SECONDS, Station, TagSpec, build_line
from .reconstruct import LineReconstruction
from .schema import EventType, LineAdapter, LineSnapshot, SensorTier, StationState
from .simulator import AssemblyLineSim
from .spc import (
    MEWMA,
    CrosierMCUSUM,
    HotellingT2,
    Reference,
    SpecLimitDetector,
    T2CUSUM,
    UnivariateCUSUM,
    calibrate_threshold,
)

__version__ = "0.3.0"

__all__ = [
    "AssemblyLineSim", "LineReconstruction", "build_line", "Station", "TagSpec",
    "TAKT_SECONDS", "EventType", "SensorTier", "StationState", "LineAdapter",
    "LineSnapshot", "SimAdapter", "BoschAdapter", "BottleneckWalkDetector",
    "ActivePeriodDetector", "UtilisationDetector", "QueueLengthDetector",
    "Reference", "SpecLimitDetector", "UnivariateCUSUM", "HotellingT2",
    "MEWMA", "CrosierMCUSUM", "T2CUSUM", "calibrate_threshold", "__version__",
]
