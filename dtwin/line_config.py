"""
Line configuration for a mixed-model vehicle assembly line.

ASSUMPTIONS REGISTER
--------------------
Every number here is an assumption, not a measurement. Rationale is inline so
a reviewer can challenge any single one.

  takt 60 s                -> 60 units/hour, the rate the Round-1 deck assumes
  35 stations              -> brief: "roughly 30-50 across body, paint, final"
  target load 55.5 s       -> 92.5% planned utilisation; high enough that
                              buffers and blocking matter, low enough to run
  balance offsets +/-2%    -> real lines are line-balanced to a few percent.
                              This is WHY the bottleneck shifts: no station is
                              structurally dominant, so small perturbations
                              decide which one binds. A line with one obviously
                              slowest station would make the problem trivial
                              and would not resemble a real one.
  buffer capacity 2-6      -> small inter-station banks, larger banks between
                              zones and around the paint oven
  sensor tiers 60/22/18%   -> brief: "majority well-instrumented, meaningful
                              minority reliant on manual checks". Sparse+blind
                              = 40%, matching the Round-1 design envelope.
  Cpk 1.33                 -> spec limits at +/-4 sigma. This is what makes the
                              drift scenario honest: a sub-sigma mean shift
                              cannot realistically breach a 4-sigma limit.
  defect base rate ~0.6%   -> anchored to Bosch Production Line Performance,
                              whose observed rate is just under 1%
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .schema import SensorTier

TAKT_SECONDS = 60.0

# Mix-weighted station load the line is balanced to, in seconds per part.
#
# 50 s against a 60 s takt is 83% planned utilisation. This is deliberate. At
# 92% every station runs near-continuously, so there is no momentary bottleneck
# to find and the question is ill-posed. At 83% the line has slack in normal
# operation -- correctly ambiguous -- and a disruption creates an unambiguous
# constraint. "When there IS a bottleneck, do you find it, and how fast" is the
# question worth scoring.
TARGET_LOAD_SECONDS = 50.0

# Breakdowns are real, but with MTTR in the hundreds of seconds the random
# downtime swamps every injected episode and the momentary bottleneck becomes
# whichever station happens to be broken. Scaled so availability sits near 99%:
# breakdowns stay present as realistic noise without becoming the whole signal.
MTBF_SCALE = 1.6
MTTR_SCALE = 0.32


@dataclass
class TagSpec:
    """A process tag with engineering spec limits.

    sigma is the natural process standard deviation. Spec limits sit at
    mu0 +/- cpk_sigma * sigma  (Cpk 1.33 -> 4 sigma).
    """

    name: str
    mu0: float
    sigma: float
    unit: str
    cpk_sigma: float = 4.0

    @property
    def lsl(self) -> float:
        return self.mu0 - self.cpk_sigma * self.sigma

    @property
    def usl(self) -> float:
        return self.mu0 + self.cpk_sigma * self.sigma


@dataclass
class Station:
    station_id: str
    zone: str
    mean_service: float          # seconds, variant A, derived from target load
    cv: float                    # coefficient of variation of service time
    buffer_out: int              # capacity of buffer downstream of this station
    mtbf: float                  # seconds of busy time between failures
    mttr: float                  # mean repair seconds
    tier: SensorTier
    balance_offset: float = 0.0  # fractional deviation from the balanced target
    tags: list[TagSpec] = field(default_factory=list)
    sparse_every: int = 1        # sample tags on every k-th part (sparse tier)
    manual: bool = False

    @property
    def availability(self) -> float:
        return self.mtbf / (self.mtbf + self.mttr)


# --------------------------------------------------------------------------
# Tag libraries per zone
# --------------------------------------------------------------------------

def _body_tags(prefix: str) -> list[TagSpec]:
    return [
        TagSpec(f"{prefix}_clamp_force_N", 4800.0, 60.0, "N"),
        TagSpec(f"{prefix}_gap_left_mm", 1.50, 0.040, "mm"),
        TagSpec(f"{prefix}_gap_right_mm", 1.50, 0.040, "mm"),
        TagSpec(f"{prefix}_weld_current_A", 9500.0, 110.0, "A"),
        TagSpec(f"{prefix}_weld_time_ms", 280.0, 6.0, "ms"),
        TagSpec(f"{prefix}_electrode_wear_pct", 22.0, 3.0, "%"),
    ]


def _paint_tags(prefix: str) -> list[TagSpec]:
    return [
        TagSpec(f"{prefix}_booth_temp_C", 23.0, 0.6, "C"),
        TagSpec(f"{prefix}_humidity_pct", 55.0, 2.5, "%"),
        TagSpec(f"{prefix}_flow_rate_ml_min", 320.0, 9.0, "ml/min"),
        TagSpec(f"{prefix}_film_thickness_um", 105.0, 3.5, "um"),
    ]


def _final_tags(prefix: str) -> list[TagSpec]:
    return [
        TagSpec(f"{prefix}_torque_Nm", 45.0, 1.4, "Nm"),
        TagSpec(f"{prefix}_angle_deg", 92.0, 2.2, "deg"),
        TagSpec(f"{prefix}_leak_rate_ccm", 4.0, 0.5, "ccm"),
    ]


# --------------------------------------------------------------------------
# The line
# --------------------------------------------------------------------------

# (station_id, zone, balance_offset, cv, buffer_out, mtbf, mttr, tier, sparse_every)
#
# balance_offset is how far this station sits from the balanced target load.
# Everything is within +/-2%, so the identity of the baseline bottleneck is
# genuinely marginal -- which is the realistic and the difficult case.
_LINE_SPEC = [
    # ---- BODY CONSTRUCTION: 12 stations, heavily automated -----------------
    ("S01", "body", -0.012, 0.09, 3, 14400, 300, SensorTier.RICH, 1),
    ("S02", "body", +0.006, 0.09, 3, 12600, 320, SensorTier.RICH, 1),
    ("S03", "body", -0.018, 0.10, 2, 14400, 280, SensorTier.SPARSE, 4),
    ("S04", "body", +0.011, 0.09, 3, 10800, 360, SensorTier.RICH, 1),
    ("S05", "body", -0.004, 0.11, 2, 14400, 300, SensorTier.RICH, 1),
    ("S06", "body", +0.014, 0.10, 3,  9000, 420, SensorTier.RICH, 1),
    ("S07", "body", -0.009, 0.09, 2, 12600, 300, SensorTier.SPARSE, 3),
    ("S08", "body", +0.008, 0.10, 3, 11000, 340, SensorTier.RICH, 1),   # drift station
    ("S09", "body", -0.006, 0.12, 2, 14400, 280, SensorTier.BLIND, 0),  # manual fit check
    ("S10", "body", +0.003, 0.09, 3, 12600, 300, SensorTier.RICH, 1),
    ("S11", "body", -0.015, 0.10, 2, 14400, 260, SensorTier.RICH, 1),
    ("S12", "body", +0.009, 0.11, 6, 10800, 380, SensorTier.SPARSE, 4),  # zone exit bank

    # ---- PAINT: 10 stations, process-heavy, larger banks -------------------
    ("S13", "paint", -0.007, 0.08, 4, 18000, 400, SensorTier.RICH, 1),
    ("S14", "paint", +0.012, 0.08, 4, 18000, 420, SensorTier.RICH, 1),
    ("S15", "paint", -0.014, 0.09, 3, 16200, 380, SensorTier.SPARSE, 5),
    ("S16", "paint", +0.016, 0.07, 6, 21600, 500, SensorTier.RICH, 1),   # oven
    ("S17", "paint", +0.001, 0.08, 4, 18000, 400, SensorTier.RICH, 1),
    ("S18", "paint", -0.010, 0.09, 3, 16200, 360, SensorTier.BLIND, 0),  # manual sand
    ("S19", "paint", +0.010, 0.08, 4, 18000, 420, SensorTier.RICH, 1),
    ("S20", "paint", -0.005, 0.09, 3, 16200, 380, SensorTier.RICH, 1),
    ("S21", "paint", +0.007, 0.08, 4, 18000, 400, SensorTier.SPARSE, 4),
    ("S22", "paint", -0.016, 0.10, 6, 14400, 340, SensorTier.RICH, 1),   # zone exit bank

    # ---- FINAL ASSEMBLY: 13 stations, most manual content ------------------
    ("S23", "final", +0.013, 0.14, 3, 21600, 240, SensorTier.BLIND, 0),  # manual trim
    ("S24", "final", -0.008, 0.10, 2, 14400, 300, SensorTier.RICH, 1),
    ("S25", "final", +0.005, 0.13, 3, 18000, 260, SensorTier.SPARSE, 3),
    ("S26", "final", -0.013, 0.15, 2, 21600, 220, SensorTier.BLIND, 0),  # manual harness
    ("S27", "final", +0.009, 0.10, 3, 14400, 320, SensorTier.RICH, 1),
    ("S28", "final", -0.002, 0.11, 2, 16200, 280, SensorTier.RICH, 1),
    ("S29", "final", +0.015, 0.12, 3, 12600, 340, SensorTier.RICH, 1),
    ("S30", "final", -0.011, 0.16, 2, 21600, 200, SensorTier.BLIND, 0),  # manual interior
    ("S31", "final", +0.004, 0.10, 3, 14400, 300, SensorTier.RICH, 1),
    ("S32", "final", -0.017, 0.13, 2, 18000, 260, SensorTier.SPARSE, 4),
    ("S33", "final", +0.002, 0.11, 3, 14400, 320, SensorTier.RICH, 1),
    ("S34", "final", -0.006, 0.14, 2, 21600, 220, SensorTier.BLIND, 0),  # manual finish
    ("S35", "final", -0.020, 0.08, 4, 28800, 180, SensorTier.RICH, 1),   # EOL INSPECTION
]

EOL_STATION = "S35"
DRIFT_STATION = "S08"


# --------------------------------------------------------------------------
# Mixed-model variants
# --------------------------------------------------------------------------

VARIANTS = {
    "A": {"mix": 0.50, "body": 1.00, "paint": 1.00, "final": 1.00},
    "B": {"mix": 0.32, "body": 1.02, "paint": 1.00, "final": 1.09},
    "C": {"mix": 0.18, "body": 1.05, "paint": 1.03, "final": 1.14},
}


def variant_multiplier(variant: str, zone: str) -> float:
    return VARIANTS[variant][zone]


def mean_variant_multiplier(zone: str) -> float:
    """Mix-weighted multiplier -- used for the theoretical-load ground truth."""
    return sum(v["mix"] * v[zone] for v in VARIANTS.values())


def build_line() -> list[Station]:
    """Derive nominal service times from the balanced target load.

        load = mean_service * mix_multiplier / availability

    so to hit a target load we invert:

        mean_service = target * (1 + offset) * availability / mix_multiplier

    Deriving rather than hard-coding means the line stays balanced when
    availability or variant mix is changed, and makes the balance explicit
    instead of buried in 35 magic numbers.
    """
    stations: list[Station] = []
    for sid, zone, off, cv, buf, mtbf, mttr, tier, every in _LINE_SPEC:
        mtbf, mttr = mtbf * MTBF_SCALE, mttr * MTTR_SCALE
        avail = mtbf / (mtbf + mttr)
        mean_service = TARGET_LOAD_SECONDS * (1.0 + off) * avail / mean_variant_multiplier(zone)

        if tier is SensorTier.BLIND:
            tags: list[TagSpec] = []
        elif zone == "body":
            tags = _body_tags(sid.lower())
        elif zone == "paint":
            tags = _paint_tags(sid.lower())
        else:
            tags = _final_tags(sid.lower())

        if tier is SensorTier.SPARSE:
            tags = tags[:2]          # a handful of tags only

        stations.append(
            Station(
                station_id=sid, zone=zone, mean_service=mean_service, cv=cv,
                buffer_out=buf, mtbf=mtbf, mttr=mttr, tier=tier,
                balance_offset=off, tags=tags,
                sparse_every=max(1, every), manual=(tier is SensorTier.BLIND),
            )
        )
    return stations


# --------------------------------------------------------------------------
# Correlation structure and defect physics for the drift station (S08)
# --------------------------------------------------------------------------

# The fixture holds a panel between a left and a right locating pin. In healthy
# operation both gaps move together (thermal growth, panel batch variation), so
# they are strongly positively correlated. A fixture that twists moves them in
# OPPOSITE directions -- individually small, jointly near-impossible.
S08_CORRELATION = {
    ("s08_gap_left_mm", "s08_gap_right_mm"): 0.86,
    ("s08_clamp_force_N", "s08_weld_current_A"): 0.42,
    ("s08_weld_current_A", "s08_weld_time_ms"): -0.35,
    ("s08_electrode_wear_pct", "s08_weld_current_A"): 0.28,
}

# DEGRADED structure: the fixture develops play. The two locating pins stop
# moving together -- the coupling that held them in agreement is gone. Every
# marginal distribution is UNCHANGED: same mean, same standard deviation, same
# spec margin. Only the joint structure moves.
#
# This is what makes the scenario worth building. A mean shift, however small,
# is eventually visible to a univariate CUSUM. A pure coupling loss is not
# visible to ANY univariate method, at any threshold, for any run length,
# because no single tag's distribution has changed. It is only detectable in
# the relationship between tags.
S08_CORRELATION_DEGRADED = {
    ("s08_gap_left_mm", "s08_gap_right_mm"): -0.10,
    ("s08_clamp_force_N", "s08_weld_current_A"): 0.42,
    ("s08_weld_current_A", "s08_weld_time_ms"): -0.35,
    ("s08_electrode_wear_pct", "s08_weld_current_A"): 0.28,
}

# Defect physics: the panel is mis-set when the fixture is out of square.
# |gap_left - gap_right| is the physical squareness error.
#
# CALIBRATED, NOT GUESSED. scripts/calibrate_defect_physics.py solves for
# (tolerance, steepness) such that the in-control rate lands at ~0.6% (Bosch
# anchor) and the full-drift rate at ~12%: elevated enough to matter, low
# enough that end-of-line inspection does not catch it on the first unit.
GAP_DIFF_TOLERANCE_MM = 0.0820    # squareness error at the logistic midpoint
GAP_DIFF_STEEPNESS = 120.0        # 1/mm
BASE_DEFECT_RATE = 0.005          # background from all other causes
