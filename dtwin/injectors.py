"""
Controlled fault injection.

Two injectors, one purpose: create events whose cause and onset time we know
exactly, so that detection latency becomes a measurable quantity rather than
an illustrative figure on a slide.

1. BottleneckSchedule -- degrades station service times on a timetable so the
   momentary bottleneck genuinely migrates WITHIN a shift, not just between
   shifts. This is the condition Roser's active-period method exists for and
   that static utilisation reports miss.

2. FixtureDriftInjector -- ramps the S08 fixture out of square. The left and
   right locating gaps move in OPPOSITE directions. Because they are strongly
   positively correlated in healthy operation, this is a large Mahalanobis
   excursion built entirely out of individually unremarkable readings.

   Design constraint (asserted in tests): under full drift, NO single tag
   reading may breach its 4-sigma spec limit at a rate materially above the
   in-control rate. If a univariate chart could catch it, the scenario would
   not be testing anything.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .line_config import (
    GAP_DIFF_STEEPNESS,
    GAP_DIFF_TOLERANCE_MM,
    Station,
    TagSpec,
    mean_variant_multiplier,
)


# --------------------------------------------------------------------------
# 1. Bottleneck migration
# --------------------------------------------------------------------------

@dataclass
class DegradationEpisode:
    """A station slows down, holds, then recovers.

    severity is the peak multiplier on mean service time.
    1.18 on a station already at 54 s pushes it to ~64 s against a 60 s takt --
    decisively over takt, but not catastrophically so.
    """

    station_id: str
    t_start: float
    t_ramp: float        # seconds to reach peak severity
    t_hold: float        # seconds at peak
    t_recover: float     # seconds to return to nominal
    severity: float
    label: str = ""

    @property
    def t_end(self) -> float:
        return self.t_start + t_total(self)

    def multiplier(self, t: float) -> float:
        if t < self.t_start:
            return 1.0
        dt = t - self.t_start
        if dt < self.t_ramp:
            frac = dt / self.t_ramp
        elif dt < self.t_ramp + self.t_hold:
            frac = 1.0
        elif dt < self.t_ramp + self.t_hold + self.t_recover:
            frac = 1.0 - (dt - self.t_ramp - self.t_hold) / self.t_recover
        else:
            return 1.0
        return 1.0 + (self.severity - 1.0) * frac


def t_total(ep: DegradationEpisode) -> float:
    return ep.t_ramp + ep.t_hold + ep.t_recover


class BottleneckSchedule:
    """Collection of episodes + the theoretical-load ground truth."""

    def __init__(self, episodes: list[DegradationEpisode], stations: list[Station]):
        self.episodes = episodes
        self.stations = {s.station_id: s for s in stations}

    def multiplier(self, station_id: str, t: float) -> float:
        m = 1.0
        for ep in self.episodes:
            if ep.station_id == station_id:
                m *= ep.multiplier(t)
        return m

    def theoretical_load(self, station_id: str, t: float) -> float:
        """Expected seconds of station time consumed per part.

        load = nominal service * mix-weighted variant factor
               * degradation multiplier / availability

        The station with the highest load is the true bottleneck: it is the
        station whose capacity binds. This is computed from configuration,
        never from the event log, so it is genuinely independent of anything
        the detector sees.
        """
        s = self.stations[station_id]
        return (
            s.mean_service
            * mean_variant_multiplier(s.zone)
            * self.multiplier(station_id, t)
            / s.availability
        )

    def true_bottleneck(self, t: float) -> tuple[str, float, str, float]:
        loads = sorted(
            ((self.theoretical_load(sid, t), sid) for sid in self.stations),
            reverse=True,
        )
        (l1, s1), (l2, s2) = loads[0], loads[1]
        return s1, l1, s2, l1 - l2


def default_schedule(stations: list[Station], shift_seconds: float) -> BottleneckSchedule:
    """Episodes spread across the whole horizon, at mixed severity.

    Timings are fractions of the horizon so the schedule scales with run
    length. Severities deliberately range from decisive (1.42) to milder
    (1.24 on a station near the bottom of the balance, which only just becomes
    the constraint). The marginal cases are the ones
    a utilisation report gets wrong, so they are the ones worth scoring on.

    Two pairs overlap in time -- the case where "which station is the
    bottleneck right now" has a genuinely non-obvious answer.
    """
    H = shift_seconds
    specs = [
        # station, start_frac, ramp_frac, hold_frac, recover_frac, severity, label
        ("S29", 0.055, 0.012, 0.045, 0.018, 1.38, "operator fatigue / tool wear"),
        ("S16", 0.150, 0.020, 0.055, 0.025, 1.30, "paint oven fouling"),
        ("S06", 0.255, 0.015, 0.040, 0.015, 1.26, "weld gun degradation"),
        ("S23", 0.285, 0.012, 0.038, 0.012, 1.33, "manual trim variability"),   # overlaps S06
        ("S35", 0.390, 0.018, 0.050, 0.020, 1.42, "EOL inspection backlog"),
        ("S12", 0.500, 0.014, 0.035, 0.014, 1.24, "marginal: zone exit bank"),
        ("S27", 0.620, 0.016, 0.048, 0.018, 1.36, "fastener feed jam"),
        ("S03", 0.660, 0.010, 0.030, 0.012, 1.25, "marginal: sparse station"),  # overlaps S27
        ("S19", 0.760, 0.018, 0.052, 0.020, 1.34, "robot recalibration drift"),
        ("S26", 0.870, 0.014, 0.042, 0.016, 1.31, "manual harness rework"),
    ]
    eps = [
        DegradationEpisode(
            station_id=sid, t_start=sf * H, t_ramp=rf * H, t_hold=hf * H,
            t_recover=cf * H, severity=sev, label=label,
        )
        for sid, sf, rf, hf, cf, sev, label in specs
    ]
    return BottleneckSchedule(eps, stations)


# --------------------------------------------------------------------------
# 2. Multivariate tag generation and fixture drift
# --------------------------------------------------------------------------

class TagGenerator:
    """Correlated multivariate-normal tag readings for one station."""

    def __init__(
        self,
        tags: list[TagSpec],
        correlations: dict[tuple[str, str], float] | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.tags = tags
        self.names = [t.name for t in tags]
        self.rng = rng or np.random.default_rng(0)
        self.mu0 = np.array([t.mu0 for t in tags])
        self.sigma = np.array([t.sigma for t in tags])
        self.corr = self._build_corr(correlations or {})
        self.chol = np.linalg.cholesky(self.corr)

    def _build_corr(self, correlations: dict[tuple[str, str], float]) -> np.ndarray:
        n = len(self.names)
        c = np.eye(n)
        idx = {nm: i for i, nm in enumerate(self.names)}
        for (a, b), rho in correlations.items():
            if a in idx and b in idx:
                c[idx[a], idx[b]] = rho
                c[idx[b], idx[a]] = rho
        # nearest-PD repair in case the supplied correlations are inconsistent
        w, v = np.linalg.eigh(c)
        if w.min() < 1e-8:
            w = np.clip(w, 1e-8, None)
            c = v @ np.diag(w) @ v.T
            d = np.sqrt(np.diag(c))
            c = c / np.outer(d, d)
        return c

    def set_degraded(self, correlations: dict[tuple[str, str], float]):
        """Register the degraded joint structure to interpolate toward."""
        self.corr_deg = self._build_corr(correlations)
        self._chol_cache: dict[int, np.ndarray] = {}

    def _chol_at(self, frac: float) -> np.ndarray:
        """Cholesky of the correlation matrix interpolated by frac.

        Cached on a 1% grid: the fixture degrades over half an hour, so
        resolution finer than 1% of the ramp is meaningless and recomputing a
        6x6 decomposition per part is wasted work.
        """
        if frac <= 0.0 or not hasattr(self, "corr_deg"):
            return self.chol
        key = int(round(frac * 100))
        if key not in self._chol_cache:
            c = (1 - key / 100) * self.corr + (key / 100) * self.corr_deg
            w, v = np.linalg.eigh(c)
            if w.min() < 1e-8:
                c = v @ np.diag(np.clip(w, 1e-8, None)) @ v.T
                d = np.sqrt(np.diag(c))
                c = c / np.outer(d, d)
            self._chol_cache[key] = np.linalg.cholesky(c)
        return self._chol_cache[key]

    def sample(self, shift_sigmas: np.ndarray | None = None,
               degrade_frac: float = 0.0,
               latent: np.ndarray | None = None,
               own_scale: np.ndarray | None = None) -> dict[str, float]:
        """Draw one part's readings.

        latent carries the shared causes (incoming batch, ambient conditions)
        in sigma units; own_scale = sqrt(1 - w^2) shrinks the idiosyncratic part
        so total variance is preserved and marginal distributions are unchanged.
        """
        z = self.rng.standard_normal(len(self.names))
        corr_z = self._chol_at(degrade_frac) @ z
        if own_scale is not None:
            corr_z = corr_z * own_scale
        if latent is not None:
            corr_z = corr_z + latent
        shift = shift_sigmas if shift_sigmas is not None else 0.0
        vals = self.mu0 + self.sigma * (corr_z + shift)
        return dict(zip(self.names, vals))


@dataclass
class FixtureDriftInjector:
    """S08 fixture rotates slowly out of square.

    shift_sigmas is expressed in process-sigma units so the scenario is
    scale-free. Peak magnitudes are chosen to sit well inside a 4-sigma spec
    limit: the largest is 0.6 sigma, leaving 3.4 sigma of headroom, so the
    per-reading probability of a univariate spec breach stays near its
    in-control level. The magnitude is also tuned so a SINGLE-POINT
    Hotelling T^2 chart alarms only ~8% of the time under full drift:
    instantaneous multivariate testing is not enough either, and genuine
    accumulation (CUSUM/EWMA) is required. That is the claim under test.
    """

    station_id: str
    t_onset: float
    t_full: float                    # ramp completion
    tag_names: list[str]
    peak_shift: dict[str, float]     # tag name -> shift in sigma units
    mechanism: str = "coupling_loss"

    def shift_vector(self, t: float) -> np.ndarray:
        if t <= self.t_onset:
            frac = 0.0
        elif t >= self.t_full:
            frac = 1.0
        else:
            frac = (t - self.t_onset) / (self.t_full - self.t_onset)
        return np.array([frac * self.peak_shift.get(nm, 0.0) for nm in self.tag_names])

    def fraction(self, t: float) -> float:
        if t <= self.t_onset:
            return 0.0
        if t >= self.t_full:
            return 1.0
        return (t - self.t_onset) / (self.t_full - self.t_onset)


def default_drift(t_onset: float, ramp_seconds: float = 1800.0) -> FixtureDriftInjector:
    names = [
        "s08_clamp_force_N",
        "s08_gap_left_mm",
        "s08_gap_right_mm",
        "s08_weld_current_A",
        "s08_weld_time_ms",
        "s08_electrode_wear_pct",
    ]
    return FixtureDriftInjector(
        station_id="S08",
        t_onset=t_onset,
        t_full=t_onset + ramp_seconds,
        tag_names=names,
        # NO mean shift. The mechanism is pure loss of coupling between the
        # two locating pins, so every marginal distribution is untouched.
        peak_shift={},
        mechanism="coupling_loss",
    )


# --------------------------------------------------------------------------
# Defect physics
# --------------------------------------------------------------------------

def defect_probability_from_gap(gap_left: float, gap_right: float, base: float) -> float:
    """Panel mis-set probability as a function of fixture squareness error.

    Both gaps can sit comfortably inside spec while their DIFFERENCE is out of
    tolerance. This is the entire point of the scenario: the defect lives in a
    relationship between tags, not in any tag.
    """
    err = abs(gap_left - gap_right)
    x = GAP_DIFF_STEEPNESS * (err - GAP_DIFF_TOLERANCE_MM)
    p_mechanism = 1.0 / (1.0 + np.exp(-x))
    return base + (1.0 - base) * p_mechanism
