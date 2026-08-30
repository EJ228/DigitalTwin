"""
Monte Carlo counterfactual rollout.

An alert says something is wrong. A supervisor needs to know what to do about
it, and what it will cost either way. Because the twin IS a simulator, we can
fork the line from its currently observed state, run each candidate intervention
forward several times, and return the distribution of outcomes.

WHAT THE FORK IS SEEDED FROM
----------------------------
Everything is estimated from the event log, never from ground truth:

  buffer levels      counted directly from entry and exit timestamps
  degradation        observed median cycle time over the last 30 minutes,
                     divided by the station's nominal service time
  drift state        whether the quality engine is currently in alarm

WHAT IT DOES NOT MODEL
----------------------
Operator response time, material availability, and the possibility that the
diagnosis is wrong. Rollout horizons are short (30 minutes) precisely because
the estimated degradation is only credible over a short horizon. We report a
spread across replicates rather than a single number, because a counterfactual
quoted to three significant figures is a lie about how much we know.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .injectors import BottleneckSchedule, default_drift
from .line_config import (
    DRIFT_STATION,
    EOL_STATION,
    TAKT_SECONDS,
    build_line,
    mean_variant_multiplier,
)
from .simulator import AssemblyLineSim

HORIZON = 1800.0        # 30 minutes
WARMUP = 900.0          # discarded: the forked line needs time to reach flow
REPLICATES = 6


class FrozenSchedule(BottleneckSchedule):
    """Degradation held at its currently observed level for the whole rollout."""

    def __init__(self, stations, multipliers: dict[str, float]):
        super().__init__([], stations)
        self.mult = multipliers

    def multiplier(self, station_id: str, t: float) -> float:
        return float(self.mult.get(station_id, 1.0))


@dataclass
class Intervention:
    key: str
    label: str
    detail: str

    def apply(self, mult: dict[str, float], drift_on: bool,
              bottleneck: str, stations) -> tuple[dict, bool, float, float]:
        """Return (multipliers, drift_on, downtime_seconds, release_scale)."""
        return dict(mult), drift_on, 0.0, 1.0


class NoAction(Intervention):
    pass


class StopAndFix(Intervention):
    def apply(self, mult, drift_on, bottleneck, stations):
        return dict(mult), False, 600.0, 1.0     # 10 min down, fixture reset


class SlowLine(Intervention):
    """Release work more slowly. A plant reduces speed by releasing less, not
    by making machines faster."""

    def apply(self, mult, drift_on, bottleneck, stations):
        return dict(mult), drift_on, 0.0, 1.08


class Rebalance(Intervention):
    def apply(self, mult, drift_on, bottleneck, stations):
        ids = [s.station_id for s in stations]
        i = ids.index(bottleneck)
        m = dict(mult)
        m[bottleneck] = m.get(bottleneck, 1.0) * 0.92
        for j in (i - 1, i + 1):            # work pushed to the neighbours
            if 0 <= j < len(ids):
                m[ids[j]] = m.get(ids[j], 1.0) * 1.05
        return m, drift_on, 0.0, 1.0


def observed_state(twin, t: float, lookback: float = 1800.0) -> tuple[dict, dict]:
    """Buffer levels and per-station degradation, from the event log alone."""
    buffers = {sid: twin.buffer_level(sid, t) for sid in twin.ids}

    # Degradation must be estimated from UNBLOCKED cycles. Raw occupancy
    # includes time a station stood blocked holding a finished part, so using it
    # would make every station upstream of the constraint look degraded too, and
    # the rollout would start from a line that is slow everywhere.
    mult = {}
    for s in twin.stations:
        occ = twin.recon.occupancy(s.station_id)
        w = occ[(occ.enter > t - lookback) & (occ["exit"] <= t) & (~occ.blocked_end)]
        if len(w) < 5:
            mult[s.station_id] = 1.0
            continue
        obs = float((w["exit"] - w.enter).median())
        # nominal service is quoted for variant A; observed cycles are a mix, so
        # divide the mix multiplier out or every station reads ~5% degraded
        nominal = s.mean_service * mean_variant_multiplier(s.zone)
        mult[s.station_id] = float(np.clip(obs / nominal, 0.85, 1.8))
    return buffers, mult


def rollout(twin, t: float, replicates: int = REPLICATES,
            horizon: float = HORIZON, seed: int = 0) -> dict:
    """Simulate every candidate intervention forward from the observed state."""
    stations = build_line()
    buffers, mult = observed_state(twin, t)
    bn = twin.bottleneck(t)
    bottleneck = bn["station"]
    drift_on = bool(twin.drift(t).get("alarm"))

    candidates = [
        NoAction("no_action", "Do nothing",
                 "Let the line run. Baseline for every other option."),
        SlowLine("slow_line", f"Slow {bottleneck} by 6%",
                 "Reduce line speed to the constraint's real cycle time. Trades "
                 "throughput for stability and fewer speed-induced parameter shifts."),
        Rebalance("rebalance", f"Rebalance work around {bottleneck}",
                  "Move ~8% of work content to the adjacent stations. No downtime, "
                  "but the neighbours absorb the load."),
        StopAndFix("stop_and_fix", f"Stop and reset the {DRIFT_STATION} fixture",
                   "10 minutes of planned downtime now, against continued defect "
                   "production if the drift is real."),
    ]

    ideal = 3600.0 / TAKT_SECONDS
    results = []
    for c in candidates:
        m, keep_drift, downtime, release = c.apply(mult, drift_on, bottleneck, stations)
        tph, ftt, defects = [], [], []
        for r in range(replicates):
            # A drift already in progress is carried into the rollout fully
            # developed: onset well in the past, so the fixture starts the
            # horizon in its degraded state rather than ramping again.
            drift = default_drift(t_onset=t - 100_000.0) if keep_drift else None
            sim = AssemblyLineSim(
                stations=stations, horizon=horizon + WARMUP, seed=seed * 1000 + r,
                schedule=FrozenSchedule(stations, m), drift=drift,
                enable_drift=keep_drift, enable_bottlenecks=False,
                initial_buffers=buffers, start_time=t + downtime,
                release_scale=release,
            )
            f = sim.run()
            # Discard the warm-up. A forked line starts with parts in buffers
            # but no part mid-service at any station, so the first minutes are
            # pipeline fill, not production. Measuring through them would report
            # the same depressed throughput for every option.
            ev = f["events"]
            measure_from = t + downtime + WARMUP
            ev = ev[ev.t >= measure_from]
            done = ev[(ev.station_id == EOL_STATION) & (ev.event_type == "exit")]
            td = f["truth_defects"]
            eff = horizon + downtime
            tph.append(len(done) / (horizon / 3600.0) * (horizon / eff))

            # Defects are counted where they are CREATED, not where they are
            # caught. Nothing built in the next 30 minutes reaches end-of-line
            # inside the horizon -- that lag is the whole problem we are solving,
            # so scoring on completions would report zero defects for every
            # option and make the comparison meaningless.
            made = ev[(ev.station_id == DRIFT_STATION) & (ev.event_type == "exit")]
            d = td[td.part_id.isin(made.part_id)]
            defects.append(int(d.is_defective.sum()))
            ftt.append(1.0 - float(d.is_defective.mean()) if len(d) else 1.0)

        mtph, mftt = float(np.mean(tph)), float(np.mean(ftt))
        results.append({
            "key": c.key, "label": c.label, "detail": c.detail,
            "downtime_min": round(downtime / 60.0, 1),
            "throughput_per_hour": round(mtph, 1),
            "throughput_sd": round(float(np.std(tph)), 2),
            "first_time_through": round(mftt, 4),
            "defects": round(float(np.mean(defects)), 1),
            "defects_per_hour": round(float(np.mean(defects)) / (horizon / 3600.0), 1),
            "good_units_per_hour": round(mtph * mftt, 1),
            "oee_proxy": round(mtph / ideal * mftt, 4),
        })

    base = next(r for r in results if r["key"] == "no_action")
    for r in results:
        r["delta_throughput"] = round(r["throughput_per_hour"] - base["throughput_per_hour"], 1)
        r["delta_ftt_pp"] = round(100 * (r["first_time_through"] - base["first_time_through"]), 2)
        r["delta_defects"] = round(r["defects"] - base["defects"], 1)
        r["delta_oee_pp"] = round(100 * (r["oee_proxy"] - base["oee_proxy"]), 2)

    ranked = sorted(results, key=lambda r: -r["good_units_per_hour"])

    # Break-even for stopping the line. Downtime costs a fixed number of units
    # now; the drift costs defects every hour it runs. Over a 30-minute window
    # doing nothing usually wins, which is exactly why a 30-minute window is the
    # wrong basis for the decision -- so we state when the trade flips instead of
    # letting the short horizon quietly make the call.
    fix = next((r for r in results if r["key"] == "stop_and_fix"), None)
    payback_min = None
    if fix:
        saved = base["defects_per_hour"] - fix["defects_per_hour"]
        lost = fix["downtime_min"] / 60.0 * base["throughput_per_hour"]
        if saved > 0.01:
            payback_min = round(60.0 * lost / saved, 1)

    return {
        "payback_min": payback_min,
        "payback_note": (None if payback_min is None else
                         f"Stopping to fix costs about {fix['downtime_min']:.0f} minutes of "
                         f"output now and pays for itself after roughly "
                         f"{payback_min/60:.1f} hours of continued drift."),
        "t": t, "clock": twin.snapshot(t)["clock"],
        "bottleneck": bottleneck,
        "bottleneck_confident": bn["confident"],
        "drift_alarm": drift_on,
        "horizon_min": round(horizon / 60.0),
        "replicates": replicates,
        "candidates": results,
        "recommended": ranked[0]["key"],
        "caveat": ("Rollouts assume the observed degradation holds for the next "
                   f"{int(horizon/60)} minutes and that the diagnosis is correct. "
                   "Spread across replicates is reported; treat differences "
                   "smaller than the spread as noise."),
    }
