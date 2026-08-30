"""
The live twin: everything the UI needs, precomputed once and served by time.

A run is a fixed event log, so rather than recompute detectors on every frame we
evaluate them once on a 30-second grid and then serve snapshots by timestamp.
Replay is therefore an index lookup, which is what makes a 1 Hz websocket cheap.

Everything here reads the event log. Ground truth is used only to label the
disruption log and the escape-window comparison, both of which are explanatory
overlays for the demo, never inputs to a prediction.
"""

from __future__ import annotations

import functools
import pathlib

import numpy as np
import pandas as pd

from .detectors import BottleneckWalkDetector, UtilisationDetector
from .line_config import DRIFT_STATION, EOL_STATION, TAKT_SECONDS, build_line
from .reconstruct import LineReconstruction
from .blind import VirtualMetrology
from .conformal import ConformalThreshold
from .forecast import LSTMForecaster, make_windows
from .spc import Reference, T2CUSUM

GRID = 30.0
PHASE1_N = 400
DEFAULT_T2CUSUM_THRESHOLD = 9.32     # calibrated: 1 false alarm / 1200 parts

# Below this score margin the top two stations are tied and the line has slack.
# Chosen so quiet periods read as "no clear constraint" rather than naming a
# station at random.
MARGIN_CONFIDENT = 0.08

# Drift-statistic projection. Trained on sibling runs so the forecaster has seen
# a CUSUM actually rise; a model trained only on in-control data would predict a
# flat line for ever, which is honest but useless.
PROJ_LOOKBACK = 12
PROJ_HORIZON = 12
PROJ_ALPHA = 0.10          # 90% conformal band


class LiveTwin:
    """One loaded run, replayable and queryable by simulation time."""

    def __init__(
        self,
        run_dir: str | pathlib.Path,
        blind_stations: set[str] | None = None,
        t2_threshold: float = DEFAULT_T2CUSUM_THRESHOLD,
    ):
        self.run_dir = pathlib.Path(run_dir)
        self.blind = set(blind_stations or ())
        self.t2_threshold = t2_threshold
        self.stations = build_line()
        self.ids = [s.station_id for s in self.stations]

        self.events = pd.read_parquet(self.run_dir / "events.parquet")
        self.tags = pd.read_parquet(self.run_dir / "tags.parquet")
        self.t0 = float(self.events.t.min())
        self.t1 = float(self.events.t.max())

        self.recon = LineReconstruction(self.events, self.stations)
        self.grid = np.arange(self.t0 + 600.0, self.t1, GRID)

        self._build_states()
        self._build_bottleneck()
        self._build_drift()
        self._build_projection()
        self._build_metrics()

    # ------------------------------------------------------------------
    # station state timeline
    # ------------------------------------------------------------------

    def _build_states(self):
        self.state_by_station: dict[str, tuple] = {}
        for sid in self.ids:
            st = self.recon.state_intervals(sid).sort_values("t_start")
            if st.empty:
                self.state_by_station[sid] = (np.array([0.0]), np.array(["idle"]))
                continue
            self.state_by_station[sid] = (st.t_start.values, st.state.values)

        self.buffer_by_station: dict[str, tuple] = {}
        for i, sid in enumerate(self.ids[:-1]):
            b = self.recon.buffer(sid)
            self.buffer_by_station[sid] = (b.t.values, b.level.values)

    def station_state(self, sid: str, t: float) -> str:
        ts, vs = self.state_by_station[sid]
        j = int(np.searchsorted(ts, t, side="right") - 1)
        return str(vs[j]) if j >= 0 else "idle"

    def buffer_level(self, sid: str, t: float) -> int:
        d = self.buffer_by_station.get(sid)
        if d is None:
            return 0
        ts, vs = d
        j = int(np.searchsorted(ts, t, side="right") - 1)
        return int(vs[j]) if j >= 0 else 0

    # ------------------------------------------------------------------
    # flow engine
    # ------------------------------------------------------------------

    def _build_bottleneck(self):
        walk = BottleneckWalkDetector(self.recon)
        util = UtilisationDetector(self.recon, window=1800.0)
        self.pred_walk = walk.predict(self.grid)
        self.pred_util = util.predict(self.grid)

    def bottleneck(self, t: float) -> dict:
        """Current constraint, or an explicit admission that there isn't one.

        Below MARGIN_CONFIDENT the top two stations are effectively tied and the
        line has slack. Reporting a winner there would be noise dressed as
        insight, so we return confident=False and the UI shows "no clear
        constraint" rather than pointing at a station.
        """
        j = int(np.clip(np.searchsorted(self.grid, t, side="right") - 1, 0, len(self.grid) - 1))
        margin = float(self.pred_walk.margin.iloc[j])
        return {
            "station": str(self.pred_walk.predicted.iloc[j]),
            "runner_up": str(self.pred_walk.runner_up.iloc[j]),
            "margin": round(margin, 4),
            "confident": bool(margin >= MARGIN_CONFIDENT),
            "utilisation_report_says": str(self.pred_util.predicted.iloc[j]),
        }

    # ------------------------------------------------------------------
    # quality engine
    # ------------------------------------------------------------------

    def _build_drift(self):
        """T2-CUSUM on the drift station, unless it has been blinded."""
        s08 = next(s for s in self.stations if s.station_id == DRIFT_STATION)
        names = [t.name for t in s08.tags]
        sub = self.tags[self.tags.station_id == DRIFT_STATION]

        if DRIFT_STATION in self.blind or sub.empty:
            self.drift_t = np.array([])
            self.drift_stat = np.array([])
            self.drift_alarm_t = None
            self.drift_available = False
            return

        wide = sub.pivot_table(index="part_id", columns="tag", values="value",
                               aggfunc="first").sort_index()
        tmap = sub.groupby("part_id").t.first().reindex(wide.index)
        X = wide[names].values
        ref = Reference.fit(X[:PHASE1_N], names)
        det = T2CUSUM(ref, threshold=self.t2_threshold)

        self.drift_t = tmap.values
        self.drift_stat = det.statistic(X)
        idx = np.where(self.drift_stat[PHASE1_N:] > self.t2_threshold)[0]
        self.drift_alarm_t = float(self.drift_t[idx[0] + PHASE1_N]) if len(idx) else None
        self.drift_available = True

    def _build_projection(self):
        """LSTM forecast of the drift statistic with a conformal band.

        Replaces a band that used to be drawn in the frontend from two hardcoded
        constants. The forecast comes from the same LSTM used for buffer
        trajectories, and the interval is a split-conformal residual quantile --
        so its width is measured, not chosen to look reassuring.
        """
        self.proj_t, self.proj_mu, self.proj_lo, self.proj_hi = (
            np.array([]), np.array([]), np.array([]), np.array([]))
        if not self.drift_available or len(self.drift_stat) < 200:
            return

        sib = [p for p in sorted(self.run_dir.parent.glob("run_s*"))
               if p != self.run_dir and (p / "tags.parquet").exists()]
        if not sib:
            return

        def stat_of(run: pathlib.Path):
            tw = LiveTwin.__new__(LiveTwin)
            tw.run_dir, tw.blind, tw.t2_threshold = run, set(), self.t2_threshold
            tw.stations, tw.ids = self.stations, self.ids
            tw.tags = pd.read_parquet(run / "tags.parquet")
            tw._build_drift()
            return tw.drift_stat

        series = [stat_of(r) for r in sib[:3]]
        series = [s_ for s_ in series if len(s_) > PROJ_LOOKBACK + PROJ_HORIZON + 50]
        if not series:
            return

        sc = max(1.0, float(np.percentile(np.concatenate(series), 99)))
        Xs, Ys = zip(*[make_windows((s_ / sc)[:, None], PROJ_LOOKBACK, PROJ_HORIZON)
                       for s_ in series])
        X, Y = np.concatenate(Xs), np.concatenate(Ys)
        cut = int(len(X) * 0.8)
        m = LSTMForecaster(1, 24, 1, seed=0).fit(X[:cut], Y[:cut], epochs=12, batch=128)

        # conformal residual band from the held-out slice of the SAME runs
        resid = np.abs(m.predict(X[cut:]).ravel() - Y[cut:].ravel())
        width = float(ConformalThreshold(PROJ_ALPHA).fit(resid).threshold) * sc

        Xe, _ = make_windows((self.drift_stat / sc)[:, None], PROJ_LOOKBACK, PROJ_HORIZON)
        if len(Xe) == 0:
            return
        mu = m.predict(Xe).ravel() * sc
        idx = np.arange(len(Xe)) + PROJ_LOOKBACK + PROJ_HORIZON - 1
        self.proj_t = self.drift_t[np.clip(idx, 0, len(self.drift_t) - 1)]
        self.proj_mu = mu
        self.proj_lo = np.maximum(mu - width, 0.0)
        self.proj_hi = mu + width
        self.proj_width = width

    def blind_confidence(self) -> dict:
        """Real GP posterior width for the drift station when it is blinded.

        Previously this was a constant of 0.12 sitting in this file. The
        Gaussian process that should produce it existed but was never connected.
        """
        if DRIFT_STATION not in self.blind:
            return {"blinded": False}
        vm = VirtualMetrology()
        M = vm.cycle_matrix(self.events, self.ids)
        r, _, _ = vm.infer(M, DRIFT_STATION, self.ids)
        sd = None if not np.isfinite(r.posterior_sd) else round(float(r.posterior_sd), 3)
        naive = None if not np.isfinite(r.naive_rmse) else round(float(r.naive_rmse), 3)
        return {
            "blinded": True,
            "posterior_sd_seconds": sd,
            "naive_sd_seconds": naive,
            "inference_skill": None if not np.isfinite(r.skill) else round(float(r.skill), 3),
            "coverage_95": None if not np.isfinite(r.coverage_95) else round(float(r.coverage_95), 3),
            "neighbours": r.neighbours,
            "source": "Gaussian-process posterior over neighbouring stations",
        }

    def drift(self, t: float, history: float = 3600.0) -> dict:
        """Chart series plus alarm state, as of time t."""
        if not self.drift_available:
            return {
                "available": False,
                "reason": f"{DRIFT_STATION} has no process sensors in this configuration",
                "series": [], "projection": [], "threshold": self.t2_threshold,
                "value": None, "alarm": False,
                "blind_confidence": self.blind_confidence(),
            }
        m = (self.drift_t <= t) & (self.drift_t >= t - history)
        alarm = self.drift_alarm_t is not None and t >= self.drift_alarm_t
        val = float(self.drift_stat[self.drift_t <= t][-1]) if (self.drift_t <= t).any() else 0.0
        proj = []
        if len(self.proj_t):
            pm = (self.proj_t > t) & (self.proj_t <= t + PROJ_HORIZON * 60.0)
            proj = [{"t": float(a), "mu": float(b), "lo": float(c), "hi": float(d)}
                    for a, b, c, d in zip(self.proj_t[pm], self.proj_mu[pm],
                                          self.proj_lo[pm], self.proj_hi[pm])]
        return {
            "available": True,
            "series": [{"t": float(a), "v": float(b)}
                       for a, b in zip(self.drift_t[m], self.drift_stat[m])],
            "projection": proj,
            "projection_band": {
                "level": 1 - PROJ_ALPHA,
                "half_width": round(getattr(self, "proj_width", float("nan")), 3),
                "source": "LSTM forecast, split-conformal residual interval",
            },
            "threshold": self.t2_threshold,
            "value": val,
            "exceedance": round(val / self.t2_threshold, 3) if self.t2_threshold else None,
            "alarm": bool(alarm),
            "alarm_t": self.drift_alarm_t,
            "false_alarm_budget_parts": 1200,
            "blind_confidence": {"blinded": False},
        }

    # ------------------------------------------------------------------
    # counters
    # ------------------------------------------------------------------

    def _build_metrics(self):
        ex = self.events[(self.events.station_id == EOL_STATION)
                         & (self.events.event_type == "exit")]
        self.completion_t = ex.t.values
        insp = self.events[self.events.event_type == "rework_in"]
        self.rework_t = insp.t.values

    def metrics(self, t: float, window: float = 3600.0) -> dict:
        lo = max(self.t0, t - window)
        span = max(t - lo, 1.0)
        built = int(((self.completion_t > lo) & (self.completion_t <= t)).sum())
        rework = int(((self.rework_t > lo) & (self.rework_t <= t)).sum())
        tph = built / (span / 3600.0)
        ftt = 1.0 - (rework / built) if built else 1.0

        # Duration-weighted, not event-counted. Availability here is flow
        # efficiency: the share of station-time not lost to being blocked by a
        # downstream station or starved by an upstream one.
        idle = 0.0
        for sid in self.ids:
            st = self.recon.state_intervals(sid)
            m = (st.t_end > lo) & (st.t_start < t) & st.state.isin(["blocked", "starved"])
            if m.any():
                seg = st[m]
                idle += float((np.minimum(seg.t_end, t) - np.maximum(seg.t_start, lo)).sum())
        availability = float(np.clip(1.0 - idle / (len(self.ids) * span), 0.0, 1.0))

        perf = float(np.clip(tph / (3600.0 / TAKT_SECONDS), 0.0, 1.0))
        return {
            "throughput_per_hour": round(tph, 1),
            "takt_target_per_hour": round(3600.0 / TAKT_SECONDS, 1),
            "built_total": int((self.completion_t <= t).sum()),
            "first_time_through": round(float(np.clip(ftt, 0, 1)), 4),
            "oee": round(availability * perf * float(np.clip(ftt, 0, 1)), 4),
            "availability": round(availability, 4),
            "performance": round(perf, 4),
            "quality": round(float(np.clip(ftt, 0, 1)), 4),
        }

    # ------------------------------------------------------------------
    # the snapshot the websocket sends
    # ------------------------------------------------------------------

    def snapshot(self, t: float) -> dict:
        bn = self.bottleneck(t)
        stations = []
        for i, s in enumerate(self.stations):
            sid = s.station_id
            is_blind = sid in self.blind or s.tier.value == "blind"
            stations.append({
                "id": sid,
                "zone": s.zone,
                "index": i,
                "state": self.station_state(sid, t),
                "buffer": self.buffer_level(sid, t),
                "buffer_capacity": s.buffer_out,
                "tier": "blind" if sid in self.blind else s.tier.value,
                "blind": bool(is_blind),
                "is_bottleneck": sid == bn["station"],
                "manual": s.manual,
            })
        return {
            "t": float(t),
            "clock": _clock(t),
            "shift": int(t // (8 * 3600)) + 1,
            "stations": stations,
            "bottleneck": bn,
            "drift": self.drift(t),
            "metrics": self.metrics(t),
        }

    # ------------------------------------------------------------------
    # manager and leadership views
    # ------------------------------------------------------------------

    def shift_timeline(self) -> list[dict]:
        """Contiguous runs of the predicted bottleneck -- the migration band."""
        p = self.pred_walk
        change = (p.predicted != p.predicted.shift()).cumsum()
        out = []
        for _, g in p.groupby(change):
            out.append({
                "station": str(g.predicted.iloc[0]),
                "t_start": float(g.t.iloc[0]),
                "t_end": float(g.t.iloc[-1]),
                "duration_min": round((float(g.t.iloc[-1]) - float(g.t.iloc[0])) / 60.0, 1),
            })
        return [o for o in out if o["duration_min"] >= 1.0]

    def disruption_log(self) -> list[dict]:
        """Injected disruptions with how long the twin took to name each one.

        Ground truth used as an explanatory overlay for the demo. The detector
        never sees it.
        """
        path = self.run_dir / "truth_episodes.parquet"
        if not path.exists():
            return []
        from .scoring import episode_scores
        ep = pd.read_parquet(path)
        es = episode_scores(self.pred_walk, ep)
        out = []
        for e, r in zip(ep.itertuples(index=False), es.itertuples(index=False)):
            out.append({
                "station": e.station_id,
                "cause": e.label,
                "severity": round(float(e.severity), 2),
                "t_start": float(e.t_start),
                "clock": _clock(float(e.t_start)),
                "detected": bool(r.detected),
                "detect_lag_min": None if not np.isfinite(r.detect_lag_s)
                                  else round(float(r.detect_lag_s) / 60.0, 1),
                "hold_hit_rate": round(float(r.hold_hit_rate), 2),
            })
        return out

    def escape_window(self) -> dict:
        """The split-screen comparison: cars built before anyone knew."""
        dr = self.run_dir / "truth_drift.parquet"
        td = self.run_dir / "truth_defects.parquet"
        if not dr.exists() or not td.exists():
            return {"available": False}
        onset = float(pd.read_parquet(dr).t_onset.iloc[0])
        ex = (self.events[(self.events.station_id == DRIFT_STATION)
                          & (self.events.event_type == "exit")]
              .sort_values("t").reset_index(drop=True))
        pos = {p: i for i, p in enumerate(ex.part_id)}
        onset_pos = int(ex[ex.t >= onset].index[0])

        defects = pd.read_parquet(td)
        caused = defects[(defects.cause_mechanism == "coupling_loss") & defects.is_defective]
        caused = caused[caused.part_id.map(pos).fillna(-1) >= onset_pos]
        bad = {int(pos[int(p)]) - onset_pos for p in caused.part_id if int(p) in pos}

        caught = caused[caused.detected_at.notna()]
        eol = None
        if len(caught):
            t_det = float(caught.sort_values("detected_at").iloc[0].detected_at)
            eol = int(np.searchsorted(ex.t.values, t_det, "right")) - onset_pos

        twin = None
        if self.drift_alarm_t is not None:
            twin = int(np.searchsorted(ex.t.values, self.drift_alarm_t, "right")) - onset_pos

        n = max(filter(None, [eol, twin, 0])) + 6
        return {
            "available": True,
            "onset_t": onset,
            "baseline_flag_at": eol,
            "twin_flag_at": twin,
            "reduction_pct": None if not (eol and twin) else round(100 * (1 - twin / eol)),
            "cars": [{"i": i, "defective": i in bad} for i in range(n)],
        }

    def coverage(self) -> dict:
        tiers: dict[str, int] = {}
        for s in self.stations:
            k = "blind" if s.station_id in self.blind else s.tier.value
            tiers[k] = tiers.get(k, 0) + 1
        n = len(self.stations)
        return {
            "counts": tiers,
            "total": n,
            "under_instrumented_pct": round(
                100 * (tiers.get("sparse", 0) + tiers.get("blind", 0)) / n, 1),
        }


def _clock(t: float) -> str:
    """Simulation seconds as a shift clock, starting 06:00."""
    total = int(t) % 86400
    h = (6 + total // 3600) % 24
    return f"{h:02d}:{(total % 3600) // 60:02d}"


@functools.lru_cache(maxsize=8)
def load_twin(run_dir: str, blind: tuple[str, ...] = ()) -> LiveTwin:
    return LiveTwin(run_dir, blind_stations=set(blind))
