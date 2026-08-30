"""
Reconstruct line state from the event log alone.

This module is the load-bearing part of the Round 1 claim that the twin works
"from the one signal every station emits -- the timestamp of a part entering
and leaving." Everything here is derived from ENTER/EXIT timestamps plus the
buffer capacities, which are engineering design data, not sensor data and not
ground truth.

Nothing in this file reads a truth table.

Derivation
----------
Buffer level. The buffer between station k and k+1 holds exactly those parts
that have exited k but not yet entered k+1. That is a pure event-log query, so
buffer occupancy is EXACTLY recoverable even at stations with no sensors at all.

Occupancy. A station holds a part over [enter, exit]. Between the exit of one
part and the enter of the next, it holds nothing.

Blocked vs working. A station finishes a part and then cannot release it while
its downstream buffer is full; the part leaves at the instant room appears.
So the blocked interval is the maximal suffix of the occupancy during which the
downstream buffer sat at capacity. Everything else in the occupancy is the
station being constrained by itself -- service, setup or repair.

  ACTIVE   = working + down     (constrained by itself)
  INACTIVE = blocked + starved  (constrained by another station)

That split is precisely what Roser's active-period method needs, and it is the
reason the bottleneck can be found without a single process sensor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .line_config import Station


def buffer_timeline(
    events: pd.DataFrame, upstream: str, downstream: str
) -> pd.DataFrame:
    """Signed level changes for the buffer between two stations.

    Returns a frame of (t, level) step points.
    """
    outs = events[(events.station_id == upstream) & (events.event_type == "exit")][["t"]]
    ins = events[(events.station_id == downstream) & (events.event_type == "enter")][["t"]]
    steps = pd.concat(
        [outs.assign(d=1), ins.assign(d=-1)], ignore_index=True
    ).sort_values(["t", "d"], ascending=[True, False], kind="stable")
    steps["level"] = steps.d.cumsum()
    # Collapse to one row per distinct timestamp, keeping the level AFTER all
    # simultaneous moves. Without this, a release (get then put at the same
    # instant) shows a one-step dip below capacity that falsely breaks the
    # "buffer was full" run.
    steps = steps.drop_duplicates(subset="t", keep="last")
    return steps[["t", "level"]].reset_index(drop=True)


class LineReconstruction:
    """Event log -> per-station state intervals, without touching sensors."""

    def __init__(self, events: pd.DataFrame, stations: list[Station]):
        self.stations = stations
        self.ids = [s.station_id for s in stations]
        self.cap = {s.station_id: s.buffer_out for s in stations}
        ev = events[events.event_type.isin(["enter", "exit"])]
        self.ev = ev.sort_values("t", kind="stable").reset_index(drop=True)
        self._occ: dict[str, pd.DataFrame] = {}
        self._buf: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------

    def occupancy(self, sid: str) -> pd.DataFrame:
        """One row per part: (part_id, enter, exit)."""
        if sid in self._occ:
            return self._occ[sid]
        e = self.ev[self.ev.station_id == sid]
        enters = e[e.event_type == "enter"][["part_id", "t", "variant"]].rename(
            columns={"t": "enter"})
        exits = e[e.event_type == "exit"][["part_id", "t"]].rename(columns={"t": "exit"})
        occ = enters.merge(exits, on="part_id", how="inner").sort_values("enter")

        # did this cycle end against a full downstream buffer?
        cap = self.cap[sid]
        buf = self.buffer(sid)
        if cap > 0 and len(buf) > 1:
            bt, bl = buf.t.values, buf.level.values
            j = np.searchsorted(bt, occ["exit"].values, side="left") - 1
            before = np.where(j >= 0, bl[np.clip(j, 0, None)], 0)
            occ["blocked_end"] = before >= cap
        else:
            occ["blocked_end"] = False

        self._occ[sid] = occ.reset_index(drop=True)
        return self._occ[sid]

    def buffer(self, sid: str) -> pd.DataFrame:
        """Downstream buffer level timeline for station sid."""
        if sid in self._buf:
            return self._buf[sid]
        i = self.ids.index(sid)
        if i == len(self.ids) - 1:
            tl = pd.DataFrame({"t": [0.0], "level": [0]})
        else:
            tl = buffer_timeline(self.ev, sid, self.ids[i + 1])
        self._buf[sid] = tl
        return tl

    # ------------------------------------------------------------------

    def _service_estimate(self, sid: str) -> dict:
        """Median unblocked cycle time per variant -- our estimate of service.

        An occupancy that did NOT end against a full buffer is pure station
        time: service, plus setup or repair if either occurred. Taking the
        median over those makes the estimate robust to repairs. We stratify by
        variant because a mixed-model line has genuinely different work content
        per variant, and the log tells us which variant each part was.
        """
        occ = self.occupancy(sid)
        if occ.empty:
            return {}
        free = occ[~occ.blocked_end]
        if free.empty:
            free = occ
        med = free.groupby("variant").apply(
            lambda g: float(np.median(g["exit"] - g.enter)), include_groups=False
        ).to_dict()
        med["_default"] = float(np.median(free["exit"] - free.enter))
        return med

    def state_intervals(self, sid: str) -> pd.DataFrame:
        """(state, t_start, t_end) per station, inferred from timestamps only.

        states: active | blocked | starved

        Blocking is identified in two steps.

        1. WHETHER a cycle ended blocked. The part leaves at the instant room
           appears downstream, so a cycle ended in blocking exactly when the
           downstream buffer stood at capacity immediately before the exit.
           That test is a pure event-log query.

        2. HOW MUCH of the cycle was blocked. We do not know when service
           actually finished, so we estimate it as the median UNBLOCKED cycle
           time for that station and variant, and attribute the remainder of
           the occupancy to blocking. Using unblocked cycles as the reference
           is what keeps this from being circular.
        """
        occ = self.occupancy(sid)
        if occ.empty:
            return pd.DataFrame(columns=["station_id", "state", "t_start", "t_end"])

        svc = self._service_estimate(sid)
        rows = []
        prev_exit = None
        for part_id, t_in, t_out, variant, blocked_end in occ[
            ["part_id", "enter", "exit", "variant", "blocked_end"]
        ].itertuples(index=False):
            if prev_exit is not None and t_in > prev_exit + 1e-9:
                rows.append((sid, "starved", prev_exit, t_in))

            span = t_out - t_in
            if blocked_end:
                est = svc.get(variant, svc.get("_default", span))
                active_dur = float(min(span, max(est, 0.0)))
            else:
                active_dur = span

            split = t_in + active_dur
            if active_dur > 1e-9:
                rows.append((sid, "active", t_in, split))
            if t_out > split + 1e-9:
                rows.append((sid, "blocked", split, t_out))
            prev_exit = t_out

        return pd.DataFrame(rows, columns=["station_id", "state", "t_start", "t_end"])

    def all_states(self) -> pd.DataFrame:
        return pd.concat([self.state_intervals(s) for s in self.ids], ignore_index=True)

    # ------------------------------------------------------------------

    def active_periods(self, sid: str) -> pd.DataFrame:
        """Merge adjacent active intervals into maximal active PERIODS.

        Roser's definition: consecutive active intervals separated by no
        inactive time belong to the same active period. In practice two parts
        processed back to back with no starvation between them form one long
        period -- which is exactly the signature of a station that is never
        waiting for anyone else.
        """
        st = self.state_intervals(sid)
        act = st[st.state == "active"].sort_values("t_start")
        if act.empty:
            return pd.DataFrame(columns=["station_id", "t_start", "t_end", "duration"])
        gaps = act.t_start.values[1:] > act.t_end.values[:-1] + 1e-6
        grp = np.concatenate(([0], np.cumsum(gaps)))
        act = act.assign(_g=grp)
        per = act.groupby("_g").agg(t_start=("t_start", "min"), t_end=("t_end", "max"))
        per["duration"] = per.t_end - per.t_start
        per["station_id"] = sid
        return per.reset_index(drop=True)[["station_id", "t_start", "t_end", "duration"]]

    def all_active_periods(self) -> pd.DataFrame:
        return pd.concat([self.active_periods(s) for s in self.ids], ignore_index=True)
