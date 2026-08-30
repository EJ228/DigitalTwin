"""
Momentary bottleneck detection, and the two baselines we claim to beat.

All three consume the event log only.

1. ActivePeriodDetector  (Roser, Nakano & Tanaka 2001/2002)
   The bottleneck is the station that has been active -- constrained by itself
   rather than blocked or starved -- for the longest uninterrupted stretch.

2. UtilisationDetector   -- the "static utilisation report" strawman
   Highest occupied fraction over a trailing window. This is what most plants
   actually use, and the Round 1 deck's claim is that it names the bottleneck
   after it has already moved. That claim is only worth making if we measure it.

3. QueueLengthDetector   -- "longest queue wins"
   Largest upstream buffer level. The deck states outright that the longest
   queue is not the bottleneck; this detector exists so that statement becomes
   a measured result rather than an assertion.

ONLINE HONESTY
--------------
The active-period detector defaults to online=True, meaning at query time t it
may only use active time ELAPSED so far, never the eventual length of a period
that has not finished. The offline variant (online=False) is the canonical form
in the literature and scores higher, but it peeks past t. We report the online
number as the headline because that is the one a plant could actually run.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .reconstruct import LineReconstruction


class BottleneckDetector(ABC):
    name: str

    @abstractmethod
    def predict(self, times: np.ndarray) -> pd.DataFrame:
        """Return (t, predicted, score) for each query time."""


def _rank(times, scores, ids, smooth):
    """Winner, runner-up and margin.

    The margin is what lets a view say "no clear constraint" instead of naming
    a station at random. On an 83%-utilisation line the quiet periods genuinely
    have no bottleneck, and a dashboard that always points somewhere teaches the
    floor to stop believing it.
    """
    order = np.argsort(-scores, axis=1)
    best, second = order[:, 0], order[:, 1]
    rows = np.arange(len(times))
    top, run = scores[rows, best], scores[rows, second]
    return pd.DataFrame({
        "t": times,
        "predicted": _smooth([ids[b] for b in best], smooth),
        "score": top,
        "runner_up": [ids[b] for b in second],
        "margin": top - run,
    })


def _smooth(pred: list[str], k: int) -> list[str]:
    """Modal filter over a trailing window of k samples.

    A raw argmax flickers between near-tied stations. A floor supervisor cannot
    act on a signal that changes every 30 seconds, so smoothing is a usability
    requirement, not a scoring trick. It is applied identically to all three
    detectors.
    """
    if k <= 1:
        return pred
    out = []
    for i in range(len(pred)):
        w = pred[max(0, i - k + 1): i + 1]
        vals, counts = np.unique(w, return_counts=True)
        out.append(str(vals[np.argmax(counts)]))
    return out


def _overlap_fraction(starts, ends, times, W):
    """Fraction of the trailing window [t-W, t] covered by the intervals."""
    out = np.zeros(len(times))
    if len(starts) == 0:
        return out
    for i, t in enumerate(times):
        lo = t - W
        ov = np.minimum(ends, t) - np.maximum(starts, lo)
        out[i] = ov[ov > 0].sum() / W
    return out


class ActivePeriodDetector(BottleneckDetector):
    """Roser's active-period method.

    online=True  -- ACTIVE FRACTION over a trailing window, tie-broken by the
                    length of the current uninterrupted run. Bounded and
                    responsive: usable on a live line.
    online=False -- the canonical offline form, ranking by the full duration of
                    the active period covering t. Scores higher but peeks past
                    t, so we report it only as an upper bound.

    The distinction from UtilisationDetector is the whole argument. Utilisation
    measures OCCUPANCY, which counts a station standing blocked with a part in
    it as busy -- so it systematically indicts the stations upstream of the real
    constraint. Active fraction counts only time a station is held up by ITSELF.
    Same event log, one definitional change, and the answer inverts.
    """

    name = "active_period"

    def __init__(self, recon: LineReconstruction, online: bool = True,
                 window: float = 900.0, smooth: int = 5):
        self.recon = recon
        self.online = online
        self.window = window
        self.smooth = smooth
        self.periods = {sid: recon.active_periods(sid) for sid in recon.ids}

    def predict(self, times: np.ndarray) -> pd.DataFrame:
        ids = self.recon.ids
        scores = np.zeros((len(times), len(ids)))
        for j, sid in enumerate(ids):
            p = self.periods[sid]
            if p.empty:
                continue
            starts, ends, durs = p.t_start.values, p.t_end.values, p.duration.values
            idx = np.searchsorted(starts, times, side="right") - 1
            k = np.clip(idx, 0, len(starts) - 1)
            inside = (idx >= 0) & (times <= ends[k])

            if self.online:
                frac = _overlap_fraction(starts, ends, times, self.window)
                run = np.where(inside, np.minimum(times - starts[k], self.window), 0.0)
                scores[:, j] = frac + 1e-3 * (run / self.window)
            else:
                scores[inside, j] = durs[k][inside]

        return _rank(times, scores, ids, self.smooth)


class UtilisationDetector(BottleneckDetector):
    name = "utilisation"

    def __init__(self, recon: LineReconstruction, window: float = 1800.0, smooth: int = 5):
        self.recon = recon
        self.window = window
        self.smooth = smooth
        self.occ = {sid: recon.occupancy(sid) for sid in recon.ids}

    def predict(self, times: np.ndarray) -> pd.DataFrame:
        ids = self.recon.ids
        scores = np.zeros((len(times), len(ids)))
        for j, sid in enumerate(ids):
            o = self.occ[sid]
            if o.empty:
                continue
            a, b = o.enter.values, o["exit"].values
            for i, t in enumerate(times):
                lo = t - self.window
                ov = np.minimum(b, t) - np.maximum(a, lo)
                scores[i, j] = ov[ov > 0].sum() / self.window
        return _rank(times, scores, ids, self.smooth)


class QueueLengthDetector(BottleneckDetector):
    name = "queue_length"

    def __init__(self, recon: LineReconstruction, smooth: int = 5):
        self.recon = recon
        self.smooth = smooth
        # the queue FEEDING station k is the buffer downstream of station k-1
        self.levels = {}
        for i, sid in enumerate(recon.ids):
            if i == 0:
                continue
            self.levels[sid] = recon.buffer(recon.ids[i - 1])

    def predict(self, times: np.ndarray) -> pd.DataFrame:
        ids = self.recon.ids
        scores = np.zeros((len(times), len(ids)))
        for j, sid in enumerate(ids):
            tl = self.levels.get(sid)
            if tl is None or tl.empty:
                continue
            idx = np.searchsorted(tl.t.values, times, side="right") - 1
            scores[:, j] = np.where(idx >= 0, tl.level.values[np.clip(idx, 0, None)], 0)
        return _rank(times, scores, ids, self.smooth)


class BottleneckWalkDetector(BottleneckDetector):
    """Active period combined with inventory observations.

    Roser, Lorentzen & Deuse (2015), "Reliable shop floor bottleneck detection
    for flow lines through process and inventory observations" -- the paper the
    Round 1 deck already cites.

    Active fraction alone has a failure mode we measured rather than assumed.
    An assembly line carries tens of buffer slots between body and final
    assembly. When a final-assembly station starts binding, that disruption
    takes HOURS to propagate back through those banks, so upstream stations
    keep running 100% active and tie with the true constraint. The tie is real
    physics, not a bug, and no amount of active-time bookkeeping resolves it.

    Inventory resolves it. The constraint is the station with a FULL buffer
    behind it and an EMPTY buffer in front: everything upstream is banking up,
    everything downstream is running dry. That gradient appears immediately,
    long before blocking propagates.

        score = active_fraction + w * (upstream_fill - downstream_fill)

    Both terms come from the event log alone -- buffer levels are exactly
    recoverable from entry and exit timestamps, so this still works at a
    station with no process sensors whatsoever.
    """

    name = "bottleneck_walk"

    # window=1800s, weight=0.6 selected by grid search on HELD-OUT seeds 11 and
    # 12, never on the reported seeds. The surface is nearly flat -- weight from
    # 0.3 to 1.5 moves the score by under 0.01 -- so the method is not living on
    # a tuned constant. Window matters more than weight: 600s is too noisy.
    def __init__(self, recon: LineReconstruction, window: float = 1800.0,
                 weight: float = 0.6, smooth: int = 5):
        self.recon = recon
        self.window = window
        self.weight = weight
        self.smooth = smooth
        self.periods = {sid: recon.active_periods(sid) for sid in recon.ids}
        self.fill: dict[str, tuple] = {}
        for i, sid in enumerate(recon.ids):
            if i == len(recon.ids) - 1:
                continue
            b = recon.buffer(sid)
            cap = max(recon.cap[sid], 1)
            self.fill[sid] = (b.t.values, np.clip(b.level.values / cap, 0, 1))

    def _fill_at(self, sid: str, times: np.ndarray) -> np.ndarray:
        d = self.fill.get(sid)
        if d is None:
            return np.zeros(len(times))
        bt, bv = d
        idx = np.searchsorted(bt, times, side="right") - 1
        return np.where(idx >= 0, bv[np.clip(idx, 0, None)], 0.0)

    def predict(self, times: np.ndarray) -> pd.DataFrame:
        ids = self.recon.ids
        scores = np.zeros((len(times), len(ids)))
        for j, sid in enumerate(ids):
            p = self.periods[sid]
            if p.empty:
                continue
            af = _overlap_fraction(p.t_start.values, p.t_end.values, times, self.window)
            up = self._fill_at(ids[j - 1], times) if j > 0 else np.ones(len(times))
            dn = self._fill_at(sid, times) if j < len(ids) - 1 else np.zeros(len(times))
            scores[:, j] = af + self.weight * (up - dn)

        return _rank(times, scores, ids, self.smooth)
