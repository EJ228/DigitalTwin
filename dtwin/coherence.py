"""
Little's Law coherence check -- the twin auditing itself.

    WIP = throughput x flow time

This is an identity, not a model. It holds for any stable queueing system, so
if the twin's reconstructed numbers stop satisfying it, the twin has drifted
away from the line it claims to represent.

That is the point. Every other module here produces a prediction; this one
produces a reason to distrust the predictions. Tooling wears, variants change,
buffers get resized by a shift supervisor and nobody updates the model. Without
a self-audit, an unmaintained twin does not fail loudly -- it keeps reporting
confidently and quietly becomes fiction.

Nothing here reads ground truth. WIP, throughput and flow time are all counted
from entry and exit timestamps, so the check works on a real MES feed exactly as
it works here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TOLERANCE = 0.10        # 10% relative error before we call the twin incoherent


def wip_series(events: pd.DataFrame, first: str, last: str,
               grid: np.ndarray) -> np.ndarray:
    """Parts that have entered the line but not yet left it, at each time."""
    ev = events[events.event_type.isin(["enter", "exit"])]
    ins = ev[(ev.station_id == first) & (ev.event_type == "enter")].t.values
    outs = ev[(ev.station_id == last) & (ev.event_type == "exit")].t.values
    ins, outs = np.sort(ins), np.sort(outs)
    return (np.searchsorted(ins, grid, "right")
            - np.searchsorted(outs, grid, "right")).astype(float)


def flow_time(events: pd.DataFrame, first: str, last: str) -> pd.Series:
    """Per-part time from entering the line to leaving it."""
    ev = events[events.event_type.isin(["enter", "exit"])]
    a = ev[(ev.station_id == first) & (ev.event_type == "enter")].set_index("part_id").t
    b = ev[(ev.station_id == last) & (ev.event_type == "exit")].set_index("part_id").t
    common = a.index.intersection(b.index)
    return (b.loc[common] - a.loc[common]).sort_index()


def check(events: pd.DataFrame, stations: list[str], window: float = 3600.0,
          step: float = 300.0) -> dict:
    """Rolling coherence between observed WIP and Little's Law prediction."""
    first, last = stations[0], stations[-1]
    t0, t1 = float(events.t.min()), float(events.t.max())
    grid = np.arange(t0 + window, t1, step)
    if len(grid) < 3:
        return {"available": False}

    wip = wip_series(events, first, last, grid)
    ft = flow_time(events, first, last)

    ev = events[events.event_type.isin(["enter", "exit"])]
    outs = np.sort(ev[(ev.station_id == last) & (ev.event_type == "exit")].t.values)
    ft_t = np.sort(ev[(ev.station_id == first) & (ev.event_type == "enter")]
                   .set_index("part_id").t.reindex(ft.index).dropna().values)
    ft_v = ft.values

    rows = []
    for i, t in enumerate(grid):
        lo = t - window
        thr = ((outs > lo) & (outs <= t)).sum() / window          # parts/second
        m = (ft_t > lo) & (ft_t <= t)
        if m.sum() < 5 or thr <= 0:
            continue
        mean_ft = float(np.mean(ft_v[m]))
        predicted = thr * mean_ft
        observed = wip[i]
        err = (predicted - observed) / observed if observed > 0 else np.nan
        rows.append({"t": float(t), "wip_observed": float(observed),
                     "wip_predicted": float(predicted),
                     "throughput_per_hour": float(thr * 3600),
                     "flow_time_min": mean_ft / 60.0,
                     "relative_error": float(err)})

    if not rows:
        return {"available": False}
    df = pd.DataFrame(rows)
    mae = float(df.relative_error.abs().mean())
    worst = float(df.relative_error.abs().max())
    breaches = int((df.relative_error.abs() > TOLERANCE).sum())
    return {
        "available": True,
        "tolerance": TOLERANCE,
        "mean_abs_error": round(mae, 4),
        "worst_abs_error": round(worst, 4),
        "n_windows": int(len(df)),
        "n_breaches": breaches,
        "breach_rate": round(breaches / len(df), 4),
        "coherent": bool(mae <= TOLERANCE),
        "verdict": ("Twin is coherent with the line."
                    if mae <= TOLERANCE else
                    "Twin has drifted from the line -- recalibrate before trusting it."),
        "series": df.to_dict("records"),
    }
