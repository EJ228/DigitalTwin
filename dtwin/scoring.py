"""
Scoring. THIS IS THE ONLY MODULE PERMITTED TO READ A TRUTH TABLE.

Detectors read events. The scorer reads truth. Keeping the boundary at a module
level means "did you tune on the labels?" has a checkable answer rather than a
reassuring one.

Headline metric is shift hit rate: the fraction of sampled moments at which the
detector names the station that genuinely binds. It is reported stratified by
MARGIN -- how far the true bottleneck leads its runner-up in theoretical load.

The stratification is the point. On decisive cases (a station degraded well past
everything else) any method looks good, so an unstratified average mostly
measures how many easy moments happen to be in the run. The marginal cases are
where a utilisation report gets it wrong, and they are the ones that decide
whether the detector is worth deploying.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MARGIN_BINS = [
    ("marginal", 0.0, 1.0),      # under 1 s of load separates first from second
    ("moderate", 1.0, 4.0),
    ("decisive", 4.0, np.inf),
]


def align_truth(pred: pd.DataFrame, truth_bn: pd.DataFrame) -> pd.DataFrame:
    """Attach the true bottleneck (and its margin) to each prediction time."""
    tb = truth_bn.sort_values("t")
    idx = np.searchsorted(tb.t.values, pred.t.values, side="right") - 1
    idx = np.clip(idx, 0, len(tb) - 1)
    out = pred.copy()
    out["truth"] = tb.true_bottleneck.values[idx]
    out["margin"] = tb.margin.values[idx]
    out["runner_up"] = tb.runner_up.values[idx]
    out["hit"] = out.predicted == out.truth
    out["hit_top2"] = out.hit | (out.predicted == out.runner_up)
    return out


def shift_hit_rate(pred: pd.DataFrame, truth_bn: pd.DataFrame) -> dict:
    a = align_truth(pred, truth_bn)
    res = {
        "n": int(len(a)),
        "hit_rate": float(a.hit.mean()),
        "hit_rate_top2": float(a.hit_top2.mean()),
    }
    for name, lo, hi in MARGIN_BINS:
        m = a[(a.margin >= lo) & (a.margin < hi)]
        res[f"hit_{name}"] = float(m.hit.mean()) if len(m) else float("nan")
        res[f"n_{name}"] = int(len(m))
    return res


def detection_lag(pred: pd.DataFrame, truth_bn: pd.DataFrame) -> dict:
    """How long after the bottleneck moves does the detector notice?

    For each genuine change in the true bottleneck, measure the delay until the
    detector's output settles on the new station for three consecutive samples.
    This is the quantity behind the Round 1 claim that a static report "names
    the bottleneck after it has already moved".
    """
    a = align_truth(pred, truth_bn).sort_values("t").reset_index(drop=True)
    changes = a.index[a.truth != a.truth.shift()].tolist()[1:]
    lags, missed = [], 0
    for ci in changes:
        target = a.truth.iloc[ci]
        window = a.iloc[ci: ci + 200]
        settled = None
        run = 0
        for _, row in window.iterrows():
            run = run + 1 if row.predicted == target else 0
            if run >= 3:
                settled = row.t
                break
        if settled is None:
            missed += 1
        else:
            lags.append(settled - a.t.iloc[ci])
    return {
        "n_shifts": len(changes),
        "n_detected": len(lags),
        "n_missed": missed,
        "median_lag_s": float(np.median(lags)) if lags else float("nan"),
        "p90_lag_s": float(np.percentile(lags, 90)) if lags else float("nan"),
    }


def compare(results: dict[str, dict]) -> pd.DataFrame:
    return pd.DataFrame(results).T


# --------------------------------------------------------------------------
# Episode-based scoring
# --------------------------------------------------------------------------

def episode_scores(pred: pd.DataFrame, episodes: pd.DataFrame,
                   settle: int = 3, dt: float = 30.0) -> pd.DataFrame:
    """Score against the injected disruptions.

    This is the metric that matches the business question. A plant does not
    care who holds the theoretical maximum load during a quiet hour; it cares
    that when a station starts binding, the system names it, and names it fast.

    We deliberately moved to this after finding that a purely config-derived
    "theoretical load" ground truth can name a station that has not started
    degrading yet, while a station still saturated from an earlier disruption
    is genuinely the constraint. The episode is the thing we injected and
    therefore the thing we can honestly claim to have detected.

    For each episode:
      hit_rate  -- share of the HOLD phase (full severity) named correctly
      lag_s     -- seconds from ramp start until the detector settles on the
                   station for `settle` consecutive samples
    """
    rows = []
    for e in episodes.itertuples(index=False):
        hold = pred[(pred.t >= e.t_hold_start) & (pred.t <= e.t_hold_end)]
        hit = float((hold.predicted == e.station_id).mean()) if len(hold) else np.nan

        win = pred[(pred.t >= e.t_start) & (pred.t <= e.t_end)]
        lag, run = np.nan, 0
        for _, r in win.iterrows():
            run = run + 1 if r.predicted == e.station_id else 0
            if run >= settle:
                lag = r.t - (settle - 1) * dt - e.t_start
                break
        rows.append({
            "station": e.station_id, "severity": e.severity, "label": e.label,
            "hold_hit_rate": hit, "detect_lag_s": lag,
            "detected": bool(np.isfinite(lag)),
        })
    return pd.DataFrame(rows)


def quiet_period_noise(pred: pd.DataFrame, episodes: pd.DataFrame) -> dict:
    """How often does the detector name a station during undisrupted periods?

    Reported for honesty. Outside a disruption an 83%-utilisation line has
    slack and there is no meaningful constraint, so any confident answer here
    is noise rather than insight.
    """
    mask = np.ones(len(pred), dtype=bool)
    for e in episodes.itertuples(index=False):
        mask &= ~((pred.t.values >= e.t_start) & (pred.t.values <= e.t_end))
    quiet = pred[mask]
    if quiet.empty:
        return {"n_quiet": 0}
    vc = quiet.predicted.value_counts(normalize=True)
    return {
        "n_quiet": int(len(quiet)),
        "top_station": str(vc.index[0]),
        "top_share": float(vc.iloc[0]),
        "distinct_stations": int(quiet.predicted.nunique()),
    }
