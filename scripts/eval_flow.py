"""
Evaluate bottleneck detection from timestamps alone.

    python scripts/eval_flow.py --run data/run_s7
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dtwin.detectors import (
    ActivePeriodDetector,
    BottleneckWalkDetector,
    QueueLengthDetector,
    UtilisationDetector,
)
from dtwin.line_config import build_line
from dtwin.reconstruct import LineReconstruction
from dtwin.scoring import (compare, detection_lag, episode_scores,
                           quiet_period_noise, shift_hit_rate)

pd.set_option("display.width", 200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="data/run_s7")
    ap.add_argument("--warmup", type=float, default=3600.0)
    args = ap.parse_args()

    run = pathlib.Path(args.run)
    events = pd.read_parquet(run / "events.parquet")
    truth_bn = pd.read_parquet(run / "truth_bottleneck.parquet")
    true_states = pd.read_parquet(run / "states.parquet")
    episodes = pd.read_parquet(run / "truth_episodes.parquet")

    stations = build_line()
    recon = LineReconstruction(events, stations)

    # ---- how good is the timestamps-only reconstruction? -----------------
    print("=== RECONSTRUCTION FIDELITY (timestamps only vs simulator truth) ===")
    inferred = recon.all_states()
    inferred["dur"] = inferred.t_end - inferred.t_start
    ts = true_states.copy()
    ts["dur"] = ts.t_end - ts.t_start
    ts["state2"] = ts.state.map(
        {"working": "active", "down": "active", "blocked": "blocked", "starved": "starved"}
    )
    A = inferred.pivot_table(index="station_id", columns="state", values="dur", aggfunc="sum")
    B = ts.pivot_table(index="station_id", columns="state2", values="dur", aggfunc="sum")
    A = A.div(A.sum(axis=1), axis=0)
    B = B.div(B.sum(axis=1), axis=0)
    common = sorted(set(A.columns) & set(B.columns))
    err = (A[common] - B[common]).abs()
    print(f"mean absolute error in state fractions: {100*err.values.mean():.2f} pp")
    print(f"worst station/state error:              {100*err.values.max():.2f} pp")
    print("\nactive fraction, inferred vs true (first 8 stations):")
    cmpdf = pd.DataFrame({"inferred": A["active"], "true": B["active"]})
    cmpdf["err_pp"] = 100 * (cmpdf.inferred - cmpdf["true"]).abs()
    print((cmpdf.head(8) * [100, 100, 1]).round(1).to_string())

    # ---- detection -------------------------------------------------------
    t0 = args.warmup
    t1 = float(events.t.max())
    times = np.arange(t0, t1, 30.0)
    truth_bn = truth_bn[(truth_bn.t >= t0 - 30) & (truth_bn.t <= t1)]

    detectors = {
        "bottleneck_walk (online)": BottleneckWalkDetector(recon),
        "active_period (online)": ActivePeriodDetector(recon, online=True),
        "active_period (offline)": ActivePeriodDetector(recon, online=False),
        "utilisation (30 min)": UtilisationDetector(recon, window=1800.0),
        "utilisation (10 min)": UtilisationDetector(recon, window=600.0),
        "queue_length": QueueLengthDetector(recon),
    }

    hits, lags, preds = {}, {}, {}
    for name, det in detectors.items():
        pred = det.predict(times)
        preds[name] = pred
        hits[name] = shift_hit_rate(pred, truth_bn)
        lags[name] = detection_lag(pred, truth_bn)

    print("\n=== EPISODE DETECTION (the metric that matches the business question) ===")
    summary = {}
    for name, pred in preds.items():
        es = episode_scores(pred, episodes)
        summary[name] = {
            "episodes": len(es),
            "detected": int(es.detected.sum()),
            "mean_hold_hit": float(es.hold_hit_rate.mean()),
            "median_lag_s": float(es.detect_lag_s.median()),
            "p90_lag_s": float(es.detect_lag_s.quantile(0.9)),
        }
    print(compare(summary).round(2).to_string())

    print("\nper-episode, bottleneck_walk (online):")
    es = episode_scores(preds["bottleneck_walk (online)"], episodes)
    print(es.round(2).to_string(index=False))

    print("\nquiet-period behaviour (no disruption active):")
    for name in ["active_period (online)", "utilisation (30 min)"]:
        print(f"  {name}: {quiet_period_noise(preds[name], episodes)}")

    print("\n=== SHIFT HIT RATE (stratified by how close the call is) ===")
    h = compare(hits)
    show = h[["hit_rate", "hit_rate_top2", "hit_marginal", "hit_moderate", "hit_decisive"]]
    print((show * 100).round(1).to_string())
    print("\nsample counts:", {k: int(h[f"n_{k}"].iloc[0]) for k in
                               ["marginal", "moderate", "decisive"]})

    print("\n=== DETECTION LAG AFTER THE BOTTLENECK MOVES ===")
    print(compare(lags).round(1).to_string())

    print("\n--- interpretation ---")
    ap_r = hits["active_period (online)"]
    ut_r = hits["utilisation (30 min)"]
    ql_r = hits["queue_length"]
    print(f"active-period beats a 30-min utilisation report by "
          f"{100*(ap_r['hit_rate']-ut_r['hit_rate']):+.1f} pp overall, "
          f"{100*(ap_r['hit_marginal']-ut_r['hit_marginal']):+.1f} pp on marginal calls")
    print(f"longest-queue heuristic: {100*ql_r['hit_rate']:.1f}% "
          "-- the deck's claim that queue length is not the bottleneck, measured")
    l_ap = lags["active_period (online)"]["median_lag_s"]
    l_ut = lags["utilisation (30 min)"]["median_lag_s"]
    print(f"median lag after a shift: active-period {l_ap/60:.1f} min "
          f"vs utilisation {l_ut/60:.1f} min")


if __name__ == "__main__":
    main()
