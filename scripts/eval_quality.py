"""
Measure the escape window.

    python scripts/eval_quality.py

Protocol
--------
1. Phase I reference from the first 400 measured parts of each run. No detector
   is told when the drift starts.
2. Thresholds calibrated on DRIFT-FREE runs to a fixed false-alarm budget,
   expressed in parts.
3. Evaluated on drift-on runs never used for calibration.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dtwin.line_config import DRIFT_STATION, build_line
from dtwin.spc import (
    CrosierMCUSUM,
    HotellingT2,
    MEWMA,
    Reference,
    SpecLimitDetector,
    T2CUSUM,
    UnivariateCUSUM,
    calibrate_threshold,
)

PHASE1_N = 400
TARGET_ARL0 = 1200          # parts between false alarms == 20 h at 60 units/h


def load_s08(run: pathlib.Path):
    tags = pd.read_parquet(run / "tags.parquet")
    s = tags[tags.station_id == DRIFT_STATION]
    wide = s.pivot_table(index="part_id", columns="tag", values="value",
                         aggfunc="first").sort_index()
    tmap = s.groupby("part_id").t.first().reindex(wide.index)
    return wide, tmap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", nargs="+",
                    default=["data/nodrift_s11", "data/nodrift_s12"])
    ap.add_argument("--eval", nargs="+",
                    default=["data/run_s7", "data/run_s21", "data/run_s22", "data/run_s23"])
    args = ap.parse_args()

    stations = build_line()
    s08 = next(s for s in stations if s.station_id == DRIFT_STATION)
    names = [t.name for t in s08.tags]
    lsl = np.array([t.lsl for t in s08.tags])
    usl = np.array([t.usl for t in s08.tags])

    calib_sets, refs = [], []
    for c in args.calib:
        w, _ = load_s08(pathlib.Path(c))
        X = w[names].values
        refs.append(Reference.fit(X[:PHASE1_N], names))
        calib_sets.append(X[PHASE1_N:])
    ref0 = refs[0]

    print(f"=== THRESHOLD CALIBRATION (budget: 1 false alarm / {TARGET_ARL0} parts) ===")
    print(f"calibrated on drift-free runs: {args.calib}")
    thresholds = {}
    for label, factory, rng in [
        ("univariate_cusum", lambda h: UnivariateCUSUM(ref0, threshold=h), (1.0, 40.0)),
        ("hotelling_t2", lambda h: HotellingT2(ref0, threshold=h), (5.0, 400.0)),
        ("mewma", lambda h: MEWMA(ref0, threshold=h), (2.0, 300.0)),
        ("mcusum", lambda h: CrosierMCUSUM(ref0, threshold=h), (1.0, 80.0)),
        ("t2_cusum", lambda h: T2CUSUM(ref0, threshold=h), (0.5, 200.0)),
    ]:
        thresholds[label] = calibrate_threshold(
            factory, calib_sets, TARGET_ARL0, lo=rng[0], hi=rng[1])
        print(f"  {label:18s} h = {thresholds[label]:.2f}")

    rows = []
    for r in args.eval:
        run = pathlib.Path(r)
        w, tmap = load_s08(run)
        X = w[names].values
        ref = Reference.fit(X[:PHASE1_N], names)
        onset = float(pd.read_parquet(run / "truth_drift.parquet").t_onset.iloc[0])
        onset_i = int(np.searchsorted(tmap.values, onset))

        events = pd.read_parquet(run / "events.parquet")
        s08_exit = (events[(events.station_id == DRIFT_STATION)
                           & (events.event_type == "exit")]
                    .sort_values("t").reset_index(drop=True))
        pos = {p: i for i, p in enumerate(s08_exit.part_id)}
        onset_pos = int(s08_exit[s08_exit.t >= onset].index[0])

        dets = {
            "spec_limits": SpecLimitDetector(lsl, usl),
            "univariate_cusum": UnivariateCUSUM(ref, threshold=thresholds["univariate_cusum"]),
            "hotelling_t2": HotellingT2(ref, threshold=thresholds["hotelling_t2"]),
            "mewma": MEWMA(ref, threshold=thresholds["mewma"]),
            "mcusum": CrosierMCUSUM(ref, threshold=thresholds["mcusum"]),
            "t2_cusum": T2CUSUM(ref, threshold=thresholds["t2_cusum"]),
        }
        for name, det in dets.items():
            i = det.first_alarm(X, start=onset_i)
            if i is None:
                rows.append((r, name, np.nan, np.nan))
                continue
            part = int(w.index[i])
            rows.append((r, name, i - onset_i, pos.get(part, np.nan) - onset_pos))

        td = pd.read_parquet(run / "truth_defects.parquet")
        caused = td[(td.cause_mechanism == "coupling_loss") & td.is_defective]
        caused = caused[caused.part_id.map(pos).fillna(-1) >= onset_pos]
        caught = caused[caused.detected_at.notna()]
        if len(caught):
            # The escape window is how many vehicles were BUILT at S08 before
            # anyone knew -- not the build position of the part that happened to
            # be flagged. End-of-line inspection raises its flag 27 stations and
            # roughly an hour of WIP downstream, and every unit built at S08 in
            # that interval is already carrying the same fault. Measuring the
            # flagged part's own position would silently credit the baseline
            # with the whole transit lag it is actually paying.
            t_detect = float(caught.sort_values("detected_at").iloc[0].detected_at)
            built = int(np.searchsorted(s08_exit.t.values, t_detect, side="right"))
            rows.append((r, "end_of_line_inspection", np.nan, built - onset_pos))
        else:
            rows.append((r, "end_of_line_inspection", np.nan, np.nan))

    df = pd.DataFrame(rows, columns=["run", "detector", "arl1_parts", "escape_vehicles"])
    order = ["end_of_line_inspection", "spec_limits", "univariate_cusum",
             "hotelling_t2", "mewma", "mcusum", "t2_cusum"]
    g = df.groupby("detector").agg(
        runs=("escape_vehicles", "size"),
        detected=("escape_vehicles", lambda s: int(s.notna().sum())),
        mean_arl1=("arl1_parts", "mean"),
        mean_escape=("escape_vehicles", "mean"),
        worst_escape=("escape_vehicles", "max"),
    ).reindex([o for o in order if o in df.detector.values])

    print(f"\n=== ESCAPE WINDOW ({len(args.eval)} drift-on runs) ===")
    print(g.round(1).to_string())
    print("\nper-run escape window (vehicles built at S08 before the flag):")
    print(df.pivot(index="run", columns="detector", values="escape_vehicles")
            .reindex(columns=[o for o in order if o in df.detector.values]).to_string())

    if {"t2_cusum", "end_of_line_inspection"} <= set(g.index):
        a, b = g.loc["t2_cusum", "mean_escape"], g.loc["end_of_line_inspection", "mean_escape"]
        if np.isfinite(a) and np.isfinite(b):
            print(f"\nT2-CUSUM cuts the escape window from {b:.0f} vehicles to {a:.0f} "
                  f"({100*(1-a/b):.0f}% reduction), at one false alarm per "
                  f"{TARGET_ARL0} parts.")


if __name__ == "__main__":
    main()
