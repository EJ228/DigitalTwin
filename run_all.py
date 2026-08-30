#!/usr/bin/env python3
"""
One command: generate every dataset, run every check, write every result.

    python run_all.py              # full pipeline, ~3 minutes
    python run_all.py --quick      # 2 seeds instead of 4, ~1 minute

Produces
--------
    data/                 generated event logs (gitignored)
    results/results.json  machine-readable results
    results/RESULTS.md    the table that goes in the submission

Seed protocol, enforced here rather than described:
    7, 21, 22, 23   EVALUATION      drift on   -- every reported number
    11, 12          CALIBRATION     drift off  -- SPC thresholds only
    31, 32          TUNING          drift on   -- detector hyperparameters only
No seed is ever used for two purposes.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dtwin.audit import assert_no_fabricated_figures, assert_no_truth_leak
from dtwin.engines import evaluate_all
from dtwin.detectors import (
    ActivePeriodDetector,
    BottleneckWalkDetector,
    QueueLengthDetector,
    UtilisationDetector,
)
from dtwin.line_config import DRIFT_STATION, build_line
from dtwin.reconstruct import LineReconstruction
from dtwin.scoring import episode_scores
from dtwin.spc import (
    MEWMA,
    CrosierMCUSUM,
    HotellingT2,
    Reference,
    SpecLimitDetector,
    T2CUSUM,
    UnivariateCUSUM,
    calibrate_threshold,
)

EVAL_SEEDS = [7, 21, 22, 23]
CALIB_SEEDS = [11, 12]
TUNE_SEEDS = [31, 32]
PHASE1_N = 400
TARGET_ARL0 = 1200


def sh(cmd: list[str]):
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr, file=sys.stderr)
        raise SystemExit(f"failed: {' '.join(cmd)}")
    return r.stdout


def generate(seeds, shifts, drift: bool, prefix: str):
    out = []
    for s in seeds:
        d = f"data/{prefix}_s{s}"
        if not (ROOT / d / "events.parquet").exists():
            cmd = [sys.executable, "scripts/run_sim.py", "--seed", str(s),
                   "--shifts", str(shifts), "--out", d]
            if not drift:
                cmd.append("--no-drift")
            sh(cmd)
        out.append(d)
    return out


# ---------------------------------------------------------------- flow

def evaluate_flow(runs: list[str]) -> dict:
    rows = []
    for d in runs:
        run = ROOT / d
        ev = pd.read_parquet(run / "events.parquet")
        ep = pd.read_parquet(run / "truth_episodes.parquet")
        recon = LineReconstruction(ev, build_line())
        times = np.arange(3600.0, float(ev.t.max()), 30.0)
        dets = {
            "bottleneck_walk": BottleneckWalkDetector(recon),
            "active_period": ActivePeriodDetector(recon, online=True),
            "utilisation_30min": UtilisationDetector(recon, window=1800.0),
            "queue_length": QueueLengthDetector(recon),
        }
        for name, det in dets.items():
            e = episode_scores(det.predict(times), ep)
            rows.append({
                "run": d, "detector": name,
                "detected": float(e.detected.mean()),
                "hold_hit": float(e.hold_hit_rate.mean()),
                "median_lag_s": float(e.detect_lag_s.median()),
            })
    df = pd.DataFrame(rows)
    agg = (df.groupby("detector")
             .agg(detection_rate=("detected", "mean"),
                  hold_hit_rate=("hold_hit", "mean"),
                  hold_hit_sd=("hold_hit", "std"),
                  median_lag_s=("median_lag_s", "mean"))
             .sort_values("hold_hit_rate", ascending=False))
    return {"per_run": rows, "summary": agg.round(4).to_dict("index"),
            "n_episodes": int(len(runs) * 10)}


# ------------------------------------------------------------- quality

def _s08(run: pathlib.Path):
    tags = pd.read_parquet(run / "tags.parquet")
    s = tags[tags.station_id == DRIFT_STATION]
    wide = s.pivot_table(index="part_id", columns="tag", values="value",
                         aggfunc="first").sort_index()
    return wide, s.groupby("part_id").t.first().reindex(wide.index)


def evaluate_quality(eval_runs: list[str], calib_runs: list[str]) -> dict:
    s08 = next(s for s in build_line() if s.station_id == DRIFT_STATION)
    names = [t.name for t in s08.tags]
    lsl = np.array([t.lsl for t in s08.tags])
    usl = np.array([t.usl for t in s08.tags])

    calib_sets = []
    ref0 = None
    for c in calib_runs:
        w, _ = _s08(ROOT / c)
        X = w[names].values
        if ref0 is None:
            ref0 = Reference.fit(X[:PHASE1_N], names)
        calib_sets.append(X[PHASE1_N:])

    thresholds = {}
    for label, factory, rng in [
        ("univariate_cusum", lambda h: UnivariateCUSUM(ref0, threshold=h), (1.0, 40.0)),
        ("hotelling_t2", lambda h: HotellingT2(ref0, threshold=h), (5.0, 400.0)),
        ("mewma", lambda h: MEWMA(ref0, threshold=h), (2.0, 300.0)),
        ("mcusum", lambda h: CrosierMCUSUM(ref0, threshold=h), (1.0, 80.0)),
        ("t2_cusum", lambda h: T2CUSUM(ref0, threshold=h), (0.5, 200.0)),
    ]:
        thresholds[label] = float(calibrate_threshold(
            factory, calib_sets, TARGET_ARL0, lo=rng[0], hi=rng[1]))

    rows = []
    for d in eval_runs:
        run = ROOT / d
        w, tmap = _s08(run)
        X = w[names].values
        ref = Reference.fit(X[:PHASE1_N], names)
        onset = float(pd.read_parquet(run / "truth_drift.parquet").t_onset.iloc[0])
        onset_i = int(np.searchsorted(tmap.values, onset))

        ev = pd.read_parquet(run / "events.parquet")
        ex = (ev[(ev.station_id == DRIFT_STATION) & (ev.event_type == "exit")]
              .sort_values("t").reset_index(drop=True))
        pos = {p: i for i, p in enumerate(ex.part_id)}
        onset_pos = int(ex[ex.t >= onset].index[0])

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
                rows.append({"run": d, "detector": name, "escape": None, "arl1": None})
            else:
                part = int(w.index[i])
                rows.append({"run": d, "detector": name,
                             "escape": float(pos.get(part, np.nan) - onset_pos),
                             "arl1": float(i - onset_i)})

        td = pd.read_parquet(run / "truth_defects.parquet")
        caused = td[(td.cause_mechanism == "coupling_loss") & td.is_defective]
        caused = caused[caused.part_id.map(pos).fillna(-1) >= onset_pos]
        caught = caused[caused.detected_at.notna()]
        esc = None
        if len(caught):
            t_det = float(caught.sort_values("detected_at").iloc[0].detected_at)
            esc = float(int(np.searchsorted(ex.t.values, t_det, "right")) - onset_pos)
        rows.append({"run": d, "detector": "end_of_line_inspection",
                     "escape": esc, "arl1": None})

    df = pd.DataFrame(rows)
    agg = df.groupby("detector").agg(
        runs=("escape", "size"),
        detected=("escape", lambda s: int(s.notna().sum())),
        mean_escape=("escape", "mean"),
        median_escape=("escape", "median"),
        worst_escape=("escape", "max"),
    )
    return {"thresholds": thresholds, "target_arl0_parts": TARGET_ARL0,
            "per_run": rows, "summary": agg.round(2).to_dict("index")}


# ------------------------------------------------------------- report

def write_report(flow, quality, engines, meta):
    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    (res / "results.json").write_text(json.dumps(
        {"meta": meta, "flow": flow, "quality": quality, "engines": engines},
        indent=2, default=str))

    L = ["# Results", "",
         f"Generated by `run_all.py` in {meta['wall_seconds']:.0f}s. "
         f"Evaluation seeds {meta['eval_seeds']}; calibration seeds "
         f"{meta['calib_seeds']} (drift-free, thresholds only).", "",
         "## Flow: bottleneck detection", "",
         f"{flow['n_episodes']} injected disruptions. Hold-phase hit rate is the share of "
         "the full-severity window in which the detector names the degraded station.", "",
         "| Detector | Disruptions found | Hold-phase hit rate | sd across seeds |",
         "|---|---|---|---|"]
    for k, v in flow["summary"].items():
        star = "**" if k == "bottleneck_walk" else ""
        L.append(f"| {star}{k}{star} | {100*v['detection_rate']:.0f}% | "
                 f"{star}{v['hold_hit_rate']:.2f}{star} | {v['hold_hit_sd']:.3f} |")

    L += ["", "## Quality: escape window", "",
          f"Vehicles built at {DRIFT_STATION} before anything flags the drift. "
          f"Alert budget: one false alarm per {quality['target_arl0_parts']} parts "
          "(20 h at 60 units/h).", "",
          "| Detector | Runs detected | Mean escape | Median | Worst |", "|---|---|---|---|---|"]
    order = ["end_of_line_inspection", "spec_limits", "univariate_cusum",
             "mcusum", "hotelling_t2", "mewma", "t2_cusum"]
    for k in order:
        v = quality["summary"].get(k)
        if not v:
            continue
        star = "**" if k in ("t2_cusum", "end_of_line_inspection") else ""
        me = "n/a" if pd.isna(v["mean_escape"]) else f"{v['mean_escape']:.1f}"
        md = "n/a" if pd.isna(v["median_escape"]) else f"{v['median_escape']:.1f}"
        wo = "n/a" if pd.isna(v["worst_escape"]) else f"{v['worst_escape']:.0f}"
        L.append(f"| {star}{k}{star} | {v['detected']}/{v['runs']} | {star}{me}{star} | {md} | {wo} |")

    eol = quality["summary"].get("end_of_line_inspection", {}).get("mean_escape")
    ours = quality["summary"].get("t2_cusum", {}).get("mean_escape")
    if eol and ours:
        L += ["", f"**{eol:.0f} vehicles to {ours:.0f} — a {100*(1-ours/eol):.0f}% "
                  "reduction in the escape window, at a fixed alert budget.**"]
    L += ["", "## Calibrated thresholds", "",
          "| Chart | Threshold |", "|---|---|"]
    for k, v in quality["thresholds"].items():
        L.append(f"| {k} | {v:.2f} |")

    h = engines.get("hazard")
    if h:
        L += ["", "## Hazard model \u2014 defect risk at station k from stations 1..k", "",
              f"Trained on {', '.join(h['trained_on'])}, tested on {h['tested_on']}. "
              f"{h['n_features']} features, all upstream. Labels are end-of-line "
              "inspection outcomes only, never cause attribution.", "",
              "| Metric | Value |", "|---|---|",
              f"| MCC | **{h['mcc']:.3f}** |",
              f"| ROC AUC | {h['auc']:.3f} |",
              f"| Average precision | {h['average_precision']:.3f} (base rate {100*h['positive_rate']:.1f}%) |",
              f"| Predict-all-good accuracy | {100*h['baselines']['predict_all_good_accuracy']:.1f}% |",
              f"| Predict-all-good MCC | {h['baselines']['predict_all_good_mcc']:.3f} |",
              f"| Downstream leak check | {'clean' if h['leak_check']['clean'] else 'FAILED'} |", "",
              "Top features: " + ", ".join(f"`{f['feature']}`" for f in h["top_features"][:5]) + "."]

    b = engines.get("blind")
    if b:
        L += ["", "## Blind-station engine", "",
              f"Median inference skill {b['median_skill']}, positive at "
              f"{b['n_with_positive_skill']} of {b['n_evaluated']} stations. "
              "Skill is 1 - rmse/naive, where naive is the station's own historical mean.", "",
              "| Station | Tier | Skill | Posterior sd | EIG (bits) |", "|---|---|---|---|---|"]
        for r_ in b["ranking"][:6]:
            L.append(f"| {r_['station']} | {r_['tier']} | {r_['inference_skill']} | "
                     f"{r_['posterior_sd']} | {r_['eig_bits']} |")

    c = engines.get("coherence")
    if c and c.get("available"):
        L += ["", "## Little's Law self-audit", "",
              f"Mean absolute error between observed WIP and throughput x flow time: "
              f"**{100*c['mean_abs_error']:.1f}%** over {c['n_windows']} windows "
              f"(tolerance {100*c['tolerance']:.0f}%). {c['verdict']}"]

    fc = engines.get("forecast")
    if fc:
        L += ["", "## LSTM buffer forecaster", "",
              f"Trained on {', '.join(fc['trained_on'])}, tested on {fc['tested_on']}. "
              f"{fc['lookback_min']:.0f}-minute lookback. Beats persistence at "
              f"**{fc['beats_persistence_at']}** horizons.", "",
              "| Horizon | LSTM RMSE | Persistence | Linear extrap. | Skill vs persistence |",
              "|---|---|---|---|---|"]
        for r_ in fc["horizons"]:
            L.append(f"| {r_['horizon_min']:.0f} min | **{r_['lstm_rmse']:.4f}** | "
                     f"{r_['persistence_rmse']:.4f} | {r_['linear_rmse']:.4f} | "
                     f"{r_['skill_vs_persistence']:+.3f} |")
        L += ["", fc["note"]]

    gs = engines.get("graphsage")
    if gs:
        a, b = gs["results"]["graphsage"], gs["results"]["no_aggregation"]
        L += ["", "## GraphSAGE on the station topology graph", "",
              f"Task: {gs['task']}. Labels come from the detector's own output "
              "shifted forward, so this is self-supervised on the event log.", "",
              "| Model | ROC AUC | MCC |", "|---|---|---|",
              f"| **GraphSAGE (mean aggregator)** | **{a['auc']:.3f}** | **{a['mcc']:.3f}** |",
              f"| Same net, aggregation off | {b['auc']:.3f} | {b['mcc']:.3f} |",
              f"| Difference | {gs['delta_auc']:+.3f} | {gs['delta_mcc']:+.3f} |", "",
              gs["note"]]

    k = engines.get("conformal")
    if k:
        L += ["", "## Conformal alert calibration", "",
              f"Nominal false-alarm rate alpha = {k['alpha']:.5f} "
              f"(1 per {k['target_arl0_parts']:.0f} parts). "
              f"Threshold {k['threshold']} from {k['calibration_n']} in-control scores; "
              f"finite-sample exact: {k['finite_sample_exact']}. "
              f"Empirical rate on held-out drift-free runs: "
              f"{k.get('empirical_false_alarm_rate')}. "
              f"Escape window at this threshold: {k['escape_window_parts']} parts "
              f"({k['runs_detected']} runs detected).", "",
              f"> {k['guarantee']}"]
    (res / "RESULTS.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="2 eval seeds, 2 shifts")
    ap.add_argument("--shifts", type=float, default=3.0)
    args = ap.parse_args()

    t0 = time.time()
    eval_seeds = EVAL_SEEDS[:2] if args.quick else EVAL_SEEDS
    shifts = 2.0 if args.quick else args.shifts

    print("[1/6] anti-circularity audit")
    assert_no_truth_leak(ROOT / "dtwin")
    print("      clean: no prediction module references ground truth")
    assert_no_fabricated_figures(ROOT / "web" / "src")
    print("      clean: no fabricated figures in the frontend display path")

    print("[2/6] generating datasets")
    ev_runs = generate(eval_seeds, shifts, True, "run")
    ca_runs = generate(CALIB_SEEDS, shifts, False, "nodrift")
    generate(TUNE_SEEDS, shifts, True, "tune")
    print(f"      {len(ev_runs)} evaluation, {len(ca_runs)} calibration, {len(TUNE_SEEDS)} tuning")

    print("[3/6] invariant suite")
    out = sh([sys.executable, "scripts/test_invariants.py"])
    print("      " + out.strip().splitlines()[-1])

    print("[4/6] flow engine")
    flow = evaluate_flow(ev_runs)
    for k, v in flow["summary"].items():
        print(f"      {k:20s} found {100*v['detection_rate']:3.0f}%  hit {v['hold_hit_rate']:.2f}")

    print("[5/6] quality engine")
    quality = evaluate_quality(ev_runs, ca_runs)
    for k in ["end_of_line_inspection", "univariate_cusum", "mcusum", "t2_cusum"]:
        v = quality["summary"].get(k)
        if v:
            print(f"      {k:24s} escape {v['mean_escape']:6.1f}  ({v['detected']}/{v['runs']} runs)")

    print("[6/6] hazard, blind, coherence, conformal, forecast, graphsage")
    engines = evaluate_all("data", "run_s7")
    if "hazard" in engines:
        h = engines["hazard"]
        print(f"      hazard      MCC {h['mcc']:.3f}  AUC {h['auc']:.3f}  "
              f"(all-good baseline MCC {h['baselines']['predict_all_good_mcc']:.3f})")
    if "blind" in engines:
        b = engines["blind"]
        print(f"      blind       median skill {b['median_skill']}  "
              f"positive at {b['n_with_positive_skill']}/{b['n_evaluated']} stations")
    if "coherence" in engines:
        c = engines["coherence"]
        print(f"      coherence   WIP error {100*c['mean_abs_error']:.1f}%  coherent={c['coherent']}")
    if "conformal" in engines:
        k = engines["conformal"]
        print(f"      conformal   threshold {k['threshold']}  escape {k['escape_window_parts']} parts")
    if "forecast" in engines:
        f_ = engines["forecast"]
        print(f"      forecast    beats persistence at {f_['beats_persistence_at']} horizons")
    if "graphsage" in engines:
        g_ = engines["graphsage"]
        print(f"      graphsage   AUC {g_['results']['graphsage']['auc']:.3f} vs "
              f"{g_['results']['no_aggregation']['auc']:.3f} without aggregation "
              f"({g_['delta_auc']:+.3f})")

    meta = {"wall_seconds": time.time() - t0, "eval_seeds": eval_seeds,
            "calib_seeds": CALIB_SEEDS, "tune_seeds": TUNE_SEEDS, "shifts": shifts}
    write_report(flow, quality, engines, meta)
    print(f"\ndone in {meta['wall_seconds']:.0f}s -> results/RESULTS.md, results/results.json")


if __name__ == "__main__":
    main()
