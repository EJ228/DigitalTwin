"""
Engine orchestration: runs every engine and returns one result bundle.

Used by `run_all.py` for the reported numbers and by the API for the dashboard,
so the dashboard cannot show a figure the evaluation did not produce.

Engines here
------------
  hazard      upstream-only defect risk at station k        (dtwin/hazard.py)
  blind       virtual metrology + GP posterior + EIG        (dtwin/blind.py)
  coherence   Little's Law self-audit                       (dtwin/coherence.py)
  conformal   distribution-free alert threshold             (dtwin/conformal.py)
"""

from __future__ import annotations

import functools
import pathlib
import warnings

import numpy as np
import pandas as pd

from .blind import SensorPlacement, VirtualMetrology
from .coherence import check as coherence_check
from .conformal import ConformalThreshold
from .forecast import (LSTMForecaster, linear_extrapolation, make_windows,
                       persistence, rmse, skill)
from .graphsage import GraphSAGE, line_adjacency
from .hazard import baseline_scores, fit_across_runs
from .line_config import DRIFT_STATION, build_line
from .spc import Reference, T2CUSUM

warnings.filterwarnings("ignore")
PHASE1_N = 400


def inspection_labels(run_dir) -> pd.Series:
    """Did this unit fail end-of-line inspection?

    The only label a real plant has. NOT which station caused it -- that exists
    in our simulator and would never exist on a real line, so training on it
    would be a result that cannot transfer.
    """
    td = pd.read_parquet(pathlib.Path(run_dir) / "truth_defects.parquet")
    return td.set_index("part_id").detected_at.notna()


# ---------------------------------------------------------------- hazard

def run_hazard(train_runs: list[str], test_run: str) -> dict:
    m, rep, (Xte, yte, p) = fit_across_runs(
        DRIFT_STATION, train_runs, test_run, inspection_labels)
    base = baseline_scores(yte, test_fraction=1.0)
    return {
        "station": rep.station,
        "trained_on": [pathlib.Path(r).name for r in train_runs],
        "tested_on": pathlib.Path(test_run).name,
        "n_train": rep.n_train, "n_test": rep.n_test,
        "n_features": rep.n_features,
        "positive_rate": round(rep.positive_rate, 4),
        "mcc": round(rep.mcc, 4),
        "auc": round(rep.auc, 4),
        "average_precision": round(rep.average_precision, 4),
        "baselines": {k: round(v, 4) for k, v in base.items()},
        "leak_check": rep.leak_check,
        "top_features": [{"feature": f["feature"],
                          "importance": round(f["importance"], 5),
                          "method": f["method"]} for f in rep.top_features],
    }


# ----------------------------------------------------------------- blind

def run_blind(run_dir: str, constraint_share: dict[str, float] | None = None) -> dict:
    ev = pd.read_parquet(pathlib.Path(run_dir) / "events.parquet")
    stations = build_line()
    ids = [s.station_id for s in stations]
    tiers = {s.station_id: s.tier.value for s in stations}

    vm = VirtualMetrology()
    M = vm.cycle_matrix(ev, ids)
    results, rows = {}, []
    for sid in ids:
        r, _, _ = vm.infer(M, sid, ids)
        results[sid] = r
        rows.append({
            "station": sid, "tier": tiers[sid],
            "rmse": None if not np.isfinite(r.rmse) else round(r.rmse, 3),
            "naive_rmse": None if not np.isfinite(r.naive_rmse) else round(r.naive_rmse, 3),
            "skill": None if not np.isfinite(r.skill) else round(r.skill, 3),
            "posterior_sd": None if not np.isfinite(r.posterior_sd) else round(r.posterior_sd, 3),
            "coverage_95": None if not np.isfinite(r.coverage_95) else round(r.coverage_95, 3),
            "neighbours": r.neighbours,
        })

    skills = [x["skill"] for x in rows if x["skill"] is not None]
    ranking = SensorPlacement().rank(results, tiers, constraint_share or {})
    return {
        "stations": rows,
        "ranking": ranking,
        "median_skill": round(float(np.median(skills)), 3) if skills else None,
        "n_with_positive_skill": int(sum(s > 0 for s in skills)),
        "n_evaluated": len(skills),
        "validation": ("Every station is inferred from its neighbours with its own "
                       "readings withheld, including stations we can see. That gives "
                       "an honest error bar for the same procedure at a genuinely "
                       "blind bay."),
    }


# ------------------------------------------------------------- coherence

def run_coherence(run_dir: str) -> dict:
    ev = pd.read_parquet(pathlib.Path(run_dir) / "events.parquet")
    r = coherence_check(ev, [s.station_id for s in build_line()])
    r.pop("series", None)          # the API serves the series separately
    return r


# -------------------------------------------------------------- conformal

def _s08_matrix(run_dir):
    tags = pd.read_parquet(pathlib.Path(run_dir) / "tags.parquet")
    s = tags[tags.station_id == DRIFT_STATION]
    w = s.pivot_table(index="part_id", columns="tag", values="value",
                      aggfunc="first").sort_index()
    return w, s.groupby("part_id").t.first().reindex(w.index)


def run_conformal(calib_runs: list[str], eval_runs: list[str],
                  arl0_parts: float = 1200.0) -> dict:
    names = [t.name for t in next(s for s in build_line()
                                  if s.station_id == DRIFT_STATION).tags]
    w0, _ = _s08_matrix(calib_runs[0])
    X0 = w0[names].values
    ref = Reference.fit(X0[:PHASE1_N], names)
    det = T2CUSUM(ref, threshold=np.inf)

    # calibrate on the first drift-free run, verify on the second
    fit_scores = det.statistic(X0[PHASE1_N:])
    alpha = ConformalThreshold.alpha_for_arl0(arl0_parts)
    ct = ConformalThreshold(alpha).fit(fit_scores)

    hold = []
    for r in calib_runs[1:]:
        w, _ = _s08_matrix(r)
        hold.append(det.statistic(w[names].values[PHASE1_N:]))
    holdout = np.concatenate(hold) if hold else None
    rep = ct.report(holdout)

    # detection cost at the conformal threshold
    escapes = []
    for r in eval_runs:
        w, tm = _s08_matrix(r)
        onset = float(pd.read_parquet(pathlib.Path(r) / "truth_drift.parquet").t_onset.iloc[0])
        oi = int(np.searchsorted(tm.values, onset))
        stat = det.statistic(w[names].values)
        hit = np.where(stat[oi:] > ct.threshold)[0]
        escapes.append(int(hit[0]) if len(hit) else None)
    got = [e for e in escapes if e is not None]
    rep["escape_window_parts"] = round(float(np.mean(got)), 1) if got else None
    rep["runs_detected"] = f"{len(got)}/{len(escapes)}"
    rep["target_arl0_parts"] = arl0_parts
    return rep


# -------------------------------------------------------------- forecast

GRID_STEP = 30.0
LOOKBACK = 20


def _station_panel(run_dir: str) -> tuple[np.ndarray, list[str]]:
    """(T, N, F) per-station panel on a 30-second grid, from the event log."""
    from .reconstruct import LineReconstruction

    ev = pd.read_parquet(pathlib.Path(run_dir) / "events.parquet")
    recon = LineReconstruction(ev, build_line())
    ids = recon.ids
    grid = np.arange(float(ev.t.min()) + 600.0, float(ev.t.max()), GRID_STEP)

    fill = np.zeros((len(grid), len(ids)))
    active = np.zeros((len(grid), len(ids)))
    blocked = np.zeros((len(grid), len(ids)))
    starved = np.zeros((len(grid), len(ids)))
    for j, sid in enumerate(ids):
        cap = max(recon.cap[sid], 1)
        b = recon.buffer(sid)
        if len(b) > 1:
            k = np.searchsorted(b.t.values, grid, "right") - 1
            fill[:, j] = np.clip(
                np.where(k >= 0, b.level.values[np.clip(k, 0, None)], 0) / cap, 0, 1)
        st = recon.state_intervals(sid)
        for name, arr in (("active", active), ("blocked", blocked), ("starved", starved)):
            seg = st[st.state == name]
            if seg.empty:
                continue
            starts, ends = seg.t_start.values, seg.t_end.values
            for i, t in enumerate(grid):
                lo = t - 600.0
                ov = np.minimum(ends, t) - np.maximum(starts, lo)
                arr[i, j] = ov[ov > 0].sum() / 600.0
    panel = np.stack([fill, active, blocked, starved], axis=2)
    return panel, ids


def run_forecast(train_runs: list[str], test_run: str,
                 horizons: tuple[int, ...] = (4, 10, 20, 40),
                 epochs: int = 15) -> dict:
    """Buffer-trajectory forecasting against persistence and linear extrapolation."""
    tr = [_station_panel(r)[0][:, :, 0] for r in train_runs]     # fill ratio
    te = _station_panel(test_run)[0][:, :, 0]

    rows = []
    for h in horizons:
        Xs, Ys = zip(*[make_windows(s, LOOKBACK, h) for s in tr])
        Xtr, Ytr = np.concatenate(Xs), np.concatenate(Ys)
        Xte, Yte = make_windows(te, LOOKBACK, h)
        if len(Xte) == 0:
            continue
        m = LSTMForecaster(Xtr.shape[2], 48, Ytr.shape[1], seed=0)
        m.fit(Xtr, Ytr, epochs=epochs, batch=128)
        r_l = rmse(m.predict(Xte), Yte)
        r_p = rmse(persistence(Xte), Yte)
        r_e = rmse(linear_extrapolation(Xte, h), Yte)
        rows.append({
            "horizon_min": round(h * GRID_STEP / 60, 1),
            "lstm_rmse": round(r_l, 4),
            "persistence_rmse": round(r_p, 4),
            "linear_rmse": round(r_e, 4),
            "skill_vs_persistence": round(skill(r_l, r_p), 4),
            "skill_vs_linear": round(skill(r_l, r_e), 4),
            "beats_persistence": bool(r_l < r_p),
        })
    won = sum(r["beats_persistence"] for r in rows)
    return {
        "horizons": rows,
        "beats_persistence_at": f"{won}/{len(rows)}",
        "lookback_min": LOOKBACK * GRID_STEP / 60,
        "trained_on": [pathlib.Path(r).name for r in train_runs],
        "tested_on": pathlib.Path(test_run).name,
        "note": ("Persistence is a hard baseline for buffer levels, not a straw "
                 "man: they are strongly autocorrelated over short horizons. The "
                 "comparison is reported whichever way it comes out."),
    }


# ------------------------------------------------------------- graphsage

def run_graphsage(train_runs: list[str], test_run: str, horizon: int = 20,
                  epochs: int = 60) -> dict:
    """Will this station be the constraint in ten minutes?

    Compared against an identical network with aggregation switched off, so the
    only difference between the two numbers is whether a station may see its
    neighbours.
    """
    from .detectors import BottleneckWalkDetector
    from .reconstruct import LineReconstruction
    from sklearn.metrics import matthews_corrcoef, roc_auc_score

    def prep(run_dir):
        panel, ids = _station_panel(run_dir)
        ev = pd.read_parquet(pathlib.Path(run_dir) / "events.parquet")
        recon = LineReconstruction(ev, build_line())
        grid = np.arange(float(ev.t.min()) + 600.0, float(ev.t.max()), GRID_STEP)
        pred = BottleneckWalkDetector(recon).predict(grid)
        # label: is station j the constraint `horizon` steps from now
        lab = np.zeros((len(grid), len(ids)))
        for i, sid in enumerate(pred.predicted.values):
            lab[i, ids.index(sid)] = 1.0
        X, Y = panel[:-horizon], lab[horizon:]
        mu, sd = X.reshape(-1, X.shape[2]).mean(0), X.reshape(-1, X.shape[2]).std(0) + 1e-6
        return (X - mu) / sd, Y

    tr = [prep(r) for r in train_runs]
    Xtr = np.concatenate([a for a, _ in tr])
    Ytr = np.concatenate([b for _, b in tr])
    Xte, Yte = prep(test_run)
    A = line_adjacency(Xtr.shape[1], radius=2)

    out = {}
    for name, agg in (("graphsage", True), ("no_aggregation", False)):
        m = GraphSAGE(Xtr.shape[2], 24, 16, aggregate=agg, seed=0)
        m.fit(Xtr, Ytr, A, epochs=epochs, batch=32)
        p = m.predict_proba(Xte, A).ravel()
        yt = Yte.ravel()
        ptr = m.predict_proba(Xtr, A).ravel()
        grid_t = np.unique(np.quantile(ptr, np.linspace(0.5, 0.999, 60)))
        th = max(grid_t, key=lambda t: matthews_corrcoef(Ytr.ravel(), (ptr >= t).astype(int)))
        out[name] = {
            "auc": round(float(roc_auc_score(yt, p)), 4),
            "mcc": round(float(matthews_corrcoef(yt, (p >= th).astype(int))), 4),
        }

    d_auc = out["graphsage"]["auc"] - out["no_aggregation"]["auc"]
    d_mcc = out["graphsage"]["mcc"] - out["no_aggregation"]["mcc"]
    return {
        "task": f"constraint at station in {horizon * GRID_STEP / 60:.0f} minutes",
        "trained_on": [pathlib.Path(r).name for r in train_runs],
        "tested_on": pathlib.Path(test_run).name,
        "results": out,
        "delta_auc": round(d_auc, 4),
        "delta_mcc": round(d_mcc, 4),
        "aggregation_helps": bool(d_auc > 0.01),
        "note": ("The ablation is the same network with neighbour aggregation "
                 "switched off: same features, depth, width and optimiser. An "
                 "assembly line is close to a path graph, so a small or absent "
                 "gain is a finding about lines, not a broken model."),
    }


# ------------------------------------------------------------------ all

@functools.lru_cache(maxsize=8)
def evaluate_all(data_dir: str = "data", test_run: str = "run_s7") -> dict:
    d = pathlib.Path(data_dir)
    # The test run is held out of training. Without this, asking for the engines
    # of run_s21/22/23 -- which are themselves the training set -- would fit and
    # score on the same data and report a flattering, meaningless MCC.
    train = [str(d / r) for r in ["run_s21", "run_s22", "run_s23"]
             if (d / r).exists() and r != test_run]
    calib = [str(d / r) for r in ["nodrift_s11", "nodrift_s12"] if (d / r).exists()]
    evalr = [str(d / r) for r in ["run_s7", "run_s21", "run_s22", "run_s23"]
             if (d / r).exists()]
    test = str(d / test_run)

    out = {}
    if train and (d / test_run).exists():
        out["hazard"] = run_hazard(train, test)
    out["coherence"] = run_coherence(test)

    from .detectors import BottleneckWalkDetector
    from .reconstruct import LineReconstruction
    ev = pd.read_parquet(d / test_run / "events.parquet")
    recon = LineReconstruction(ev, build_line())
    grid = np.arange(3600.0, float(ev.t.max()), 30.0)
    share = (BottleneckWalkDetector(recon).predict(grid)
             .predicted.value_counts(normalize=True).to_dict())
    out["blind"] = run_blind(test, share)

    if len(calib) >= 2 and evalr:
        out["conformal"] = run_conformal(calib, evalr)
    if train and (d / test_run).exists():
        out["forecast"] = run_forecast(train, test)
        out["graphsage"] = run_graphsage(train, test)
    return out
