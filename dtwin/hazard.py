"""
Discrete-time hazard model: probability that a part is being spoiled at station k.

THE STRUCTURAL CONSTRAINT
-------------------------
The model may only see stations 1..k. Everything downstream is dropped before
training, not masked at inference. That matters: a model with downstream columns
would learn to *explain* a defect after the fact (the failing torque check at
station 31 predicts the failure at station 31), which scores beautifully and is
operationally worthless. Blinding it to the future forces it to predict.

WHAT IT TRAINS ON
-----------------
End-of-line inspection outcomes -- did this unit fail inspection -- which is
data every plant already has. It is NOT trained on which station caused the
defect. That label exists in our simulator and would never exist in a real
plant, so using it would be cheating in a way that does not transfer. The model
therefore has to learn attribution rather than being handed it.

SCORING
-------
Matthews correlation coefficient. At a sub-1% defect rate a model that predicts
"fine" for everything scores 99% accuracy, so accuracy is not a metric, it is a
way of not noticing. MCC collapses to zero for that model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

from .line_config import build_line


@dataclass
class HazardReport:
    station: str
    n_train: int
    n_test: int
    n_features: int
    positive_rate: float
    mcc: float
    auc: float
    average_precision: float
    threshold: float
    top_features: list = field(default_factory=list)
    leak_check: dict = field(default_factory=dict)


class UpstreamHazardModel:
    """Hazard at station k from stations 1..k only."""

    def __init__(self, station: str, horizon_only: bool = True):
        self.station = station
        self.stations = [s.station_id for s in build_line()]
        self.k = self.stations.index(station)
        self.upstream = self.stations[: self.k + 1]
        self.model: HistGradientBoostingClassifier | None = None
        self.features: list[str] = []
        self.threshold = 0.5

    # ------------------------------------------------------------------

    def build_features(self, events: pd.DataFrame, tags: pd.DataFrame) -> pd.DataFrame:
        """One row per part, columns from upstream stations only.

        Flow features come from timestamps, so they exist for every station
        including the blind ones. Process features come from tags and therefore
        only exist where there is instrumentation -- which is the honest
        situation and exactly why we use a model that handles missing values
        natively instead of imputing and pretending.
        """
        ev = events[events.event_type.isin(["enter", "exit"])]
        ev = ev[ev.station_id.isin(self.upstream)]

        enters = ev[ev.event_type == "enter"][["part_id", "station_id", "t", "variant"]]
        exits = ev[ev.event_type == "exit"][["part_id", "station_id", "t"]]
        occ = enters.merge(exits, on=["part_id", "station_id"], suffixes=("_in", "_out"))
        occ["dwell"] = occ.t_out - occ.t_in

        # per-station dwell time
        wide = occ.pivot_table(index="part_id", columns="station_id",
                               values="dwell", aggfunc="first")
        wide.columns = [f"dwell_{c}" for c in wide.columns]

        # inter-station wait: how long the part sat between stations
        occ = occ.sort_values(["part_id", "t_in"])
        occ["prev_out"] = occ.groupby("part_id").t_out.shift()
        occ["wait"] = occ.t_in - occ.prev_out
        waits = occ.pivot_table(index="part_id", columns="station_id",
                                values="wait", aggfunc="first")
        waits.columns = [f"wait_{c}" for c in waits.columns]

        f = wide.join(waits, how="outer")

        # cumulative flow summaries
        f["cum_dwell"] = wide.sum(axis=1, skipna=True)
        f["cum_wait"] = waits.sum(axis=1, skipna=True)
        f["max_dwell"] = wide.max(axis=1, skipna=True)
        f["dwell_std"] = wide.std(axis=1, skipna=True)

        # variant, one-hot
        var = enters.groupby("part_id").variant.first()
        for v in sorted(var.dropna().unique()):
            f[f"variant_{v}"] = (var.reindex(f.index) == v).astype(float)

        # process tags at upstream instrumented stations
        tg = tags[tags.station_id.isin(self.upstream)]
        if len(tg):
            tw = tg.pivot_table(index="part_id", columns="tag", values="value",
                                aggfunc="first")
            f = f.join(tw, how="left")

            # pairwise interactions at the drift station: the fault we care about
            # lives in a RELATIONSHIP between tags, so a model given only raw
            # levels would have to rediscover the difference from scratch.
            gl, gr = "s08_gap_left_mm", "s08_gap_right_mm"
            if gl in f.columns and gr in f.columns:
                f["s08_gap_diff_abs"] = (f[gl] - f[gr]).abs()
                f["s08_gap_sum"] = f[gl] + f[gr]

        f = f.sort_index()
        # guarantee no downstream column ever entered the frame
        bad = [c for c in f.columns
               for s in self.stations[self.k + 1:]
               if s.lower() in c.lower()]
        assert not bad, f"downstream leak: {bad[:5]}"
        return f

    # ------------------------------------------------------------------

    def fit_evaluate(self, X: pd.DataFrame, y: pd.Series,
                     test_fraction: float = 0.3) -> HazardReport:
        """Chronological split. Parts are ordered, so a random split would let
        the model see the future of the very drift it is meant to catch."""
        common = X.index.intersection(y.index)
        X, y = X.loc[common], y.loc[common].astype(int)
        n = len(X)
        cut = int(n * (1 - test_fraction))
        # Three-way chronological split. The operating point is chosen on a
        # VALIDATION slice taken from the end of train, not on all of train:
        # the drift arrives partway through the run, so early train parts are a
        # different regime and a threshold fitted to them does not transfer.
        vcut = int(cut * 0.78)
        Xtr, Xva, Xte = X.iloc[:vcut], X.iloc[vcut:cut], X.iloc[cut:]
        ytr, yva, yte = y.iloc[:vcut], y.iloc[vcut:cut], y.iloc[cut:]

        pos = float(ytr.mean())
        self.model = HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.06, max_depth=5,
            l2_regularization=1.0, min_samples_leaf=25,
            class_weight="balanced",       # the positive class is under 1%
            random_state=0,
        )
        self.features = list(X.columns)
        self.model.fit(Xtr, ytr)

        p = self.model.predict_proba(Xte)[:, 1]
        # choose the operating point that maximises MCC on TRAIN, then apply it
        # to test -- picking it on test would be reporting the best of many tries
        pva = self.model.predict_proba(Xva)[:, 1]
        if yva.nunique() > 1:
            grid = np.unique(np.quantile(pva, np.linspace(0.05, 0.995, 120)))
            best = max(grid, key=lambda th: matthews_corrcoef(yva, (pva >= th).astype(int)))
        else:
            best = float(np.quantile(pva, 1 - 3 * max(pos, 0.01)))
        self.threshold = float(best)

        yhat = (p >= self.threshold).astype(int)
        report = HazardReport(
            station=self.station, n_train=len(Xtr) + len(Xva), n_test=len(Xte),
            n_features=X.shape[1], positive_rate=pos,
            mcc=float(matthews_corrcoef(yte, yhat)),
            auc=float(roc_auc_score(yte, p)) if yte.nunique() > 1 else float("nan"),
            average_precision=float(average_precision_score(yte, p))
            if yte.nunique() > 1 else float("nan"),
            threshold=self.threshold,
            leak_check=self.leak_check(),
        )
        report.top_features = self.explain(Xte, top=8)
        return report

    # ------------------------------------------------------------------

    def explain(self, X: pd.DataFrame, top: int = 8) -> list[dict]:
        """SHAP attribution, with a permutation fallback.

        SHAP on a sample rather than the full test set: the ranking is stable
        well before the extra precision matters, and the demo has to stay
        interactive.
        """
        try:
            import shap

            sample = X.sample(min(len(X), 300), random_state=0)
            expl = shap.Explainer(self.model, sample)
            vals = expl(sample, check_additivity=False)
            imp = np.abs(vals.values).mean(axis=0)
            method = "shap"
        except Exception:
            from sklearn.inspection import permutation_importance

            sample = X.sample(min(len(X), 400), random_state=0)
            pred = self.model.predict_proba(sample)[:, 1]
            base = pred.std() or 1.0
            imp = np.zeros(X.shape[1])
            rng = np.random.default_rng(0)
            for j in range(X.shape[1]):
                s2 = sample.copy()
                s2.iloc[:, j] = rng.permutation(s2.iloc[:, j].values)
                imp[j] = np.abs(self.model.predict_proba(s2)[:, 1] - pred).mean() / base
            method = "permutation"

        order = np.argsort(-imp)[:top]
        return [{"feature": self.features[j], "importance": float(imp[j]),
                 "method": method} for j in order]

    def leak_check(self) -> dict:
        """Assert no feature names a downstream station."""
        downstream = self.stations[self.k + 1:]
        offenders = [f for f in self.features
                     if any(s.lower() in f.lower() for s in downstream)]
        return {"downstream_stations": len(downstream),
                "features_referencing_downstream": offenders,
                "clean": not offenders}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.features])[:, 1]


def fit_across_runs(station: str, train_runs: list, test_run, label_fn) -> tuple:
    """Train on several runs, test on a held-out one.

    A single run contains a single drift event, so a chronological split inside
    it puts every positive example in the test half and the model never sees a
    drift while training -- it scored at chance. Pooling several runs and
    holding one out is both the fix and the more honest experiment: it asks
    whether the model generalises to a shift it has never seen.

    label_fn(run_dir) -> Series indexed by part_id, True if the unit failed
    end-of-line inspection. Inspection outcomes only; never cause attribution.
    """
    import pandas as pd

    m = UpstreamHazardModel(station)

    def load(run_dir):
        ev = pd.read_parquet(f"{run_dir}/events.parquet")
        tg = pd.read_parquet(f"{run_dir}/tags.parquet")
        X = m.build_features(ev, tg)
        y = label_fn(run_dir)
        common = X.index.intersection(y.index)
        return X.loc[common].reset_index(drop=True), y.loc[common].astype(int).reset_index(drop=True)

    parts = [load(r) for r in train_runs]
    Xtr = pd.concat([a for a, _ in parts], ignore_index=True)
    ytr = pd.concat([b for _, b in parts], ignore_index=True)
    Xte, yte = load(test_run)
    Xte = Xte.reindex(columns=Xtr.columns)

    m.features = list(Xtr.columns)
    m.model = HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.06, max_depth=5, l2_regularization=1.0,
        min_samples_leaf=25, class_weight="balanced", random_state=0)
    m.model.fit(Xtr, ytr)

    # Operating point chosen on TRAIN only. Choosing it on test would be
    # reporting the best of many tries.
    ptr = m.model.predict_proba(Xtr)[:, 1]
    grid = np.unique(np.quantile(ptr, np.linspace(0.05, 0.995, 150)))
    m.threshold = float(max(grid, key=lambda t: matthews_corrcoef(ytr, (ptr >= t).astype(int))))

    p = m.model.predict_proba(Xte)[:, 1]
    yhat = (p >= m.threshold).astype(int)
    report = HazardReport(
        station=station, n_train=len(Xtr), n_test=len(Xte), n_features=Xtr.shape[1],
        positive_rate=float(yte.mean()),
        mcc=float(matthews_corrcoef(yte, yhat)),
        auc=float(roc_auc_score(yte, p)) if yte.nunique() > 1 else float("nan"),
        average_precision=float(average_precision_score(yte, p)) if yte.nunique() > 1 else float("nan"),
        threshold=m.threshold, leak_check=m.leak_check(),
    )
    report.top_features = m.explain(Xte, top=8)
    return m, report, (Xte, yte, p)


def baseline_scores(y: pd.Series, test_fraction: float = 0.3) -> dict:
    """What trivial models score, so the real number has a floor to beat."""
    y = y.astype(int)
    cut = int(len(y) * (1 - test_fraction))
    yte = y.iloc[cut:]
    rng = np.random.default_rng(0)
    always_ok = np.zeros(len(yte), dtype=int)
    coin = (rng.random(len(yte)) < max(y.iloc[:cut].mean(), 1e-6)).astype(int)
    return {
        "predict_all_good_accuracy": float((always_ok == yte).mean()),
        "predict_all_good_mcc": float(matthews_corrcoef(yte, always_ok))
        if yte.nunique() > 1 else 0.0,
        "random_at_base_rate_mcc": float(matthews_corrcoef(yte, coin))
        if yte.nunique() > 1 else 0.0,
    }
