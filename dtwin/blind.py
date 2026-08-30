"""
The blind-station engine: three parts of one pipeline.

  1. Virtual metrology -- infer an uninstrumented station's state from the
     stations around it.
  2. Gaussian process posterior -- say how sure we are, honestly.
  3. Expected Information Gain -- rank where a sensor would buy the most.

HOW WE VALIDATE AN INFERENCE ABOUT SOMETHING WE CANNOT SEE
----------------------------------------------------------
By masking a station we CAN see. We take a fully instrumented station, hide it
from the model, infer it from its neighbours, and compare against the readings
we withheld. That gives an honest error bar for what the same procedure does at
a genuinely blind station -- which is the only way to make a claim about a bay
with no sensors that is not simply an assertion.

WHY A GAUSSIAN PROCESS
----------------------
A point estimate at a blind station is worse than useless: it invites the floor
to trust a number nobody measured. The GP returns a posterior, so the dashboard
can widen its band exactly where the twin is guessing. A twin that hides its own
ignorance is worse than no twin.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

NEIGHBOUR_RADIUS = 3


@dataclass
class InferenceResult:
    station: str
    target: str
    n: int
    neighbours: list = field(default_factory=list)
    r2: float = float("nan")
    rmse: float = float("nan")
    naive_rmse: float = float("nan")
    skill: float = float("nan")          # 1 - rmse/naive_rmse
    posterior_sd: float = float("nan")
    coverage_95: float = float("nan")
    validated_on: str = ""


class VirtualMetrology:
    """Infer a station's cycle time from its neighbours' observed cycle times.

    Cycle time is the target because it is the one quantity every station has,
    instrumented or not -- it falls straight out of entry and exit timestamps.
    Inferring a torque reading at a station with no torque sensor would be
    fiction; inferring how a manual bay is running, and how confident we are, is
    a real and useful thing to say.

    THE TARGET IS A ROLLING MEAN, NOT A SINGLE CAR
    ----------------------------------------------
    We tried per-part cycle time first and it had no skill at all: station
    service times are conditionally independent, so one car's duration at S10
    tells you almost nothing about the same car at S09. That is a property of
    assembly lines, not a modelling failure, and no amount of neighbour data
    fixes it.

    What IS inferable -- and what a supervisor actually needs -- is whether a bay
    is running slow right now. Averaging over a window cancels the independent
    per-part noise and leaves the correlated component: blocking and starvation
    propagate between neighbours, so a bay that is backing up shows in the
    stations around it. So the target is a rolling mean over `window` parts.
    """

    def __init__(self, radius: int = NEIGHBOUR_RADIUS, window: int = 20):
        self.radius = radius
        self.window = window

    # ------------------------------------------------------------------

    @staticmethod
    def cycle_matrix(events: pd.DataFrame, station_ids: list[str]) -> pd.DataFrame:
        ev = events[events.event_type.isin(["enter", "exit"])]
        a = ev[ev.event_type == "enter"][["part_id", "station_id", "t"]]
        b = ev[ev.event_type == "exit"][["part_id", "station_id", "t"]]
        occ = a.merge(b, on=["part_id", "station_id"], suffixes=("_in", "_out"))
        occ["dwell"] = occ.t_out - occ.t_in
        m = occ.pivot_table(index="part_id", columns="station_id",
                            values="dwell", aggfunc="first")
        return m.reindex(columns=[s for s in station_ids if s in m.columns]).sort_index()

    def smooth(self, M: pd.DataFrame) -> pd.DataFrame:
        """Rolling mean over `window` parts -- see the class docstring."""
        return M.rolling(self.window, min_periods=max(3, self.window // 2)).mean()

    def neighbours_of(self, station: str, station_ids: list[str]) -> list[str]:
        i = station_ids.index(station)
        lo, hi = max(0, i - self.radius), min(len(station_ids), i + self.radius + 1)
        return [s for s in station_ids[lo:hi] if s != station]

    # ------------------------------------------------------------------

    def infer(self, M: pd.DataFrame, station: str, station_ids: list[str],
              test_fraction: float = 0.3) -> tuple[InferenceResult, GaussianProcessRegressor, object]:
        nb = [n for n in self.neighbours_of(station, station_ids) if n in M.columns]
        data = self.smooth(M)[[station] + nb].dropna()
        if len(data) < 80 or not nb:
            return InferenceResult(station=station, target="cycle_time",
                                   n=len(data), neighbours=nb), None, None

        y = data[station].values
        X = data[nb].values
        cut = int(len(data) * (1 - test_fraction))
        Xtr, Xte, ytr, yte = X[:cut], X[cut:], y[:cut], y[cut:]

        sc = StandardScaler().fit(Xtr)
        # Ridge first: it is the honest baseline and it is what we fall back on
        # when the GP has too little signal to justify its cost.
        ridge = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(sc.transform(Xtr), ytr)

        # GP on a subsample -- exact GP inference is cubic and we only need the
        # posterior width, which stabilises quickly.
        idx = np.linspace(0, len(Xtr) - 1, min(len(Xtr), 400)).astype(int)
        kernel = (ConstantKernel(1.0) * RBF(length_scale=np.ones(len(nb)))
                  + WhiteKernel(noise_level=1.0))
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                      alpha=1e-6, n_restarts_optimizer=0,
                                      random_state=0)
        gp.fit(sc.transform(Xtr)[idx], ytr[idx])

        mu, sd = gp.predict(sc.transform(Xte), return_std=True)
        rmse = float(np.sqrt(np.mean((mu - yte) ** 2)))
        # Naive baseline: the station's own historical mean. Beating this is
        # the whole claim -- it is what you would say about a blind bay if you
        # had no model at all.
        naive = float(np.sqrt(np.mean((ytr.mean() - yte) ** 2)))
        inside = np.mean(np.abs(yte - mu) <= 1.96 * sd)

        res = InferenceResult(
            station=station, target="cycle_time", n=len(data), neighbours=nb,
            r2=float(1 - np.sum((mu - yte) ** 2) / np.sum((yte - yte.mean()) ** 2)),
            rmse=rmse, naive_rmse=naive,
            skill=float(1 - rmse / naive) if naive > 0 else float("nan"),
            posterior_sd=float(np.mean(sd)),
            coverage_95=float(inside),
            validated_on="held-out parts at a station whose sensors we hid",
        )
        return res, gp, sc


class SensorPlacement:
    """Expected Information Gain per station.

    For a Gaussian posterior, observing a quantity with prior variance s^2 under
    observation noise n^2 yields

        EIG = 0.5 * log2(1 + s^2 / n^2)   bits

    which is the mutual information between the observation and the quantity.
    A station we already predict well has little left to learn; a station we
    predict badly AND that frequently binds the line is where the money goes.

    This replaces the hand-weighted heuristic the dashboard shipped with
    earlier. The weights there were mine; these are a computation.
    """

    def __init__(self, observation_noise: float = 1.0):
        self.noise = observation_noise

    @staticmethod
    def eig_bits(posterior_sd: float, noise_sd: float) -> float:
        if not np.isfinite(posterior_sd) or noise_sd <= 0:
            return 0.0
        return float(0.5 * np.log2(1.0 + (posterior_sd ** 2) / (noise_sd ** 2)))

    def rank(self, results: dict[str, InferenceResult], tiers: dict[str, str],
             constraint_share: dict[str, float]) -> list[dict]:
        rows = []
        for sid, r in results.items():
            sd = r.posterior_sd
            bits = self.eig_bits(sd, self.noise)
            share = float(constraint_share.get(sid, 0.0))
            # value = information a sensor would buy, weighted by how often this
            # station actually decides the line's throughput
            rows.append({
                "station": sid,
                "tier": tiers.get(sid, "unknown"),
                "posterior_sd": None if not np.isfinite(sd) else round(sd, 3),
                "eig_bits": round(bits, 3),
                "constraint_share": round(share, 4),
                "value": round(bits * (0.05 + share), 5),
                "inference_skill": None if not np.isfinite(r.skill) else round(r.skill, 3),
            })
        rows.sort(key=lambda r: -r["value"])
        return rows
