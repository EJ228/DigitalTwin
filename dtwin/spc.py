"""
Statistical process control for the fixture-drift scenario.

Five detectors, run on the same tag stream, so the Round 1 claim that "a drift
within spec on every tag alarms on none" becomes a measurement.

  1. SpecLimitDetector   -- the alarm most plants actually have. Fires when any
                            tag crosses its engineering spec limit.
  2. UnivariateCUSUM     -- a proper tabular CUSUM per tag, k=0.5s, alarm if ANY
                            tag's chart fires. A fair univariate opponent, not a
                            strawman: it accumulates, it just cannot see across
                            tags.
  3. HotellingT2         -- single-point multivariate. Sees the correlation
                            structure but has no memory.
  4. MEWMA               -- Lowry et al. (1992). Multivariate with memory.
  5. CrosierMCUSUM       -- Crosier (1988). Multivariate with memory, tuned for
                            sustained small shifts, which is what a drifting
                            fixture is.

PHASE I. All five estimate their in-control reference (mean, covariance, spec
limits) from the FIRST 400 measured parts of the run, long before drift onset.
No detector is told when the drift starts. That is a production-realistic
protocol: you commission the chart on a known-good period and then run it.

THRESHOLDS are calibrated to a fixed false-alarm budget on separate drift-free
runs, never on the run being reported. See scripts/eval_quality.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Reference:
    """Phase I in-control reference."""

    mu: np.ndarray
    cov: np.ndarray
    sigma: np.ndarray
    inv: np.ndarray
    names: list[str]

    @classmethod
    def fit(cls, X: np.ndarray, names: list[str]) -> "Reference":
        mu = X.mean(axis=0)
        cov = np.cov(X.T)
        sigma = X.std(axis=0, ddof=1)
        return cls(mu, cov, sigma, np.linalg.pinv(cov), names)


class SPCDetector(ABC):
    name: str
    threshold: float

    @abstractmethod
    def statistic(self, X: np.ndarray) -> np.ndarray:
        """Monitoring statistic, one value per observation."""

    def first_alarm(self, X: np.ndarray, start: int = 0) -> int | None:
        s = self.statistic(X)
        idx = np.where(s[start:] > self.threshold)[0]
        return int(idx[0] + start) if len(idx) else None

    def alarm_indices(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.statistic(X) > self.threshold)[0]


class SpecLimitDetector(SPCDetector):
    """Fires when any single tag leaves its engineering spec window.

    This is the incumbent. The Round 1 deck asserts it cannot see a drift that
    stays inside every tag's limits; this detector exists to prove that.
    """

    name = "spec_limits"

    def __init__(self, lsl: np.ndarray, usl: np.ndarray):
        self.lsl, self.usl = lsl, usl
        self.threshold = 0.5

    def statistic(self, X: np.ndarray) -> np.ndarray:
        return ((X < self.lsl) | (X > self.usl)).any(axis=1).astype(float)


class UnivariateCUSUM(SPCDetector):
    """Two-sided tabular CUSUM on each tag; alarm on the worst chart.

    k = 0.5 sigma is the standard choice for detecting a 1-sigma shift. The
    reported statistic is the maximum standardised CUSUM across tags, so the
    threshold has the usual "h in sigma units" interpretation.
    """

    name = "univariate_cusum"

    def __init__(self, ref: Reference, k: float = 0.5, threshold: float = 5.0):
        self.ref, self.k, self.threshold = ref, k, threshold

    def statistic(self, X: np.ndarray) -> np.ndarray:
        z = (X - self.ref.mu) / self.ref.sigma
        n, p = z.shape
        hi = np.zeros(p)
        lo = np.zeros(p)
        out = np.zeros(n)
        for t in range(n):
            hi = np.maximum(0.0, hi + z[t] - self.k)
            lo = np.maximum(0.0, lo - z[t] - self.k)
            out[t] = max(hi.max(), lo.max())
        return out


class HotellingT2(SPCDetector):
    """Single-point multivariate distance. Sees correlation, has no memory."""

    name = "hotelling_t2"

    def __init__(self, ref: Reference, threshold: float = 20.0):
        self.ref, self.threshold = ref, threshold

    def statistic(self, X: np.ndarray) -> np.ndarray:
        d = X - self.ref.mu
        return np.einsum("ij,jk,ik->i", d, self.ref.inv, d)


class MEWMA(SPCDetector):
    """Lowry, Woodall, Champ & Rigdon (1992)."""

    name = "mewma"

    def __init__(self, ref: Reference, lam: float = 0.2, threshold: float = 12.0):
        self.ref, self.lam, self.threshold = ref, lam, threshold

    def statistic(self, X: np.ndarray) -> np.ndarray:
        lam = self.lam
        z = np.zeros(X.shape[1])
        out = np.zeros(len(X))
        for t in range(len(X)):
            z = lam * (X[t] - self.ref.mu) + (1 - lam) * z
            scale = (lam / (2 - lam)) * (1 - (1 - lam) ** (2 * (t + 1)))
            inv = np.linalg.pinv(self.ref.cov * scale)
            out[t] = float(z @ inv @ z)
        return out


class CrosierMCUSUM(SPCDetector):
    """Crosier (1988) multivariate CUSUM.

    Accumulates the multivariate deviation and shrinks it back toward zero by
    k each step, so a sustained sub-sigma shift builds while independent noise
    cancels. This is the right shape of detector for a fixture that drifts
    slowly and then stays drifted.
    """

    name = "mcusum"

    def __init__(self, ref: Reference, k: float = 0.5, threshold: float = 5.0):
        self.ref, self.k, self.threshold = ref, k, threshold

    def statistic(self, X: np.ndarray) -> np.ndarray:
        s = np.zeros(X.shape[1])
        out = np.zeros(len(X))
        for t in range(len(X)):
            c = s + X[t] - self.ref.mu
            dist = float(np.sqrt(max(c @ self.ref.inv @ c, 0.0)))
            s = c * (1 - self.k / dist) if dist > self.k else np.zeros_like(c)
            out[t] = float(np.sqrt(max(s @ self.ref.inv @ s, 0.0)))
        return out


# --------------------------------------------------------------------------

def calibrate_threshold(
    detector_factory, X_incontrol: list[np.ndarray], target_arl0: float,
    lo: float = 0.5, hi: float = 200.0, iters: int = 30,
) -> float:
    """Bisect on the threshold until the in-control run length hits target.

    ARL0 is expressed in PARTS. At 60 units/hour a target of 1200 parts is one
    false alarm per 20 hours of production.

    This is the alert-budget principle from the Round 1 deck made concrete: we
    size thresholds to a false-alarm rate the floor will tolerate, and then
    report whatever detection speed that budget buys. We do not pick the
    threshold that maximises detection and report the false-alarm rate after.
    """
    def arl0(h: float) -> float:
        runs = []
        for X in X_incontrol:
            det = detector_factory(h)
            idx = det.alarm_indices(X)
            runs.append(len(X) / (len(idx) + 1) if len(idx) else float(len(X)) * 2)
        return float(np.mean(runs))

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if arl0(mid) < target_arl0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class T2CUSUM(SPCDetector):
    """CUSUM on the Hotelling T-squared statistic.

    Why this and not MCUSUM. MCUSUM and MEWMA are built to detect a shift in
    the MEAN vector. Our fixture does not shift its mean -- it loses the
    coupling between two tags. Their deviations stay centred on zero, so a
    mean-targeting accumulator sees signed excursions that cancel and gains
    nothing from memory.

    What does change is multivariate DISPERSION: observations sit further from
    the in-control centre in Mahalanobis distance, because the covariance they
    are being measured against no longer describes them. T-squared captures
    that per observation but has no memory, so at a strict alert budget it
    needs a large excursion. Accumulating T-squared above its in-control mean
    gives the memory, and the pair detects a structural change that no
    mean-based chart -- and no univariate chart of any kind -- can see.

        C_t = max(0, C_{t-1} + (T2_t - mean_T2_in_control) / p - k)
    """

    name = "t2_cusum"

    def __init__(self, ref: Reference, k: float = 0.5, threshold: float = 5.0):
        self.ref, self.k, self.threshold = ref, k, threshold
        self.p = len(ref.mu)

    def statistic(self, X: np.ndarray) -> np.ndarray:
        d = X - self.ref.mu
        t2 = np.einsum("ij,jk,ik->i", d, self.ref.inv, d)
        # standardise: in control E[T2] = p and Var[T2] = 2p for multivariate normal
        z = (t2 - self.p) / np.sqrt(2 * self.p)
        out = np.zeros(len(X))
        c = 0.0
        for i in range(len(X)):
            c = max(0.0, c + z[i] - self.k)
            out[i] = c
        return out
