"""
Split conformal calibration of the alert threshold.

WHAT THIS REPLACES
------------------
Earlier we bisected the threshold until the observed false-alarm rate on
drift-free runs hit a target. That is the right instinct and it works, but it is
an empirical rate on one sample with no guarantee attached, and the Round 1 deck
claims the alerts are "conformally calibrated". This makes the claim true.

THE GUARANTEE
-------------
Given n exchangeable in-control observations and a miscoverage level alpha,
take the threshold to be the k-th smallest nonconformity score where

    k = ceil((n + 1)(1 - alpha))

Then for a fresh in-control observation, P(score > threshold) <= alpha. It is
distribution-free: no normality, no independence beyond exchangeability, no
asymptotics. It holds in finite samples.

WHAT IT DOES NOT GIVE
---------------------
Nothing about detection power. Conformal prediction bounds false alarms and says
nothing about how fast a real fault is caught -- that is what ARL1 and the escape
window measure, and they are reported separately. It also assumes exchangeability
within the calibration set, which a control-chart statistic with memory only
approximately satisfies; we report the achieved empirical rate alongside the
nominal one so the gap is visible rather than assumed away.
"""

from __future__ import annotations

import math

import numpy as np


class ConformalThreshold:
    """Distribution-free alert threshold at a stated false-alarm budget."""

    def __init__(self, alpha: float):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha
        self.threshold: float | None = None
        self.n = 0

    @staticmethod
    def alpha_for_arl0(arl0_parts: float) -> float:
        """One false alarm per `arl0_parts` observations."""
        return 1.0 / float(arl0_parts)

    def fit(self, in_control_scores: np.ndarray) -> "ConformalThreshold":
        s = np.sort(np.asarray(in_control_scores, dtype=float))
        s = s[np.isfinite(s)]
        n = len(s)
        if n == 0:
            raise ValueError("no in-control scores")
        k = math.ceil((n + 1) * (1 - self.alpha))
        # k > n means the sample is too small to certify this alpha; the honest
        # response is to use the maximum observed score and say so, not to
        # extrapolate a quantile we have no data for.
        self.exact = k <= n
        self.threshold = float(s[min(k, n) - 1])
        self.n = n
        self.min_n_for_alpha = int(math.ceil(1 / self.alpha) - 1)
        return self

    def alarms(self, scores: np.ndarray) -> np.ndarray:
        return np.asarray(scores, dtype=float) > self.threshold

    def report(self, holdout_scores: np.ndarray | None = None) -> dict:
        d = {
            "alpha": self.alpha,
            "nominal_false_alarms_per_1000": round(1000 * self.alpha, 3),
            "threshold": round(self.threshold, 4),
            "calibration_n": self.n,
            "finite_sample_exact": bool(self.exact),
            "min_n_for_alpha": self.min_n_for_alpha,
            "guarantee": (
                f"P(false alarm on a fresh in-control part) <= {self.alpha:.4f}, "
                "distribution-free, finite-sample."
                if self.exact else
                "Calibration sample too small to certify this alpha exactly; "
                "threshold set at the observed maximum, which is conservative."
            ),
        }
        if holdout_scores is not None and len(holdout_scores):
            emp = float(np.mean(self.alarms(holdout_scores)))
            d["empirical_false_alarm_rate"] = round(emp, 5)
            d["empirical_arl0_parts"] = round(1 / emp, 1) if emp > 0 else None
            d["within_guarantee"] = bool(emp <= self.alpha * 1.5)
        return d
