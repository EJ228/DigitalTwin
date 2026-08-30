"""
Solve for the defect-physics constants instead of guessing them.

    python scripts/calibrate_defect_physics.py

The mis-set probability is logistic in the fixture squareness error:

    p(defect) = base + (1 - base) * sigmoid(k * (|gap_L - gap_R| - c))

`c` and `k` are not free artistic choices. They are solved so that:

  * the IN-CONTROL defect rate lands near 0.6%, anchored to the observed
    sub-1% rate in Bosch Production Line Performance, and
  * the FULL-DRIFT rate lands near 13% -- high enough to matter commercially,
    low enough that end-of-line inspection does not catch it on the first unit
    and the escape window is a real quantity.

Healthy:  gap_L, gap_R correlated at r = 0.86, so their difference has
          sd = sigma * sqrt(2(1-r)) and sits far inside tolerance.
Degraded: the fixture develops play, r collapses toward -0.10, the difference
          widens by ~2.8x while every marginal distribution is UNCHANGED.

Re-run this after changing sigma, either correlation, or the target rates, and
paste the result into dtwin/line_config.py.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dtwin.line_config import (
    BASE_DEFECT_RATE,
    GAP_DIFF_STEEPNESS,
    GAP_DIFF_TOLERANCE_MM,
    S08_CORRELATION,
    S08_CORRELATION_DEGRADED,
    _body_tags,
)

TARGET_IN_CONTROL = 0.006
TARGET_FULL_DRIFT = 0.13
N = 200_000
GAP_PAIR = ("s08_gap_left_mm", "s08_gap_right_mm")


def rate(sd: float, c: float, k: float, base: float, rng) -> float:
    d = rng.normal(0.0, sd, N)
    return float((base + (1 - base) / (1 + np.exp(-k * (np.abs(d) - c)))).mean())


def measure_spread():
    """Measure the gap-difference spread from an actual run.

    It cannot be derived from the S08 correlation alone any more: shared latents
    (incoming batch quality, ambient conditions) contribute to both gaps, so the
    effective coupling -- and therefore the spread of their difference -- is a
    property of the whole line, not of one station's correlation matrix.
    """
    from dtwin.simulator import AssemblyLineSim

    f = AssemblyLineSim(horizon=3 * 8 * 3600.0, seed=7).run()
    tg = f["tags"]
    w = tg.pivot_table(index="part_id", columns="tag", values="value", aggfunc="first")
    tm = tg.groupby("part_id").t.first()
    onset = float(f["truth_drift"].t_onset.iloc[0])
    diff = (w[GAP_PAIR[0]] - w[GAP_PAIR[1]])
    return float(diff[tm < onset].std()), float(diff[tm > onset + 1800].std())


def main():
    rng = np.random.default_rng(0)
    sigma = {t.name: t.sigma for t in _body_tags("s08")}[GAP_PAIR[0]]
    print("measuring gap-difference spread from a full run...")
    sd_ok, sd_bad = measure_spread()

    print(f"tag sigma                  {sigma:.4f} mm")
    print(f"gap-difference sd healthy  {sd_ok:.4f} mm  (nominal r = {S08_CORRELATION[GAP_PAIR]:+.2f})")
    print(f"gap-difference sd degraded {sd_bad:.4f} mm  (nominal r = {S08_CORRELATION_DEGRADED[GAP_PAIR]:+.2f})")
    print(f"widening factor            {sd_bad/sd_ok:.2f}x\n")

    best = None
    for c in np.arange(0.030, 0.150, 0.002):
        for k in (60, 90, 120, 160, 220, 300, 400):
            a = rate(sd_ok, c, k, BASE_DEFECT_RATE, rng)
            b = rate(sd_bad, c, k, BASE_DEFECT_RATE, rng)
            err = (abs(a - TARGET_IN_CONTROL) / TARGET_IN_CONTROL
                   + abs(b - TARGET_FULL_DRIFT) / TARGET_FULL_DRIFT)
            if best is None or err < best[0]:
                best = (err, c, k, a, b)

    _, c, k, a, b = best
    print(f"SOLVED   GAP_DIFF_TOLERANCE_MM = {c:.4f}")
    print(f"         GAP_DIFF_STEEPNESS    = {float(k):.1f}")
    print(f"  -> in-control {100*a:.2f}%   full drift {100*b:.1f}%")

    ca, cb = (rate(sd_ok, GAP_DIFF_TOLERANCE_MM, GAP_DIFF_STEEPNESS, BASE_DEFECT_RATE, rng),
              rate(sd_bad, GAP_DIFF_TOLERANCE_MM, GAP_DIFF_STEEPNESS, BASE_DEFECT_RATE, rng))
    print(f"\nCURRENT  tolerance = {GAP_DIFF_TOLERANCE_MM:.4f}  steepness = {GAP_DIFF_STEEPNESS:.1f}")
    print(f"  -> in-control {100*ca:.2f}%   full drift {100*cb:.1f}%")

    drift = abs(ca - a) / max(a, 1e-9)
    print("\nconfig is consistent with the solve" if drift < 0.35
          else "\nWARNING: config has drifted from the solved values -- update line_config.py")


if __name__ == "__main__":
    main()
