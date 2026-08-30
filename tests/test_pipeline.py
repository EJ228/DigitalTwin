"""
pytest wrapper around the invariant suite and the anti-circularity audit.

    pytest -q

These are not unit tests of code paths. They assert that the SCENARIO is still
hard and that no prediction module can see ground truth. A config change that
makes the drift univariately detectable would leave every downstream number
meaningless while all the code still ran, so these run in CI, not by hand.
"""

import pathlib
import subprocess
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dtwin.audit import assert_no_fabricated_figures, assert_no_truth_leak
from dtwin.injectors import BottleneckSchedule
from dtwin.line_config import TAKT_SECONDS, build_line
from dtwin.simulator import AssemblyLineSim


def test_no_truth_leak():
    assert assert_no_truth_leak(ROOT / "dtwin") == []


def test_no_fabricated_frontend_figures():
    """A display component may not present a typed-in number as a measurement."""
    assert assert_no_fabricated_figures(ROOT / "web" / "src") == []


def test_line_is_balanced():
    stations = build_line()
    flat = BottleneckSchedule([], stations)
    loads = np.array([flat.theoretical_load(s.station_id, 0.0) for s in stations])
    assert (loads.max() - loads.min()) / loads.max() < 0.05
    assert loads.max() < TAKT_SECONDS


def test_sensor_coverage_envelope():
    tiers = [s.tier.value for s in build_line()]
    under = sum(t in ("sparse", "blind") for t in tiers) / len(tiers)
    assert 0.35 <= under <= 0.45


@pytest.mark.slow
def test_invariant_suite():
    r = subprocess.run([sys.executable, "scripts/test_invariants.py"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "all invariants hold" in r.stdout


def test_conformal_guarantee_holds():
    """Split conformal must bound the false-alarm rate on fresh in-control data."""
    import numpy as np

    from dtwin.conformal import ConformalThreshold

    rng = np.random.default_rng(0)
    ct = ConformalThreshold(0.05).fit(rng.normal(size=4000))
    emp = float(np.mean(ct.alarms(rng.normal(size=4000))))
    assert emp <= 0.05 * 1.5, f"empirical false-alarm rate {emp} exceeds alpha"


def test_hazard_features_exclude_downstream():
    """The structural constraint is the whole idea; assert it rather than trust it."""
    import pandas as pd

    from dtwin.hazard import UpstreamHazardModel

    m = UpstreamHazardModel("S08")
    assert m.upstream[-1] == "S08"
    assert "S09" not in m.upstream and "S35" not in m.upstream
    m.features = ["dwell_S07", "s08_gap_left_mm", "cum_wait"]
    assert m.leak_check()["clean"]
    m.features = ["dwell_S07", "s31_torque_Nm"]
    assert not m.leak_check()["clean"]


def test_lstm_gradients_match_finite_differences():
    """Hand-written BPTT must agree with numerical gradients."""
    import numpy as np

    from dtwin.forecast import LSTMForecaster

    rng = np.random.default_rng(0)
    m = LSTMForecaster(3, 6, 3, seed=1)
    X, Y = rng.normal(size=(4, 5, 3)), rng.normal(size=(4, 3))
    pred, cache = m._forward(X)
    grads = m._backward(X, pred, Y, cache)
    eps = 1e-5
    for k in ["W", "b", "Wy", "by"]:
        A = m.p[k]
        idx = tuple(rng.integers(0, s) for s in A.shape)
        o = A[idx]
        A[idx] = o + eps
        l1 = np.mean((m._forward(X)[0] - Y) ** 2)
        A[idx] = o - eps
        l2 = np.mean((m._forward(X)[0] - Y) ** 2)
        A[idx] = o
        num, ana = (l1 - l2) / (2 * eps), grads[k][idx]
        assert abs(num - ana) / max(abs(num) + abs(ana), 1e-9) < 1e-4, k


def test_graphsage_gradients_match_finite_differences():
    import numpy as np

    from dtwin.graphsage import GraphSAGE, line_adjacency

    rng = np.random.default_rng(0)
    A = line_adjacency(6, 2)
    m = GraphSAGE(4, 8, 6, aggregate=True, seed=1)
    X = rng.normal(size=(3, 6, 4))
    y = (rng.random((3, 6)) < 0.4).astype(float)
    logit, cache = m.forward(X, A)
    grads = m.backward(A, cache, logit, y, 1.5)

    def loss():
        l = m.forward(X, A)[0]
        p = np.clip(1 / (1 + np.exp(-l)), 1e-7, 1 - 1e-7)
        w = np.where(y > 0.5, 1.5, 1.0)
        return -(w * (y * np.log(p) + (1 - y) * np.log(1 - p))).mean()

    eps = 1e-5
    for k in ["W1", "b1", "W2", "b2", "w", "bo"]:
        P = m.p[k]
        idx = tuple(rng.integers(0, s) for s in P.shape)
        o = P[idx]
        P[idx] = o + eps
        l1 = loss()
        P[idx] = o - eps
        l2 = loss()
        P[idx] = o
        num, ana = (l1 - l2) / (2 * eps), grads[k][idx]
        assert abs(num - ana) / max(abs(num) + abs(ana), 1e-9) < 1e-4, k


@pytest.mark.slow
def test_paired_runs_share_flow():
    """Common random numbers: drift on/off must not change part movement."""
    on = AssemblyLineSim(horizon=8 * 3600.0, seed=7).run()["events"]
    off = AssemblyLineSim(horizon=8 * 3600.0, seed=7, enable_drift=False).run()["events"]
    a = on[on.event_type.isin(["enter", "exit"])].reset_index(drop=True)
    b = off[off.event_type.isin(["enter", "exit"])].reset_index(drop=True)
    assert a.equals(b)
