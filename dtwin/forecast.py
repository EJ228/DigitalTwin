"""
LSTM buffer-trajectory forecaster.

Everything else in the flow engine is nowcasting: it tells you where the
constraint is right now. This is the piece that answers "where does it move
next" -- it forecasts buffer fill ratios across all 35 stations H steps ahead,
so the active-period detector can be run on the predicted line state rather than
the observed one.

WHY IT IS WRITTEN OUT BY HAND
-----------------------------
No deep-learning framework. A single-layer LSTM with backpropagation through
time and Adam is about two hundred lines of NumPy, and writing it means the
project has no heavyweight dependency, runs anywhere, and can be read end to end
by a reviewer who wants to check there is no leakage hiding in a framework call.

THE BASELINE IS THE POINT
-------------------------
Buffer levels are strongly autocorrelated, so persistence -- "it will be what it
is now" -- is a genuinely hard baseline over short horizons, not a straw man. We
report the LSTM against persistence and against linear extrapolation, and we
report the comparison whichever way it comes out. A forecaster that cannot beat
persistence is worth knowing about.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


@dataclass
class Adam:
    lr: float = 3e-3
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    t: int = 0
    m: dict = field(default_factory=dict)
    v: dict = field(default_factory=dict)

    def step(self, params: dict, grads: dict):
        self.t += 1
        for k, g in grads.items():
            if k not in self.m:
                self.m[k] = np.zeros_like(g)
                self.v[k] = np.zeros_like(g)
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


class LSTMForecaster:
    """Single-layer LSTM, sequence in -> vector out at a fixed horizon.

    Gate order in the packed matrices is [input, forget, output, candidate].
    The forget-gate bias is initialised to 1.0, which is the standard trick to
    stop the cell state decaying to nothing before the network has learned
    anything -- without it a short-sequence model like this trains very slowly.
    """

    def __init__(self, n_in: int, n_hidden: int = 48, n_out: int | None = None,
                 seed: int = 0):
        rng = np.random.default_rng(seed)
        self.n_in, self.n_h = n_in, n_hidden
        self.n_out = n_out or n_in
        s = 1.0 / np.sqrt(n_hidden)
        self.p = {
            "W": rng.uniform(-s, s, (n_in + n_hidden, 4 * n_hidden)),
            "b": np.zeros(4 * n_hidden),
            "Wy": rng.uniform(-s, s, (n_hidden, self.n_out)),
            "by": np.zeros(self.n_out),
        }
        self.p["b"][n_hidden:2 * n_hidden] = 1.0      # forget-gate bias
        self.opt = Adam()
        self.history: list[float] = []

    # ------------------------------------------------------------------

    def _forward(self, X):
        """X: (B, T, n_in). Returns prediction and the cache needed for BPTT."""
        B, T, _ = X.shape
        H = self.n_h
        h = np.zeros((B, H))
        c = np.zeros((B, H))
        cache = []
        for t in range(T):
            z = np.concatenate([X[:, t, :], h], axis=1)
            g = z @ self.p["W"] + self.p["b"]
            i = sigmoid(g[:, :H])
            f = sigmoid(g[:, H:2 * H])
            o = sigmoid(g[:, 2 * H:3 * H])
            u = np.tanh(g[:, 3 * H:])
            c_new = f * c + i * u
            tc = np.tanh(c_new)
            h_new = o * tc
            cache.append((z, i, f, o, u, c, c_new, tc, h))
            h, c = h_new, c_new
        y = h @ self.p["Wy"] + self.p["by"]
        return y, (cache, h)

    def _backward(self, X, y, target, cache_pack):
        cache, h_last = cache_pack
        B, T, _ = X.shape
        H = self.n_h
        # dL/dy for mean squared error taken over BOTH batch and output dims,
        # which is what np.mean((pred - target) ** 2) computes.
        d = 2.0 * (y - target) / (B * self.n_out)

        grads = {k: np.zeros_like(v) for k, v in self.p.items()}
        grads["Wy"] = h_last.T @ d
        grads["by"] = d.sum(axis=0)

        dh = d @ self.p["Wy"].T
        dc = np.zeros((B, H))
        for t in reversed(range(T)):
            z, i, f, o, u, c_prev, c_new, tc, _ = cache[t]
            do = dh * tc
            dc = dc + dh * o * (1 - tc ** 2)
            di = dc * u
            du = dc * i
            df = dc * c_prev
            dc_prev = dc * f

            gi = di * i * (1 - i)
            gf = df * f * (1 - f)
            go = do * o * (1 - o)
            gu = du * (1 - u ** 2)
            dg = np.concatenate([gi, gf, go, gu], axis=1)

            grads["W"] += z.T @ dg
            grads["b"] += dg.sum(axis=0)
            dz = dg @ self.p["W"].T
            dh = dz[:, self.n_in:]
            dc = dc_prev

        # Clip by global norm. Exploding gradients through time are the classic
        # failure mode here and clipping is cheaper than diagnosing NaNs later.
        total = np.sqrt(sum(np.sum(g * g) for g in grads.values()))
        if total > 5.0:
            for k in grads:
                grads[k] *= 5.0 / total
        return grads

    # ------------------------------------------------------------------

    def fit(self, X, Y, epochs: int = 30, batch: int = 64, verbose: bool = False):
        n = len(X)
        rng = np.random.default_rng(0)
        for ep in range(epochs):
            idx = rng.permutation(n)
            tot = 0.0
            for s in range(0, n, batch):
                b = idx[s:s + batch]
                xb, yb = X[b], Y[b]
                pred, cache = self._forward(xb)
                tot += float(np.mean((pred - yb) ** 2)) * len(b)
                self.opt.step(self.p, self._backward(xb, pred, yb, cache))
            self.history.append(tot / n)
            if verbose and (ep % 5 == 0 or ep == epochs - 1):
                print(f"  epoch {ep:3d}  mse {self.history[-1]:.5f}")
        return self

    def predict(self, X):
        return self._forward(X)[0]


# ---------------------------------------------------------------------------
# Windowing and baselines
# ---------------------------------------------------------------------------

def make_windows(series: np.ndarray, lookback: int, horizon: int):
    """series: (T, n_features) -> X (N, lookback, n_features), Y (N, n_features)."""
    T = len(series)
    n = T - lookback - horizon
    if n <= 0:
        return np.empty((0, lookback, series.shape[1])), np.empty((0, series.shape[1]))
    X = np.stack([series[i:i + lookback] for i in range(n)])
    Y = np.stack([series[i + lookback + horizon - 1] for i in range(n)])
    return X, Y


def persistence(X):
    """It will be what it is now. Hard to beat over short horizons."""
    return X[:, -1, :]


def linear_extrapolation(X, horizon: int):
    """Fit a slope over the lookback window and project it forward."""
    T = X.shape[1]
    t = np.arange(T)
    tm = t.mean()
    denom = ((t - tm) ** 2).sum()
    slope = np.einsum("t,btf->bf", t - tm, X - X.mean(axis=1, keepdims=True)) / denom
    return X[:, -1, :] + slope * horizon


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def skill(model_rmse: float, baseline_rmse: float) -> float:
    """1 - model/baseline. Positive means the model earned its place."""
    return float(1 - model_rmse / baseline_rmse) if baseline_rmse > 0 else float("nan")
