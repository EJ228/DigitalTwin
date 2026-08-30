"""
GraphSAGE over the station topology graph.

Hamilton, Ying & Leskovec (2017), mean aggregator:

    h_v^k = ReLU( W_k . [ h_v^{k-1} ; mean_{u in N(v)} h_u^{k-1} ] )

Two layers, so each station sees two hops of context. Written in NumPy with
Adam, for the same reason as the forecaster: no framework, readable end to end,
runs anywhere.

WHAT IT IS ASKED TO DO
----------------------
Node classification: will this station be the momentary constraint in the next
ten minutes? Labels come from the bottleneck-walk detector's own output shifted
forward in time, so this is self-supervised on the event log -- no ground truth
is touched.

THE BASELINE IS THE POINT AGAIN
-------------------------------
The comparison is against an identical network with the aggregation switched
off: same features, same depth, same width, same optimiser, but each station
sees only itself. That isolates exactly one thing -- whether the graph structure
carries information -- rather than comparing a GNN against some unrelated model
and calling the difference "the value of graphs".

An assembly line is close to a path graph, so there is not much topology for a
GNN to exploit. If the aggregation buys nothing here, that is a real finding
about lines, not a bug, and it is reported either way.
"""

from __future__ import annotations

import numpy as np

from .forecast import Adam, sigmoid


def line_adjacency(n: int, radius: int = 2, self_loops: bool = False) -> np.ndarray:
    """Row-normalised neighbour matrix for a linear station sequence.

    Radius 2 means a station aggregates over the two stations either side.
    Blocking and starvation propagate locally, so a wider neighbourhood mostly
    adds noise -- and two GraphSAGE layers already give an effective receptive
    field of twice this.
    """
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(max(0, i - radius), min(n, i + radius + 1)):
            if i != j or self_loops:
                A[i, j] = 1.0
    rs = A.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return A / rs


class GraphSAGE:
    """Two-layer mean-aggregator GraphSAGE with a logistic head.

    aggregate=False disables the neighbour term entirely, reducing the model to
    a per-node MLP of identical capacity. That is the ablation.
    """

    def __init__(self, n_in: int, hidden: int = 24, out: int = 16,
                 aggregate: bool = True, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.aggregate = aggregate
        f1 = n_in * (2 if aggregate else 1)
        f2 = hidden * (2 if aggregate else 1)
        s1, s2 = np.sqrt(2.0 / f1), np.sqrt(2.0 / f2)
        self.p = {
            "W1": rng.normal(0, s1, (f1, hidden)), "b1": np.zeros(hidden),
            "W2": rng.normal(0, s2, (f2, out)), "b2": np.zeros(out),
            "w": rng.normal(0, np.sqrt(2.0 / out), (out, 1)), "bo": np.zeros(1),
        }
        self.opt = Adam(lr=5e-3)
        self.history: list[float] = []

    # ------------------------------------------------------------------

    def _layer(self, H, A, W, b):
        """H: (B, N, F). Concatenate self with the neighbour mean, then project."""
        if self.aggregate:
            Z = np.concatenate([H, np.einsum("ij,bjf->bif", A, H)], axis=2)
        else:
            Z = H
        pre = Z @ W + b
        return np.maximum(pre, 0.0), (Z, pre)

    def forward(self, X, A):
        H1, c1 = self._layer(X, A, self.p["W1"], self.p["b1"])
        H2, c2 = self._layer(H1, A, self.p["W2"], self.p["b2"])
        logit = (H2 @ self.p["w"] + self.p["bo"])[..., 0]
        return logit, (X, c1, H1, c2, H2)

    def backward(self, A, cache, logit, y, pos_weight: float):
        X, (Z1, pre1), H1, (Z2, pre2), H2 = cache
        B, N = logit.shape
        p = sigmoid(logit)
        # Weighted cross-entropy: constraint windows are a small minority, and
        # an unweighted loss simply predicts "not the constraint" everywhere.
        w = np.where(y > 0.5, pos_weight, 1.0)
        dlogit = w * (p - y) / (B * N)

        g = {}
        g["w"] = np.einsum("bnf,bn->f", H2, dlogit)[:, None]
        g["bo"] = np.array([dlogit.sum()])

        dH2 = dlogit[..., None] * self.p["w"][:, 0]
        dpre2 = dH2 * (pre2 > 0)
        g["W2"] = np.einsum("bnf,bng->fg", Z2, dpre2)
        g["b2"] = dpre2.sum(axis=(0, 1))
        dZ2 = dpre2 @ self.p["W2"].T
        if self.aggregate:
            h = H1.shape[2]
            dH1 = dZ2[:, :, :h] + np.einsum("ji,bjf->bif", A, dZ2[:, :, h:])
        else:
            dH1 = dZ2

        dpre1 = dH1 * (pre1 > 0)
        g["W1"] = np.einsum("bnf,bng->fg", Z1, dpre1)
        g["b1"] = dpre1.sum(axis=(0, 1))

        total = np.sqrt(sum(np.sum(v * v) for v in g.values()))
        if total > 5.0:
            for k in g:
                g[k] *= 5.0 / total
        return g

    # ------------------------------------------------------------------

    def fit(self, X, y, A, epochs: int = 60, batch: int = 32, verbose: bool = False):
        pos = max(float(y.mean()), 1e-6)
        pos_weight = (1 - pos) / pos
        rng = np.random.default_rng(0)
        for ep in range(epochs):
            idx = rng.permutation(len(X))
            tot = 0.0
            for s in range(0, len(X), batch):
                b = idx[s:s + batch]
                logit, cache = self.forward(X[b], A)
                p = np.clip(sigmoid(logit), 1e-7, 1 - 1e-7)
                w = np.where(y[b] > 0.5, pos_weight, 1.0)
                tot += float(-(w * (y[b] * np.log(p) + (1 - y[b]) * np.log(1 - p))).mean()) * len(b)
                self.opt.step(self.p, self.backward(A, cache, logit, y[b], pos_weight))
            self.history.append(tot / len(X))
            if verbose and (ep % 15 == 0 or ep == epochs - 1):
                print(f"  epoch {ep:3d}  loss {self.history[-1]:.4f}")
        return self

    def predict_proba(self, X, A):
        return sigmoid(self.forward(X, A)[0])

    def embed(self, X, A):
        """Final-layer station embeddings, for downstream models."""
        return self.forward(X, A)[1][4]
