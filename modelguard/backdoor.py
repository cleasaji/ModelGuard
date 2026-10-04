"""Backdoor (trojan) detection in the spirit of Neural Cleanse, for softmax models.

Idea: a backdoored class is reachable with a *tiny* universal input patch (the trigger). For every candidate target
class we reverse-engineer the smallest L1 perturbation that flips (almost) all clean inputs to that class. A class
whose required perturbation is a strong low outlier (MAD anomaly index > 2) is flagged as backdoored.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


def poison(X, y, target: int, frac: float, idx, value: float, seed: int = 0):
    """Plant a trigger: set features ``idx`` to ``value`` on a fraction of samples and relabel them ``target``."""
    rng = np.random.default_rng(seed)
    Xp, yp = X.copy(), y.copy()
    sel = rng.choice(len(X), int(frac * len(X)), replace=False)
    Xp[np.ix_(sel, list(idx))] = value
    yp[sel] = target
    return Xp, yp


@dataclass
class BackdoorReport:
    l1_norms: Dict[int, float]
    success: Dict[int, float]
    anomaly_index: Dict[int, float]
    flagged: List[int] = field(default_factory=list)
    trigger_features: Dict[int, List[int]] = field(default_factory=dict)


def _soft(x, t):
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)


def _min_norm_trigger(model, X, target: int, steps: int = 150, lr: float = 0.1, need: float = 0.9):
    """Smallest-L1 universal perturbation that sends >= ``need`` of the inputs to ``target``.

    Proximal gradient (ISTA) gives exactly-sparse solutions; bisection over the L1 weight finds the sparsest one
    that still succeeds.
    """
    yt = np.full(len(X), target)

    def solve(lam):
        d = np.zeros(X.shape[1])
        for _ in range(steps):
            d = _soft(d - lr * model.loss_grad_x(X + d, yt).mean(axis=0), lr * lam)
        return d

    best = solve(0.0)
    if (model.predict(X + best) == target).mean() < need:
        return best                                    # unreachable within budget: report as-is (large norm)
    lo, hi = 0.0, 2.0
    for _ in range(9):
        mid = (lo + hi) / 2
        d = solve(mid)
        if (model.predict(X + d) == target).mean() >= need:
            best, lo = d, mid
        else:
            hi = mid
    return best


def detect_backdoor(model, X_clean, threshold: float = 2.0, need: float = 0.9) -> BackdoorReport:
    ks = range(model.k)
    norms, succ, deltas = {}, {}, {}
    for t in ks:
        d = _min_norm_trigger(model, X_clean, t, need=need)
        norms[t] = float(np.abs(d).sum())
        succ[t] = float((model.predict(X_clean + d) == t).mean())
        deltas[t] = d
    vals = np.array([norms[t] for t in ks])
    med = np.median(vals)
    mad = 1.4826 * np.median(np.abs(vals - med)) or 1e-9
    idx = {t: float((med - norms[t]) / mad) for t in ks}
    flagged = [t for t in ks if idx[t] > threshold and succ[t] >= need]
    trig = {t: [int(i) for i in np.nonzero(np.abs(deltas[t]) > 0.25 * np.abs(deltas[t]).max())[0]] for t in flagged}
    return BackdoorReport(norms, succ, idx, flagged, trig)


def weight_outliers(model, z_threshold: float = 4.0) -> Dict[int, List[int]]:
    """Exact check for linear/softmax models: a planted trigger shows up as a few input features whose weight for the
    target class is far above what the other classes give them. Returns {class: [suspicious feature indices]}."""
    W = model.W
    out = {}
    for t in range(model.k):
        others = np.delete(W, t, axis=1).mean(axis=1)
        spread = W[:, t] - others
        med = np.median(spread)
        mad = 1.4826 * np.median(np.abs(spread - med)) or 1e-9
        feats = np.nonzero((spread - med) / mad > z_threshold)[0]
        if len(feats):
            out[t] = [int(f) for f in feats]
    return out
