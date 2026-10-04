"""Adversarial robustness testing for differentiable classifiers (NumPy only).

``SoftmaxModel`` is a multinomial logistic regression with an analytic input gradient, so FGSM/PGD are exact.
Any object exposing ``predict(X)`` and ``loss_grad_x(X, y)`` can be evaluated with the same attacks.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


def make_dataset(n: int = 2000, d: int = 64, k: int = 5, noise: float = 1.5, seed: int = 0
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Gaussian class clusters in d dimensions (a stand-in for a tabular/embedding task)."""
    rng = np.random.default_rng(seed)
    means = rng.normal(0, 1.0, (k, d))
    y = rng.integers(0, k, n)
    return means[y] + rng.normal(0, noise, (n, d)), y


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class SoftmaxModel:
    def __init__(self, d: int, k: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W, self.b = rng.normal(0, 0.01, (d, k)), np.zeros(k)
        self.k = k

    def predict_proba(self, X):
        return _softmax(X @ self.W + self.b)

    def predict(self, X):
        return (X @ self.W + self.b).argmax(axis=1)

    def loss_grad_x(self, X, y):
        """Gradient of cross-entropy w.r.t. the *input* (per-sample)."""
        P = self.predict_proba(X)
        P[np.arange(len(y)), y] -= 1.0
        return P @ self.W.T

    def fit(self, X, y, epochs: int = 200, lr: float = 0.1, l2: float = 1e-3, extra_fn: Optional[Callable] = None):
        """Full-batch gradient descent. ``extra_fn(model, epoch)`` may return extra (X, y) to train on."""
        Y = np.eye(self.k)
        for ep in range(epochs):
            Xe, ye = X, y
            if extra_fn is not None:
                xa, ya = extra_fn(self, ep)
                Xe, ye = np.vstack([X, xa]), np.concatenate([y, ya])
            P = self.predict_proba(Xe)
            G = (P - Y[ye]) / len(ye)
            self.W -= lr * (Xe.T @ G + l2 * self.W)
            self.b -= lr * G.sum(axis=0)
        return self

    def accuracy(self, X, y) -> float:
        return float((self.predict(X) == y).mean())


def fgsm(model, X, y, eps: float):
    """Fast Gradient Sign Method: one step of size eps in the direction that increases the loss."""
    return X + eps * np.sign(model.loss_grad_x(X, y))


def pgd(model, X, y, eps: float, steps: int = 20, alpha: Optional[float] = None, seed: int = 0):
    """Projected Gradient Descent inside the L-infinity ball of radius eps (random start)."""
    alpha = alpha or 2.5 * eps / steps
    rng = np.random.default_rng(seed)
    Xa = X + rng.uniform(-eps, eps, X.shape)
    for _ in range(steps):
        Xa = Xa + alpha * np.sign(model.loss_grad_x(Xa, y))
        Xa = X + np.clip(Xa - X, -eps, eps)
    return Xa


def robust_accuracy(model, X, y, eps_list: Sequence[float], attack: str = "pgd") -> Dict[float, float]:
    atk = {"fgsm": fgsm, "pgd": pgd}[attack]
    return {e: float((model.predict(atk(model, X, y, e) if e else X) == y).mean()) for e in eps_list}


def adversarial_train(X, y, k: int, eps: float, epochs: int = 200, lr: float = 0.1, seed: int = 0) -> SoftmaxModel:
    """Madry-style adversarial training: every epoch, also fit on PGD examples crafted against the current model."""
    m = SoftmaxModel(X.shape[1], k, seed)
    return m.fit(X, y, epochs=epochs, lr=lr,
                 extra_fn=lambda mod, ep: (pgd(mod, X, y, eps, steps=5, seed=ep), y))
