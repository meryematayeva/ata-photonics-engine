"""
train.py
Shared training runner for PhotonicNN and ElectronicNN.

Usage
-----
    from train import train_and_evaluate
    results = train_and_evaluate(model, X_train, X_test, y_train, y_test,
                                 epochs=5000, lr=0.1)
"""

import time
import numpy as np


def train_and_evaluate(model, X_train, X_test, y_train, y_test,
                       epochs: int = 5000, lr: float = 0.1,
                       sigma_noise: float = 0.0) -> dict:
    """
    Train a model and return a results dict with all benchmark metrics.

    Parameters
    ----------
    model       : PhotonicNN or ElectronicNN instance (untrained)
    X_train, X_test, y_train, y_test : numpy arrays
    epochs      : int
    lr          : float
    sigma_noise : float  Only used for PhotonicNN; ignored otherwise.

    Returns
    -------
    dict with keys:
        losses, train_acc, test_acc, train_time_s,
        inference_time_ms, final_loss
    """
    # ── training ─────────────────────────────────────────────────────────────
    t0     = time.perf_counter()
    losses = model.train(X_train, y_train, epochs=epochs, lr=lr)
    train_time = time.perf_counter() - t0

    # ── accuracy ──────────────────────────────────────────────────────────────
    preds_tr, _ = model.predict(X_train)
    preds_te, _ = model.predict(X_test)
    train_acc   = float(np.mean(preds_tr == y_train))
    test_acc    = float(np.mean(preds_te == y_test))

    # ── inference latency (wall-clock, 100 runs) ──────────────────────────────
    n_runs = 100
    t_inf  = time.perf_counter()
    for _ in range(n_runs):
        model.predict(X_test)
    inf_time_ms = (time.perf_counter() - t_inf) / n_runs * 1000

    return {
        "losses":            losses,
        "final_loss":        losses[-1],
        "train_acc":         train_acc,
        "test_acc":          test_acc,
        "train_time_s":      round(train_time, 3),
        "inference_time_ms": round(inf_time_ms, 4),
    }
