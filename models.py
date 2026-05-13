"""
models.py
PhotonicNN and ElectronicNN model classes.

Both share the same training loop so comparisons are fair.
"""

import numpy as np
from layers import PhotonicLayer, ElectronicLayer, ReadoutLayer


class PhotonicNN:
    """
    Photonic neural network.

    Architecture
    ------------
    Input → PhotonicLayer (MZI activation) → ReadoutLayer (sigmoid)

    The hidden layer computes in the optical domain.
    The readout layer models the electronic detector circuit.

    Parameters
    ----------
    layer_sizes  : list  e.g. [2, 6, 1]
    seed         : int
    sigma_noise  : float  Phase noise std (rad). 0 = ideal, 0.05 = realistic fab.
    """

    def __init__(self, layer_sizes: list, seed: int = 42, sigma_noise: float = 0.0):
        rng  = np.random.default_rng(seed)
        self._rng         = rng
        self.sigma_noise  = sigma_noise
        self.layer_sizes  = layer_sizes

        self.hidden = PhotonicLayer(layer_sizes[0], layer_sizes[1], rng)
        self.readout = ReadoutLayer(layer_sizes[1], layer_sizes[2], rng)

    def forward(self, X: np.ndarray, inject_noise: bool = False) -> np.ndarray:
        noise = self.sigma_noise if inject_noise else 0.0
        h = self.hidden.forward(X, phase_noise_std=noise, rng=self._rng)
        return self.readout.forward(h)

    def predict(self, X: np.ndarray) -> tuple:
        probs = self.forward(X, inject_noise=False)
        return (probs >= 0.5).astype(int), probs

    def train(self, X: np.ndarray, y: np.ndarray,
              epochs: int = 5000, lr: float = 0.1,
              noise_ramp_start: float = 0.5) -> list:
        losses = []
        n      = X.shape[0]
        noise_epoch = int(epochs * noise_ramp_start)

        for epoch in range(epochs):
            inject = (epoch >= noise_epoch) and (self.sigma_noise > 0)
            y_pred = self.forward(X, inject_noise=inject)

            loss = float(np.mean((y_pred - y) ** 2))
            losses.append(loss)

            d_loss = 2 * (y_pred - y) / n

            dW2, db2, dH = self.readout.backward(d_loss)
            dW1, db1, _  = self.hidden.backward(dH)

            self.readout.W -= lr * dW2
            self.readout.b -= lr * db2
            self.hidden.W  -= lr * dW1
            self.hidden.b  -= lr * db1

        return losses

    @property
    def n_params(self) -> int:
        return self.hidden.n_params + self.readout.n_params

    @property
    def n_phase_shifters(self) -> int:
        from hardware import count_phase_shifters
        return (
            count_phase_shifters(self.hidden.W) +
            count_phase_shifters(self.readout.W)
        )


class ElectronicNN:
    """
    Electronic neural network (matched architecture for fair comparison).

    Architecture
    ------------
    Input → ElectronicLayer (ReLU) → ReadoutLayer (sigmoid)

    Same parameter count as PhotonicNN for apples-to-apples benchmarking.
    """

    def __init__(self, layer_sizes: list, seed: int = 42):
        rng  = np.random.default_rng(seed)
        self.layer_sizes = layer_sizes

        self.hidden  = ElectronicLayer(layer_sizes[0], layer_sizes[1], rng)
        self.readout = ReadoutLayer(layer_sizes[1], layer_sizes[2], rng)

    def forward(self, X: np.ndarray, **kwargs) -> np.ndarray:
        h = self.hidden.forward(X)
        return self.readout.forward(h)

    def predict(self, X: np.ndarray) -> tuple:
        probs = self.forward(X)
        return (probs >= 0.5).astype(int), probs

    def train(self, X: np.ndarray, y: np.ndarray,
              epochs: int = 5000, lr: float = 0.1, **kwargs) -> list:
        losses = []
        n      = X.shape[0]

        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss   = float(np.mean((y_pred - y) ** 2))
            losses.append(loss)

            d_loss = 2 * (y_pred - y) / n
            dW2, db2, dH = self.readout.backward(d_loss)
            dW1, db1, _  = self.hidden.backward(dH)

            self.readout.W -= lr * dW2
            self.readout.b -= lr * db2
            self.hidden.W  -= lr * dW1
            self.hidden.b  -= lr * db1

        return losses

    @property
    def n_params(self) -> int:
        return self.hidden.n_params + self.readout.n_params
