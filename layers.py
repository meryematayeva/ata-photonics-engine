"""
layers.py
Photonic and electronic neural network layers.

PhotonicLayer  : MZI phase-shifter → cos²(φ/2) activation
ElectronicLayer: standard dense layer → ReLU activation
ReadoutLayer   : electronic sigmoid output (shared by both model types)
"""

import numpy as np
from activations import mzi, d_mzi, relu, d_relu, sigmoid, d_sigmoid_from_output


class PhotonicLayer:
    """
    Dense photonic layer.

    Forward pass:
        phase = X @ W + b          # weighted phase shift
        output = cos²(phase / 2)   # MZI interference intensity

    Weights W represent programmable phase biases on each MZI arm.
    """

    def __init__(self, in_dim: int, out_dim: int, rng):
        self.W = rng.normal(0, 1.0, (in_dim, out_dim))
        self.b = rng.normal(0, 0.2, (1, out_dim))
        self._phase = None
        self._input = None

    def forward(self, X: np.ndarray, phase_noise_std: float = 0.0,
                rng=None) -> np.ndarray:
        self._input = X
        self._phase = X @ self.W + self.b
        if phase_noise_std > 0 and rng is not None:
            self._phase += rng.normal(0, phase_noise_std, self._phase.shape)
        return mzi(self._phase)

    def backward(self, d_out: np.ndarray) -> tuple:
        """Returns (dW, db, dX)."""
        d_phase = d_out * d_mzi(self._phase)
        dW = self._input.T @ d_phase
        db = np.sum(d_phase, axis=0, keepdims=True)
        dX = d_phase @ self.W.T
        return dW, db, dX

    @property
    def n_params(self) -> int:
        return self.W.size + self.b.size


class ElectronicLayer:
    """
    Standard dense layer with ReLU activation.
    """

    def __init__(self, in_dim: int, out_dim: int, rng):
        scale = np.sqrt(2.0 / in_dim)   # He initialisation
        self.W = rng.normal(0, scale, (in_dim, out_dim))
        self.b = np.zeros((1, out_dim))
        self._z    = None
        self._input = None

    def forward(self, X: np.ndarray, **kwargs) -> np.ndarray:
        self._input = X
        self._z     = X @ self.W + self.b
        return relu(self._z)

    def backward(self, d_out: np.ndarray) -> tuple:
        d_z = d_out * d_relu(self._z)
        dW  = self._input.T @ d_z
        db  = np.sum(d_z, axis=0, keepdims=True)
        dX  = d_z @ self.W.T
        return dW, db, dX

    @property
    def n_params(self) -> int:
        return self.W.size + self.b.size


class ReadoutLayer:
    """
    Electronic sigmoid readout — shared by photonic and electronic models.
    Represents the photodetector + transimpedance amplifier stage.
    """

    def __init__(self, in_dim: int, out_dim: int, rng):
        scale = np.sqrt(1.0 / in_dim)
        self.W = rng.normal(0, scale, (in_dim, out_dim))
        self.b = np.zeros((1, out_dim))
        self._input = None
        self._out   = None

    def forward(self, X: np.ndarray, **kwargs) -> np.ndarray:
        self._input = X
        self._out   = sigmoid(X @ self.W + self.b)
        return self._out

    def backward(self, d_loss: np.ndarray) -> tuple:
        d_z = d_loss * d_sigmoid_from_output(self._out)
        dW  = self._input.T @ d_z
        db  = np.sum(d_z, axis=0, keepdims=True)
        dX  = d_z @ self.W.T
        return dW, db, dX

    @property
    def n_params(self) -> int:
        return self.W.size + self.b.size
