"""
activations.py
Photonic and electronic activation functions.

Photonic:
  Mach-Zehnder Interferometer (MZI) intensity response: cos²(φ/2)

Electronic:
  sigmoid, relu, tanh — standard reference implementations.
"""

import numpy as np


# ── Photonic ──────────────────────────────────────────────────────────────────

def mzi(phase: np.ndarray) -> np.ndarray:
    """
    Ideal MZI intensity transfer function.
    I = cos²(phase / 2)
    Maps phase in radians → optical intensity in [0, 1].
    """
    return np.cos(phase / 2) ** 2


def d_mzi(phase: np.ndarray) -> np.ndarray:
    """Derivative of mzi w.r.t. phase: -½ sin(phase)."""
    return -0.5 * np.sin(phase)


# ── Electronic ────────────────────────────────────────────────────────────────

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def d_sigmoid_from_output(y: np.ndarray) -> np.ndarray:
    return y * (1.0 - y)


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def d_relu(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(float)


def tanh_act(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


def d_tanh_from_output(y: np.ndarray) -> np.ndarray:
    return 1.0 - y ** 2
