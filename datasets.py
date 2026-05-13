"""
datasets.py
Standard datasets for photonic vs electronic benchmarking.

Datasets
--------
xor         : 4-point binary classification
two_moons   : sklearn-style non-linearly separable 2D dataset
mnist_01    : MNIST digits 0 vs 1, flattened to 784-dim, PCA-reduced
"""

import numpy as np


def xor() -> tuple:
    """
    XOR classification.
    Returns X (4, 2), y (4, 1), name.
    """
    X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y = np.array([[0.], [1.], [1.], [0.]])
    return X, y, "XOR"


def two_moons(n_samples: int = 400, noise: float = 0.15, seed: int = 42) -> tuple:
    """
    Two interleaving half-circles — a classic non-linear 2D benchmark.
    Returns X (n, 2) normalised to [0,1], y (n, 1), name.
    """
    rng = np.random.default_rng(seed)
    n_each = n_samples // 2

    theta_top = np.linspace(0, np.pi, n_each)
    theta_bot = np.linspace(0, np.pi, n_each)

    X_top = np.column_stack([np.cos(theta_top), np.sin(theta_top)])
    X_bot = np.column_stack([1 - np.cos(theta_bot), 1 - np.sin(theta_bot) - 0.5])

    X = np.vstack([X_top, X_bot])
    X += rng.normal(0, noise, X.shape)

    # normalise to roughly [0,1]
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)

    y = np.array([0] * n_each + [1] * n_each, dtype=float).reshape(-1, 1)

    idx = rng.permutation(len(X))
    return X[idx], y[idx], "Two-Moons"


def mnist_01(n_components: int = 16, n_samples: int = 500, seed: int = 42) -> tuple:
    """
    MNIST digits 0 vs 1, PCA-reduced to n_components dimensions.

    Tries sklearn.datasets.fetch_openml first; falls back to a
    deterministic synthetic proxy if unavailable (offline environments).

    Returns X (n, n_components) normalised, y (n, 1), name.
    """
    try:
        from sklearn.datasets import fetch_openml
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        X_raw, y_raw = mnist.data, mnist.target.astype(int)

        mask = (y_raw == 0) | (y_raw == 1)
        X_raw, y_raw = X_raw[mask], y_raw[mask]

        rng = np.random.default_rng(seed)
        idx  = rng.choice(len(X_raw), size=min(n_samples, len(X_raw)), replace=False)
        X_raw, y_raw = X_raw[idx], y_raw[idx]

        X_sc  = StandardScaler().fit_transform(X_raw)
        X_pca = PCA(n_components=n_components, random_state=seed).fit_transform(X_sc)

        X_norm = (X_pca - X_pca.min(axis=0)) / (
            X_pca.max(axis=0) - X_pca.min(axis=0) + 1e-8
        )
        y = (y_raw == 1).astype(float).reshape(-1, 1)
        return X_norm, y, f"MNIST 0v1 (PCA-{n_components})"

    except Exception as e:
        print(f"  [datasets] MNIST unavailable ({e}). Using synthetic proxy.")
        return _synthetic_mnist_proxy(n_components, n_samples, seed)


def _synthetic_mnist_proxy(n_components: int, n_samples: int, seed: int) -> tuple:
    """
    Linearly-separable Gaussian proxy when MNIST cannot be fetched.
    Cluster 0 at origin, cluster 1 offset by 2 in the first two dims.
    """
    rng   = np.random.default_rng(seed)
    half  = n_samples // 2
    X0    = rng.normal(0, 1, (half, n_components))
    X1    = rng.normal(0, 1, (half, n_components))
    X1[:, :2] += 2.0
    X     = np.vstack([X0, X1])
    y     = np.array([0]*half + [1]*half, dtype=float).reshape(-1, 1)
    X_n   = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)
    idx   = rng.permutation(len(X))
    return X_n[idx], y[idx], f"MNIST-proxy (PCA-{n_components})"


def train_test_split(X: np.ndarray, y: np.ndarray,
                     test_ratio: float = 0.2,
                     seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    cut = int(len(X) * (1 - test_ratio))
    tr, te = idx[:cut], idx[cut:]
    return X[tr], X[te], y[tr], y[te]
