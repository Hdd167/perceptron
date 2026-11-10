"""Synthetic data generation."""

import numpy as np


def make_clusters(
    n_samples: int, seed: int = 0, separation: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    """Generate two linearly separable Gaussian clouds."""
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    if separation <= 0:
        raise ValueError("separation must be positive")
    rng = np.random.default_rng(seed)
    half = n_samples // 2
    x0 = rng.normal(loc=-separation, scale=1.0, size=(half, 2))
    x1 = rng.normal(loc=+separation, scale=1.0, size=(half, 2))
    X = np.vstack([x0, x1])
    y = np.array([0] * half + [1] * half)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]
