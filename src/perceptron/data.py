"""Synthetic data generation."""

import numpy as np


def make_clusters(
    n_samples: int, seed: int = 0, separation: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    """Generate two linearly separable Gaussian clouds of exactly n_samples points."""
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    if separation <= 0:
        raise ValueError("separation must be positive")
    rng = np.random.default_rng(seed)
    second = n_samples // 2
    first = n_samples - second
    x0 = rng.normal(loc=-separation, scale=1.0, size=(first, 2))
    x1 = rng.normal(loc=+separation, scale=1.0, size=(second, 2))
    X = np.vstack([x0, x1])
    y = np.array([0] * first + [1] * second)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def load_iris_binary() -> tuple[np.ndarray, np.ndarray]:
    """Load a two-class, two-feature slice of iris: setosa vs versicolor."""
    from sklearn.datasets import load_iris

    iris = load_iris()
    X = iris.data[:100, 2:4]
    y = iris.target[:100]
    return X, y.astype(int)
