"""Tests for the perceptron model."""

import numpy as np

from perceptron.data import make_clusters
from perceptron.model import Perceptron


def test_activate_threshold() -> None:
    """Step function returns 1 at zero and 0 below it."""
    p = Perceptron(n_features=2)
    assert p.activate(0) == 1
    assert p.activate(-1) == 0


def test_weights_length() -> None:
    """Weights have one entry per feature."""
    p = Perceptron(3)
    assert len(p.w) == 3


def test_predict_with_known_weights() -> None:
    """Predict follows the sign of w.x + bias."""
    p = Perceptron(n_features=2)
    p.w = np.array([1.0, 1.0])
    p.bias = -5.0
    assert p.predict(np.array([1.0, 2.0])) == 0
    assert p.predict(np.array([4.0, 4.0])) == 1


def test_train_separates_blobs() -> None:
    """Training on separable data reaches near-perfect accuracy."""
    X, y = make_clusters(200, seed=0, separation=3.0)
    p = Perceptron(n_features=2)
    p.train(X, y)
    assert p.score(X, y) > 0.95
