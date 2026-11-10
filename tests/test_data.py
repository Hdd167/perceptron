"""Tests for data generation and loading."""

import numpy as np

from perceptron.data import load_iris_binary, make_clusters


def test_make_clusters_shape() -> None:
    """The requested number of samples is produced with two features."""
    X, y = make_clusters(21, seed=0)
    assert X.shape == (21, 2)
    assert y.shape == (21,)
    assert sorted(np.unique(y).tolist()) == [0, 1]


def test_make_clusters_same_seed() -> None:
    """The same seed produces the same data."""
    X1, y1 = make_clusters(40, seed=7)
    X2, y2 = make_clusters(40, seed=7)
    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)


def test_iris_slice_is_two_classes() -> None:
    """The iris slice has two features, two classes and 50 samples each."""
    X, y = load_iris_binary()
    assert X.shape == (100, 2)
    assert sorted(np.unique(y).tolist()) == [0, 1]
    assert int(y.sum()) == 50
