"""Tests for the matplotlib plots."""

from pathlib import Path

from perceptron.data import make_clusters
from perceptron.model import Perceptron
from perceptron.plots import build_decision_boundary, build_error_history, save_figure


def test_figures_are_written(tmp_path: Path) -> None:
    """Both plots are built and saved to disk."""
    X, y = make_clusters(50, seed=0)
    model = Perceptron(n_features=2)
    errors = model.train(X, y)
    boundary = tmp_path / "boundary.png"
    history = tmp_path / "errors.png"

    save_figure(build_decision_boundary(model, X, y), str(boundary))
    save_figure(build_error_history(errors), str(history))

    assert boundary.stat().st_size > 0
    assert history.stat().st_size > 0
