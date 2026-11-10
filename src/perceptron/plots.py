"""Matplotlib plots for the perceptron."""

import importlib.util
import os
import sys

import matplotlib

matplotlib.use("Agg")  # figures are also written to files, no display required

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from matplotlib.figure import Figure

from perceptron.model import Perceptron


def gui_available() -> bool:
    """Return True when a windowed backend has everything it needs."""
    if importlib.util.find_spec("tkinter") is None:
        return False
    if sys.platform in {"win32", "darwin"}:
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def use_gui_backend() -> None:
    """Switch to a windowed backend so figures can be shown on screen."""
    matplotlib.use("TkAgg", force=True)


def build_decision_boundary(model: Perceptron, X: np.ndarray, y: np.ndarray) -> Figure:
    """Build a scatter of both classes with the perceptron's decision boundary."""
    x_min, x_max = float(X[:, 0].min()) - 1.0, float(X[:, 0].max()) + 1.0
    y_min, y_max = float(X[:, 1].min()) - 1.0, float(X[:, 1].max()) + 1.0
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    labels = model.predict_batch(grid).reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, labels, alpha=0.25, levels=[-0.5, 0.5, 1.5])
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=12, label="class 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=12, label="class 1")
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    ax.set_title(f"decision boundary, accuracy {model.score(X, y):.3f}")
    ax.legend()
    return fig


def build_error_history(errors: list[int]) -> Figure:
    """Build the plot of mistakes made in every training epoch."""
    fig, ax = plt.subplots()
    ax.plot(range(1, len(errors) + 1), errors, marker="o")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mistakes")
    ax.set_title("errors per epoch")
    return fig


def save_figure(fig: Figure, path: str) -> None:
    """Write a figure to disk and close it."""
    fig.savefig(path, dpi=120)
    plt.close(fig)


def show_figures() -> None:
    """Open every current figure in a window; blocks until they are closed."""
    plt.show()
