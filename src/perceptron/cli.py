"""Command-line interface for the perceptron."""

import argparse
from pathlib import Path

import numpy as np

from perceptron.data import load_iris_binary, make_clusters
from perceptron.model import Perceptron
from perceptron.plots import (
    build_decision_boundary,
    build_error_history,
    gui_available,
    save_figure,
    show_figures,
    use_gui_backend,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description="Train a perceptron on synthetic data or on iris."
    )
    parser.add_argument(
        "--dataset",
        choices=("clusters", "iris"),
        default="clusters",
        help="data source: synthetic clusters or the iris flowers",
    )
    parser.add_argument(
        "--samples", type=int, default=200, help="number of samples (at least 2)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.1, help="learning rate (positive)"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="number of epochs (at least 1)"
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--figures",
        metavar="DIR",
        help="save the decision boundary and error plots into DIR",
    )
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="open the plots in a window (default: on, use --no-show to disable)",
    )
    return parser


def split_train_test(
    X: np.ndarray, y: np.ndarray, seed: int, test_share: float = 0.2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle the data and split it into train and test parts."""
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(y))
    cut = int(len(y) * (1.0 - test_share))
    train_idx, test_idx = shuffled[:cut], shuffled[cut:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def main() -> None:
    """Generate data, train the perceptron, print accuracy and draw the plots."""
    parser = build_parser()
    args = parser.parse_args()

    if args.samples < 2:
        parser.error("--samples must be at least 2")
    if args.lr <= 0:
        parser.error("--lr must be positive")
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    show = bool(args.show)
    if show and not gui_available():
        print("No display for the plot windows; use --figures DIR to save them.")
        show = False
    if show:
        use_gui_backend()

    if args.dataset == "iris":
        X, y = load_iris_binary()
    else:
        X, y = make_clusters(args.samples, seed=args.seed, separation=3.0)

    X_train, y_train, X_test, y_test = split_train_test(X, y, seed=args.seed)

    model = Perceptron(n_features=X.shape[1], learning_rate=args.lr, epochs=args.epochs)
    errors = model.train(X_train, y_train)

    print(f"Dataset: {args.dataset}, samples: {len(y)}")
    print(f"Trained for {len(errors)} epochs, errors in the last epoch: {errors[-1]}")
    print(f"Accuracy train: {model.score(X_train, y_train):.3f}")
    print(f"Accuracy test:  {model.score(X_test, y_test):.3f}")

    if args.figures or show:
        boundary = build_decision_boundary(model, X, y)
        history = build_error_history(errors)
        if args.figures:
            output = Path(args.figures)
            output.mkdir(parents=True, exist_ok=True)
            save_figure(boundary, str(output / f"{args.dataset}_boundary.png"))
            save_figure(history, str(output / f"{args.dataset}_errors.png"))
            print(f"Figures saved to {output}")
        if show:
            print("Close the plot windows to finish.")
            show_figures()


if __name__ == "__main__":
    main()
