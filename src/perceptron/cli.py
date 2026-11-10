"""Command-line interface for the perceptron."""

import argparse

from perceptron.data import make_clusters
from perceptron.model import Perceptron


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description="Train a perceptron on synthetic data."
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
    return parser


def main() -> None:
    """Generate data, train the perceptron and print accuracy."""
    parser = build_parser()
    args = parser.parse_args()

    if args.samples < 2:
        parser.error("--samples must be at least 2")
    if args.lr <= 0:
        parser.error("--lr must be positive")
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    X, y = make_clusters(args.samples, seed=0, separation=3.0)
    model = Perceptron(n_features=2, learning_rate=args.lr, epochs=args.epochs)
    errors = model.train(X, y)

    print(f"Trained for {len(errors)} epochs, errors in the last epoch: {errors[-1]}")
    print(f"Accuracy: {model.score(X, y):.3f}")
