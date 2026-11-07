"""Perceptron for binary classification."""

import numpy as np


class Perceptron:
    """Binary classifier that learns weights by correcting its own mistakes."""

    def __init__(
        self, n_features: int, learning_rate: float = 0.1, epochs: int = 50
    ) -> None:
        """Create a perceptron with zero weights and zero bias."""
        self.w = np.zeros(n_features)
        self.bias: float = 0
        self.epochs = epochs
        self.lr = learning_rate

    def activate(self, z: float) -> int:
        """Return 1 if z >= 0, otherwise 0."""
        if z >= 0:
            return 1
        else:
            return 0

    def predict(self, x: np.ndarray) -> int:
        """Predict the class of a single sample."""
        z = np.dot(self.w, x) + self.bias
        return self.activate(z)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the share of correctly classified samples."""
        correct = 0
        for feature, target in zip(X, y):
            if self.predict(feature) == target:
                correct += 1
        return correct / len(y)

    def train(self, X: np.ndarray, y: np.ndarray) -> list[int]:
        """Train the model and return the number of errors per epoch."""
        errors_per_epoch: list[int] = []
        for _ in range(self.epochs):
            errors = 0
            for feature, target in zip(X, y):
                pred = self.predict(feature)
                error = target - pred
                if error != 0:
                    self.w += self.lr * error * feature
                    self.bias += self.lr * error
                    errors += 1
            errors_per_epoch.append(errors)
            if errors == 0:
                break
        return errors_per_epoch
