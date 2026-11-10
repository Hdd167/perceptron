"""Perceptron for binary classification."""

import numpy as np


class Perceptron:
    """Binary classifier that learns weights by correcting its own mistakes."""

    def __init__(
        self, n_features: int, learning_rate: float = 0.1, epochs: int = 50
    ) -> None:
        """Create a perceptron with zero weights and zero bias."""
        if n_features < 1:
            raise ValueError("n_features must be at least 1")
        if learning_rate <= 0:
            raise ValueError("learning rate must be positive")
        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        self.w = np.zeros(n_features)
        self.bias: float = 0.0
        self.epochs = epochs
        self.lr = learning_rate

    @staticmethod
    def activate(z: float) -> int:
        """Return 1 if z >= 0, otherwise 0."""
        if z >= 0:
            return 1
        else:
            return 0

    def predict(self, x: np.ndarray) -> int:
        """Predict the class of a single sample."""
        if x.shape != self.w.shape:
            raise ValueError(f"sample must have shape {self.w.shape}, got {x.shape}")
        z = np.dot(self.w, x) + self.bias
        return self.activate(z)

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Predict the class of many samples at once."""
        if X.ndim != 2 or X.shape[1] != len(self.w):
            raise ValueError(
                f"X must be a 2D array with {len(self.w)} columns, got shape {X.shape}"
            )
        z = X @ self.w + self.bias
        return (z >= 0).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the share of correctly classified samples."""
        self.check_data(X, y)
        return float(np.mean(self.predict_batch(X) == y))

    def train(self, X: np.ndarray, y: np.ndarray) -> list[int]:
        """Train the model and return the number of errors per epoch."""
        self.check_data(X, y)
        errors_per_epoch: list[int] = []
        for _ in range(self.epochs):
            errors_counter = 0
            for feature, target in zip(X, y):
                pred = self.predict(feature)
                error = target - pred
                if error != 0:
                    self.w += self.lr * error * feature
                    self.bias += self.lr * error
                    errors_counter += 1
            errors_per_epoch.append(errors_counter)
            if errors_counter == 0:
                break
        return errors_per_epoch

    def check_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """Raise ValueError if X and y don't fit this perceptron."""
        if X.ndim != 2 or X.shape[1] != len(self.w):
            raise ValueError(
                f"X must be a 2D array with {len(self.w)} columns, got shape {X.shape}"
            )
        if len(X) == 0:
            raise ValueError("X must contain at least one sample")
        if y.shape != (len(X),):
            raise ValueError(f"y must have shape ({len(X)},), got {y.shape}")
