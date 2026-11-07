"""Perceptron for binary classification"""

import numpy as np


class Perceptron:
    def __init__(
        self, n_features: int, learning_rate: float = 0.1, epochs: int = 50
    ) -> None:
        self.w = np.zeros(n_features)
        self.bias: float = 0
        self.epochs = epochs
        self.lr = learning_rate

    def activate(self, z: float) -> int:
        """Return 1 if z >= 0, otherwise 0"""
        if z >= 0:
            return 1
        else:
            return 0

    def predict(self, x: np.ndarray) -> int:
        z = np.dot(self.w, x) + self.bias
        return self.activate(z)

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        correct = 0
        for feacher, target in zip(x, y):
            if self.predict(feacher) == target:
                correct += 1
        return correct / len(y)
