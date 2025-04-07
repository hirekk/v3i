"""Random baseline that guesses labels randomly."""

import numpy as np

from .base import BaselineModel


class RandomChoiceBaseline(BaselineModel):
    """Random baseline that guesses labels randomly."""

    def __init__(self, random_seed: int = 42) -> None:
        self.rng = np.random.default_rng(random_seed)
        self.classes = [-1, 1]  # Even/odd labels

    def fit_batch(self, X: np.ndarray, y: np.ndarray) -> float:
        """No fitting needed, return random accuracy."""
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict random labels."""
        return self.rng.choice(self.classes, size=len(X))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy of random predictions."""
        predictions = self.predict(X)
        return np.mean(predictions == y)
