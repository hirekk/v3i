"""Logistic regression baseline."""

import numpy as np
from sklearn.linear_model import LogisticRegression

from .base import BaselineModel


class LogisticRegressionBaseline(BaselineModel):
    """Logistic regression baseline."""

    def __init__(self, max_iter: int = 1000, random_seed: int = 42) -> None:
        self.model = LogisticRegression(
            random_state=random_seed,
            max_iter=max_iter,
            warm_start=True,  # Enable incremental fitting
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> float:
        """Fit on a batch of data and return accuracy."""
        if not self.is_fitted:
            self.model.fit(X, y)
            self.is_fitted = True
        else:
            self.model.fit(X, y)  # Will use warm_start
        return self.model.score(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for input data."""
        if not self.is_fitted:
            msg = "Model must be fitted before prediction"
            raise RuntimeError(msg)
        return self.model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy score on data."""
        if not self.is_fitted:
            msg = "Model must be fitted before scoring"
            raise RuntimeError(msg)
        return self.model.score(X, y)
