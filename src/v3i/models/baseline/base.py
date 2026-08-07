"""Base class for baseline models."""

from typing import Protocol

import numpy as np


class BaselineModel(Protocol):
    """Protocol for baseline models."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> float:
        """Fit model on data and return training accuracy."""
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for input data."""
        ...

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy score on data."""
        ...
