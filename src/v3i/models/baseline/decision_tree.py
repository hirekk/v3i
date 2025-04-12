"""Decision tree baseline."""

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from .base import BaselineModel


class DecisionTreeBaseline(BaselineModel):
    """Decision tree model."""

    def __init__(self, max_depth: int = 1, max_leaf_nodes: int = 4, random_seed: int = 42) -> None:
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_seed,
        )
        self.is_fitted = False

    def fit_batch(self, X: np.ndarray, y: np.ndarray) -> float:
        """Fit on a batch of data and return accuracy."""
        self.model.fit(X, y)
        self.is_fitted = True
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
