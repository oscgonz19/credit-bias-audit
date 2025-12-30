"""Model wrappers for credit risk classification."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV


class CreditRiskModel:
    """Wrapper for credit risk classification models."""

    SUPPORTED_MODELS = ["logreg", "logreg_cv", "rf"]

    def __init__(
        self,
        model_type: str = "logreg",
        random_state: int = 42,
        **kwargs,
    ):
        """Initialize credit risk model.

        Args:
            model_type: Type of model ('logreg', 'logreg_cv', 'rf').
            random_state: Random seed for reproducibility.
            **kwargs: Additional model-specific parameters.
        """
        self.model_type = model_type
        self.random_state = random_state
        self.kwargs = kwargs
        self.model = self._create_model()
        self._is_fitted = False

    def _create_model(self) -> BaseEstimator:
        """Create the underlying sklearn model."""
        if self.model_type == "logreg":
            return LogisticRegression(
                solver="liblinear",
                random_state=self.random_state,
                max_iter=1000,
                **self.kwargs,
            )
        elif self.model_type == "logreg_cv":
            return LogisticRegressionCV(
                solver="liblinear",
                cv=5,
                random_state=self.random_state,
                max_iter=1000,
                **self.kwargs,
            )
        elif self.model_type == "rf":
            return RandomForestClassifier(
                n_estimators=self.kwargs.get("n_estimators", 100),
                random_state=self.random_state,
                n_jobs=-1,
                **{k: v for k, v in self.kwargs.items() if k != "n_estimators"},
            )
        else:
            raise ValueError(
                f"Unknown model type: {self.model_type}. Supported: {self.SUPPORTED_MODELS}"
            )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "CreditRiskModel":
        """Fit the model.

        Args:
            X: Feature matrix.
            y: Labels.
            sample_weight: Optional sample weights (for reweighing mitigation).

        Returns:
            Self for chaining.
        """
        if sample_weight is not None:
            self.model.fit(X, y, sample_weight=sample_weight)
        else:
            self.model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Feature matrix.

        Returns:
            Predicted labels.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature matrix.

        Returns:
            Predicted probabilities for each class.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        return self.model.predict_proba(X)

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            "model_type": self.model_type,
            "random_state": self.random_state,
            **self.model.get_params(),
        }


def train_baseline_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "logreg",
    random_state: int = 42,
    sample_weight: Optional[np.ndarray] = None,
) -> CreditRiskModel:
    """Train a baseline credit risk model.

    Args:
        X_train: Training features.
        y_train: Training labels.
        model_type: Type of model to train.
        random_state: Random seed.
        sample_weight: Optional sample weights.

    Returns:
        Trained CreditRiskModel instance.
    """
    model = CreditRiskModel(model_type=model_type, random_state=random_state)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def get_predictions(
    model: CreditRiskModel,
    X: pd.DataFrame,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get predictions and probabilities from model.

    Args:
        model: Trained model.
        X: Feature matrix.
        threshold: Classification threshold.

    Returns:
        Tuple of (binary predictions, positive class probabilities).
    """
    proba = model.predict_proba(X)[:, 1]
    predictions = (proba >= threshold).astype(int)
    return predictions, proba
