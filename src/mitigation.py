"""Bias mitigation strategies using AIF360."""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from aif360.algorithms.postprocessing import (
    CalibratedEqOddsPostprocessing,
    EqOddsPostprocessing,
)
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import BinaryLabelDataset

from .data import create_aif360_dataset


class ReweighingMitigation:
    """Pre-processing mitigation using Reweighing algorithm.

    Reweighing assigns weights to training samples to achieve statistical
    parity without changing labels.
    """

    def __init__(
        self,
        privileged_groups: List[Dict[str, int]],
        unprivileged_groups: List[Dict[str, int]],
    ):
        """Initialize Reweighing mitigation.

        Args:
            privileged_groups: Definition of privileged groups.
            unprivileged_groups: Definition of unprivileged groups.
        """
        self.privileged_groups = privileged_groups
        self.unprivileged_groups = unprivileged_groups
        self.reweigher = Reweighing(
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups,
        )
        self._is_fitted = False

    def fit(self, dataset: BinaryLabelDataset) -> "ReweighingMitigation":
        """Fit the reweighing transformation.

        Args:
            dataset: AIF360 BinaryLabelDataset.

        Returns:
            Self for chaining.
        """
        self.reweigher.fit(dataset)
        self._is_fitted = True
        return self

    def transform(self, dataset: BinaryLabelDataset) -> BinaryLabelDataset:
        """Apply reweighing transformation.

        Args:
            dataset: AIF360 BinaryLabelDataset.

        Returns:
            Transformed dataset with sample weights.
        """
        if not self._is_fitted:
            raise RuntimeError("ReweighingMitigation must be fitted before transform.")
        return self.reweigher.transform(dataset)

    def fit_transform(self, dataset: BinaryLabelDataset) -> BinaryLabelDataset:
        """Fit and transform in one step.

        Args:
            dataset: AIF360 BinaryLabelDataset.

        Returns:
            Transformed dataset with sample weights.
        """
        return self.fit(dataset).transform(dataset)

    def get_weights(self, dataset: BinaryLabelDataset) -> np.ndarray:
        """Get sample weights for training.

        Args:
            dataset: AIF360 BinaryLabelDataset.

        Returns:
            Array of sample weights.
        """
        transformed = self.fit_transform(dataset)
        return transformed.instance_weights


class EqOddsMitigation:
    """Post-processing mitigation using Equalized Odds.

    Adjusts predictions to achieve equalized odds (equal TPR and FPR
    across groups).
    """

    def __init__(
        self,
        privileged_groups: List[Dict[str, int]],
        unprivileged_groups: List[Dict[str, int]],
        seed: int = 42,
    ):
        """Initialize Equalized Odds post-processing.

        Args:
            privileged_groups: Definition of privileged groups.
            unprivileged_groups: Definition of unprivileged groups.
            seed: Random seed for reproducibility.
        """
        self.privileged_groups = privileged_groups
        self.unprivileged_groups = unprivileged_groups
        self.seed = seed
        self.postprocessor = EqOddsPostprocessing(
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups,
            seed=seed,
        )
        self._is_fitted = False

    def fit(
        self,
        dataset_true: BinaryLabelDataset,
        dataset_pred: BinaryLabelDataset,
    ) -> "EqOddsMitigation":
        """Fit the post-processor on validation data.

        Args:
            dataset_true: Dataset with true labels.
            dataset_pred: Dataset with predicted labels.

        Returns:
            Self for chaining.
        """
        self.postprocessor.fit(dataset_true, dataset_pred)
        self._is_fitted = True
        return self

    def predict(self, dataset_pred: BinaryLabelDataset) -> BinaryLabelDataset:
        """Apply equalized odds adjustment to predictions.

        Args:
            dataset_pred: Dataset with predicted labels.

        Returns:
            Dataset with adjusted predictions.
        """
        if not self._is_fitted:
            raise RuntimeError("EqOddsMitigation must be fitted before predict.")
        return self.postprocessor.predict(dataset_pred)


class CalibratedEqOddsMitigation:
    """Post-processing mitigation using Calibrated Equalized Odds.

    Similar to Equalized Odds but preserves calibration of predictions.
    """

    def __init__(
        self,
        privileged_groups: List[Dict[str, int]],
        unprivileged_groups: List[Dict[str, int]],
        cost_constraint: str = "weighted",
        seed: int = 42,
    ):
        """Initialize Calibrated Equalized Odds post-processing.

        Args:
            privileged_groups: Definition of privileged groups.
            unprivileged_groups: Definition of unprivileged groups.
            cost_constraint: Type of constraint ('fpr', 'fnr', or 'weighted').
            seed: Random seed for reproducibility.
        """
        self.privileged_groups = privileged_groups
        self.unprivileged_groups = unprivileged_groups
        self.cost_constraint = cost_constraint
        self.seed = seed
        self.postprocessor = CalibratedEqOddsPostprocessing(
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups,
            cost_constraint=cost_constraint,
            seed=seed,
        )
        self._is_fitted = False

    def fit(
        self,
        dataset_true: BinaryLabelDataset,
        dataset_pred: BinaryLabelDataset,
    ) -> "CalibratedEqOddsMitigation":
        """Fit the post-processor on validation data.

        Args:
            dataset_true: Dataset with true labels.
            dataset_pred: Dataset with predicted labels.

        Returns:
            Self for chaining.
        """
        self.postprocessor.fit(dataset_true, dataset_pred)
        self._is_fitted = True
        return self

    def predict(self, dataset_pred: BinaryLabelDataset) -> BinaryLabelDataset:
        """Apply calibrated equalized odds adjustment.

        Args:
            dataset_pred: Dataset with predicted labels.

        Returns:
            Dataset with adjusted predictions.
        """
        if not self._is_fitted:
            raise RuntimeError("CalibratedEqOddsMitigation must be fitted before predict.")
        return self.postprocessor.predict(dataset_pred)


def apply_reweighing(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    protected_attr: str,
    privileged_groups: List[Dict[str, int]],
    unprivileged_groups: List[Dict[str, int]],
) -> np.ndarray:
    """Apply reweighing and return sample weights.

    Args:
        X_train: Training features.
        y_train: Training labels.
        protected_attr: Protected attribute column name.
        privileged_groups: Definition of privileged groups.
        unprivileged_groups: Definition of unprivileged groups.

    Returns:
        Sample weights for training.
    """
    dataset = create_aif360_dataset(X_train, y_train, protected_attr)

    mitigation = ReweighingMitigation(privileged_groups, unprivileged_groups)
    return mitigation.get_weights(dataset)


def apply_eq_odds_postprocessing(
    X_true: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    protected_attr: str,
    privileged_groups: List[Dict[str, int]],
    unprivileged_groups: List[Dict[str, int]],
    seed: int = 42,
) -> np.ndarray:
    """Apply equalized odds post-processing to predictions.

    Args:
        X_true: Features (must include protected attribute).
        y_true: True labels.
        y_pred: Predicted labels.
        protected_attr: Protected attribute column name.
        privileged_groups: Definition of privileged groups.
        unprivileged_groups: Definition of unprivileged groups.
        seed: Random seed.

    Returns:
        Adjusted predictions.
    """
    # Create dataset with true labels
    dataset_true = create_aif360_dataset(X_true, y_true, protected_attr)

    # Create dataset with predicted labels
    dataset_pred = create_aif360_dataset(
        X_true, pd.Series(y_pred, index=y_true.index), protected_attr
    )

    # Apply post-processing
    mitigation = EqOddsMitigation(privileged_groups, unprivileged_groups, seed)
    mitigation.fit(dataset_true, dataset_pred)
    adjusted = mitigation.predict(dataset_pred)

    return adjusted.labels.ravel()


def get_mitigation_strategy(
    mitigation_name: str,
    privileged_groups: List[Dict[str, int]],
    unprivileged_groups: List[Dict[str, int]],
    seed: int = 42,
) -> Optional[Any]:
    """Factory function to get mitigation strategy by name.

    Args:
        mitigation_name: Name of mitigation ('none', 'reweighing', 'eq_odds').
        privileged_groups: Definition of privileged groups.
        unprivileged_groups: Definition of unprivileged groups.
        seed: Random seed.

    Returns:
        Mitigation object or None if 'none'.
    """
    if mitigation_name == "none":
        return None
    elif mitigation_name == "reweighing":
        return ReweighingMitigation(privileged_groups, unprivileged_groups)
    elif mitigation_name == "eq_odds":
        return EqOddsMitigation(privileged_groups, unprivileged_groups, seed)
    elif mitigation_name == "calibrated_eq_odds":
        return CalibratedEqOddsMitigation(privileged_groups, unprivileged_groups, seed=seed)
    else:
        raise ValueError(
            f"Unknown mitigation: {mitigation_name}. "
            f"Supported: none, reweighing, eq_odds, calibrated_eq_odds"
        )
