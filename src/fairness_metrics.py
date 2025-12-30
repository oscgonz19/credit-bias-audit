"""Fairness metrics computation using AIF360 and custom implementations."""

from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric


def compute_performance_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute standard classification performance metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities (for AUC).

    Returns:
        Dictionary of performance metrics.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_proba is not None:
        try:
            metrics["auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics["auc"] = np.nan

    return metrics


def compute_confusion_matrix_rates(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute confusion matrix rates.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary with TPR, FPR, TNR, FNR.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Positive Rate (Recall)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # True Negative Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # False Negative Rate

    return {"tpr": tpr, "fpr": fpr, "tnr": tnr, "fnr": fnr}


def compute_statistical_parity_difference(
    y_pred: np.ndarray,
    protected_attr: np.ndarray,
    privileged_value: int = 1,
) -> float:
    """Compute Statistical Parity Difference (SPD).

    SPD = P(Y_pred=1 | unprivileged) - P(Y_pred=1 | privileged)

    A value of 0 indicates perfect parity. Negative values indicate
    bias against the unprivileged group.

    Args:
        y_pred: Predicted labels.
        protected_attr: Protected attribute values.
        privileged_value: Value indicating privileged group.

    Returns:
        Statistical Parity Difference.
    """
    privileged_mask = protected_attr == privileged_value
    unprivileged_mask = ~privileged_mask

    p_priv = np.mean(y_pred[privileged_mask]) if privileged_mask.sum() > 0 else 0
    p_unpriv = np.mean(y_pred[unprivileged_mask]) if unprivileged_mask.sum() > 0 else 0

    return p_unpriv - p_priv


def compute_disparate_impact(
    y_pred: np.ndarray,
    protected_attr: np.ndarray,
    privileged_value: int = 1,
) -> float:
    """Compute Disparate Impact (DI).

    DI = P(Y_pred=1 | unprivileged) / P(Y_pred=1 | privileged)

    A value of 1 indicates perfect parity. Values < 0.8 often indicate
    significant disparate impact (80% rule).

    Args:
        y_pred: Predicted labels.
        protected_attr: Protected attribute values.
        privileged_value: Value indicating privileged group.

    Returns:
        Disparate Impact ratio.
    """
    privileged_mask = protected_attr == privileged_value
    unprivileged_mask = ~privileged_mask

    p_priv = np.mean(y_pred[privileged_mask]) if privileged_mask.sum() > 0 else 0
    p_unpriv = np.mean(y_pred[unprivileged_mask]) if unprivileged_mask.sum() > 0 else 0

    if p_priv == 0:
        return np.inf if p_unpriv > 0 else 1.0

    return p_unpriv / p_priv


def compute_equal_opportunity_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected_attr: np.ndarray,
    privileged_value: int = 1,
) -> float:
    """Compute Equal Opportunity Difference (EOD).

    EOD = TPR_unprivileged - TPR_privileged

    A value of 0 indicates equal opportunity. Measures difference in
    true positive rates between groups.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        protected_attr: Protected attribute values.
        privileged_value: Value indicating privileged group.

    Returns:
        Equal Opportunity Difference.
    """
    privileged_mask = protected_attr == privileged_value
    unprivileged_mask = ~privileged_mask

    # TPR for privileged group
    priv_pos = (y_true == 1) & privileged_mask
    tpr_priv = np.mean(y_pred[priv_pos]) if priv_pos.sum() > 0 else 0

    # TPR for unprivileged group
    unpriv_pos = (y_true == 1) & unprivileged_mask
    tpr_unpriv = np.mean(y_pred[unpriv_pos]) if unpriv_pos.sum() > 0 else 0

    return tpr_unpriv - tpr_priv


def compute_average_odds_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected_attr: np.ndarray,
    privileged_value: int = 1,
) -> float:
    """Compute Average Odds Difference (AOD).

    AOD = 0.5 * [(FPR_unpriv - FPR_priv) + (TPR_unpriv - TPR_priv)]

    A value of 0 indicates equalized odds. Combines both FPR and TPR
    differences.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        protected_attr: Protected attribute values.
        privileged_value: Value indicating privileged group.

    Returns:
        Average Odds Difference.
    """
    privileged_mask = protected_attr == privileged_value
    unprivileged_mask = ~privileged_mask

    # TPR
    priv_pos = (y_true == 1) & privileged_mask
    unpriv_pos = (y_true == 1) & unprivileged_mask
    tpr_priv = np.mean(y_pred[priv_pos]) if priv_pos.sum() > 0 else 0
    tpr_unpriv = np.mean(y_pred[unpriv_pos]) if unpriv_pos.sum() > 0 else 0

    # FPR
    priv_neg = (y_true == 0) & privileged_mask
    unpriv_neg = (y_true == 0) & unprivileged_mask
    fpr_priv = np.mean(y_pred[priv_neg]) if priv_neg.sum() > 0 else 0
    fpr_unpriv = np.mean(y_pred[unpriv_neg]) if unpriv_neg.sum() > 0 else 0

    return 0.5 * ((fpr_unpriv - fpr_priv) + (tpr_unpriv - tpr_priv))


def compute_fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected_attr: np.ndarray,
    privileged_value: int = 1,
) -> Dict[str, float]:
    """Compute all fairness metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        protected_attr: Protected attribute values.
        privileged_value: Value indicating privileged group.

    Returns:
        Dictionary of fairness metrics.
    """
    return {
        "statistical_parity_difference": compute_statistical_parity_difference(
            y_pred, protected_attr, privileged_value
        ),
        "disparate_impact": compute_disparate_impact(
            y_pred, protected_attr, privileged_value
        ),
        "equal_opportunity_difference": compute_equal_opportunity_difference(
            y_true, y_pred, protected_attr, privileged_value
        ),
        "average_odds_difference": compute_average_odds_difference(
            y_true, y_pred, protected_attr, privileged_value
        ),
    }


def compute_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    protected_attr: np.ndarray,
    privileged_value: int = 1,
) -> Dict[str, Dict[str, float]]:
    """Compute performance metrics by group.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities.
        protected_attr: Protected attribute values.
        privileged_value: Value indicating privileged group.

    Returns:
        Dictionary with metrics for each group and overall.
    """
    privileged_mask = protected_attr == privileged_value
    unprivileged_mask = ~privileged_mask

    results = {
        "overall": compute_performance_metrics(y_true, y_pred, y_proba),
        "privileged": compute_performance_metrics(
            y_true[privileged_mask],
            y_pred[privileged_mask],
            y_proba[privileged_mask] if y_proba is not None else None,
        ),
        "unprivileged": compute_performance_metrics(
            y_true[unprivileged_mask],
            y_pred[unprivileged_mask],
            y_proba[unprivileged_mask] if y_proba is not None else None,
        ),
    }

    return results


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    protected_attr: np.ndarray,
    privileged_value: int = 1,
) -> Dict[str, Any]:
    """Compute all performance and fairness metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities.
        protected_attr: Protected attribute values.
        privileged_value: Value indicating privileged group.

    Returns:
        Dictionary with all metrics organized by category.
    """
    return {
        "performance": compute_performance_metrics(y_true, y_pred, y_proba),
        "fairness": compute_fairness_metrics(
            y_true, y_pred, protected_attr, privileged_value
        ),
        "group_performance": compute_group_metrics(
            y_true, y_pred, y_proba, protected_attr, privileged_value
        ),
    }


def compute_aif360_metrics(
    dataset_true: BinaryLabelDataset,
    dataset_pred: BinaryLabelDataset,
    privileged_groups: List[Dict[str, int]],
    unprivileged_groups: List[Dict[str, int]],
) -> Dict[str, float]:
    """Compute fairness metrics using AIF360.

    Args:
        dataset_true: AIF360 dataset with true labels.
        dataset_pred: AIF360 dataset with predicted labels.
        privileged_groups: Definition of privileged groups.
        unprivileged_groups: Definition of unprivileged groups.

    Returns:
        Dictionary of AIF360 fairness metrics.
    """
    cm = ClassificationMetric(
        dataset_true,
        dataset_pred,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups,
    )

    return {
        "statistical_parity_difference": cm.statistical_parity_difference(),
        "disparate_impact": cm.disparate_impact(),
        "equal_opportunity_difference": cm.equal_opportunity_difference(),
        "average_odds_difference": cm.average_odds_difference(),
        "theil_index": cm.theil_index(),
    }


def metrics_to_dataframe(
    metrics: Dict[str, Any],
    model_name: str = "baseline",
    mitigation: str = "none",
) -> pd.DataFrame:
    """Convert metrics dictionary to DataFrame for reporting.

    Args:
        metrics: Dictionary of metrics from compute_all_metrics.
        model_name: Name of the model.
        mitigation: Name of mitigation applied.

    Returns:
        DataFrame with one row per metric.
    """
    rows = []

    # Performance metrics
    for metric_name, value in metrics["performance"].items():
        rows.append({
            "model": model_name,
            "mitigation": mitigation,
            "category": "performance",
            "metric": metric_name,
            "value": value,
        })

    # Fairness metrics
    for metric_name, value in metrics["fairness"].items():
        rows.append({
            "model": model_name,
            "mitigation": mitigation,
            "category": "fairness",
            "metric": metric_name,
            "value": value,
        })

    return pd.DataFrame(rows)
