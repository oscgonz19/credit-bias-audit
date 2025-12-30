#!/usr/bin/env python
"""End-to-end credit bias audit pipeline."""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_and_prepare_data, create_aif360_dataset
from src.models import train_baseline_model, get_predictions
from src.fairness_metrics import compute_all_metrics
from src.mitigation import apply_reweighing, apply_eq_odds_postprocessing
from src.reporting import generate_metrics_csv, generate_markdown_report


def run_baseline_audit(
    data: Dict[str, Any],
    model_type: str,
    seed: int,
) -> Dict[str, Any]:
    """Run baseline model audit without mitigation.

    Args:
        data: Prepared data dictionary.
        model_type: Type of model to train.
        seed: Random seed.

    Returns:
        Results dictionary.
    """
    print("\n" + "=" * 60)
    print("Running BASELINE audit (no mitigation)")
    print("=" * 60)

    # Train model
    model = train_baseline_model(
        data["X_train"],
        data["y_train"],
        model_type=model_type,
        random_state=seed,
    )

    # Get predictions
    y_pred, y_proba = get_predictions(model, data["X_test"])

    # Get protected attribute values for test set
    protected_values = data["X_test"][data["protected_attr"]].values

    # Compute metrics
    metrics = compute_all_metrics(
        data["y_test"].values,
        y_pred,
        y_proba,
        protected_values,
        privileged_value=data["privileged_groups"][0][data["protected_attr"]],
    )

    print(f"\nPerformance:")
    print(f"  Accuracy: {metrics['performance']['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {metrics['performance']['balanced_accuracy']:.4f}")
    print(f"  AUC: {metrics['performance']['auc']:.4f}")

    print(f"\nFairness:")
    print(f"  Statistical Parity Diff: {metrics['fairness']['statistical_parity_difference']:.4f}")
    print(f"  Disparate Impact: {metrics['fairness']['disparate_impact']:.4f}")
    print(f"  Equal Opportunity Diff: {metrics['fairness']['equal_opportunity_difference']:.4f}")
    print(f"  Average Odds Diff: {metrics['fairness']['average_odds_difference']:.4f}")

    return {
        "model": model_type,
        "mitigation": "none",
        "metrics": metrics,
        "predictions": y_pred,
        "probabilities": y_proba,
    }


def run_reweighing_audit(
    data: Dict[str, Any],
    model_type: str,
    seed: int,
) -> Dict[str, Any]:
    """Run audit with Reweighing pre-processing.

    Args:
        data: Prepared data dictionary.
        model_type: Type of model to train.
        seed: Random seed.

    Returns:
        Results dictionary.
    """
    print("\n" + "=" * 60)
    print("Running REWEIGHING audit (pre-processing mitigation)")
    print("=" * 60)

    # Get sample weights
    sample_weights = apply_reweighing(
        data["X_train"],
        data["y_train"],
        data["protected_attr"],
        data["privileged_groups"],
        data["unprivileged_groups"],
    )

    print(f"Applied reweighing: weights range [{sample_weights.min():.3f}, {sample_weights.max():.3f}]")

    # Train model with weights
    model = train_baseline_model(
        data["X_train"],
        data["y_train"],
        model_type=model_type,
        random_state=seed,
        sample_weight=sample_weights,
    )

    # Get predictions
    y_pred, y_proba = get_predictions(model, data["X_test"])

    # Get protected attribute values
    protected_values = data["X_test"][data["protected_attr"]].values

    # Compute metrics
    metrics = compute_all_metrics(
        data["y_test"].values,
        y_pred,
        y_proba,
        protected_values,
        privileged_value=data["privileged_groups"][0][data["protected_attr"]],
    )

    print(f"\nPerformance:")
    print(f"  Accuracy: {metrics['performance']['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {metrics['performance']['balanced_accuracy']:.4f}")
    print(f"  AUC: {metrics['performance']['auc']:.4f}")

    print(f"\nFairness:")
    print(f"  Statistical Parity Diff: {metrics['fairness']['statistical_parity_difference']:.4f}")
    print(f"  Disparate Impact: {metrics['fairness']['disparate_impact']:.4f}")
    print(f"  Equal Opportunity Diff: {metrics['fairness']['equal_opportunity_difference']:.4f}")
    print(f"  Average Odds Diff: {metrics['fairness']['average_odds_difference']:.4f}")

    return {
        "model": model_type,
        "mitigation": "reweighing",
        "metrics": metrics,
        "predictions": y_pred,
        "probabilities": y_proba,
    }


def run_eq_odds_audit(
    data: Dict[str, Any],
    model_type: str,
    seed: int,
    baseline_predictions: np.ndarray,
) -> Dict[str, Any]:
    """Run audit with Equalized Odds post-processing.

    Args:
        data: Prepared data dictionary.
        model_type: Type of model to train.
        seed: Random seed.
        baseline_predictions: Predictions from baseline model.

    Returns:
        Results dictionary.
    """
    print("\n" + "=" * 60)
    print("Running EQUALIZED ODDS audit (post-processing mitigation)")
    print("=" * 60)

    # Apply equalized odds post-processing
    y_pred_adjusted = apply_eq_odds_postprocessing(
        data["X_test"],
        data["y_test"],
        baseline_predictions,
        data["protected_attr"],
        data["privileged_groups"],
        data["unprivileged_groups"],
        seed=seed,
    )

    # Get protected attribute values
    protected_values = data["X_test"][data["protected_attr"]].values

    # Compute metrics (no probabilities for post-processed predictions)
    metrics = compute_all_metrics(
        data["y_test"].values,
        y_pred_adjusted.astype(int),
        None,  # No probabilities available
        protected_values,
        privileged_value=data["privileged_groups"][0][data["protected_attr"]],
    )

    print(f"\nPerformance:")
    print(f"  Accuracy: {metrics['performance']['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {metrics['performance']['balanced_accuracy']:.4f}")
    print(f"  AUC: N/A (post-processing)")

    print(f"\nFairness:")
    print(f"  Statistical Parity Diff: {metrics['fairness']['statistical_parity_difference']:.4f}")
    print(f"  Disparate Impact: {metrics['fairness']['disparate_impact']:.4f}")
    print(f"  Equal Opportunity Diff: {metrics['fairness']['equal_opportunity_difference']:.4f}")
    print(f"  Average Odds Diff: {metrics['fairness']['average_odds_difference']:.4f}")

    return {
        "model": model_type,
        "mitigation": "eq_odds",
        "metrics": metrics,
        "predictions": y_pred_adjusted,
        "probabilities": None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run credit bias audit pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=["german", "adult"],
        default="german",
        help="Dataset to audit",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to dataset CSV (default: data/{dataset}_credit.csv)",
    )
    parser.add_argument(
        "--protected-attr",
        choices=["sex", "age", "race", "foreign_worker"],
        default="sex",
        help="Protected attribute to analyze",
    )
    parser.add_argument(
        "--model",
        choices=["logreg", "logreg_cv", "rf"],
        default="logreg",
        help="Model type to train",
    )
    parser.add_argument(
        "--mitigation",
        choices=["none", "reweighing", "eq_odds", "all"],
        default="all",
        help="Mitigation strategy to apply",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Limit dataset size (for testing)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports"),
        help="Output directory for reports",
    )

    args = parser.parse_args()

    # Set data path
    if args.data_path is None:
        if args.dataset == "german":
            args.data_path = Path("data/german_credit.csv")
        else:
            args.data_path = Path("data/adult_data.csv")

    # Validate protected attribute for dataset
    valid_attrs = {
        "german": ["sex", "age", "foreign_worker"],
        "adult": ["sex", "race"],
    }
    if args.protected_attr not in valid_attrs[args.dataset]:
        print(f"Error: {args.protected_attr} not valid for {args.dataset} dataset")
        print(f"Valid options: {valid_attrs[args.dataset]}")
        sys.exit(1)

    # Check data exists
    if not args.data_path.exists():
        print(f"Error: Data file not found: {args.data_path}")
        print("Run 'python scripts/download_data.py' first")
        sys.exit(1)

    print("=" * 60)
    print("CREDIT BIAS AUDIT")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Data path: {args.data_path}")
    print(f"Protected attribute: {args.protected_attr}")
    print(f"Model: {args.model}")
    print(f"Mitigation: {args.mitigation}")
    print(f"Seed: {args.seed}")

    # Load and prepare data
    print("\nLoading and preparing data...")
    data = load_and_prepare_data(
        dataset=args.dataset,
        data_path=args.data_path,
        protected_attr=args.protected_attr,
        random_state=args.seed,
        sample_size=args.sample_size,
    )
    print(f"  Train size: {len(data['X_train'])}")
    print(f"  Test size: {len(data['X_test'])}")

    # Run audits
    results = []

    # Baseline (always run)
    baseline_result = run_baseline_audit(data, args.model, args.seed)
    results.append(baseline_result)

    # Mitigations
    if args.mitigation in ["reweighing", "all"]:
        reweighing_result = run_reweighing_audit(data, args.model, args.seed)
        results.append(reweighing_result)

    if args.mitigation in ["eq_odds", "all"]:
        eq_odds_result = run_eq_odds_audit(
            data, args.model, args.seed, baseline_result["predictions"]
        )
        results.append(eq_odds_result)

    # Generate reports
    print("\n" + "=" * 60)
    print("GENERATING REPORTS")
    print("=" * 60)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # CSV metrics
    csv_path = args.out_dir / "metrics.csv"
    generate_metrics_csv(results, csv_path)
    print(f"  Metrics CSV: {csv_path}")

    # Markdown report
    config = {
        "dataset": args.dataset,
        "protected_attr": args.protected_attr,
        "model": args.model,
        "seed": args.seed,
    }
    md_path = args.out_dir / "report.md"
    generate_markdown_report(results, config, md_path)
    print(f"  Report: {md_path}")

    print("\nAudit complete!")


if __name__ == "__main__":
    main()
