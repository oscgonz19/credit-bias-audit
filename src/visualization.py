"""Visualization module for credit bias audit storytelling.

This module provides comprehensive visualizations for fairness audits,
including metrics comparisons, group analysis, and mitigation effectiveness.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix, roc_curve

# Set consistent style
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {
    "privileged": "#3498db",
    "unprivileged": "#e74c3c",
    "baseline": "#95a5a6",
    "mitigated": "#2ecc71",
    "threshold": "#f39c12",
    "neutral": "#34495e",
}


def setup_style():
    """Configure matplotlib style for consistent visualizations."""
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "figure.dpi": 100,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 16,
        }
    )


# =============================================================================
# FAIRNESS METRICS VISUALIZATIONS
# =============================================================================


def plot_fairness_metrics_bar(
    metrics: Dict[str, float],
    title: str = "Fairness Metrics",
    figsize: Tuple[int, int] = (10, 6),
    show_thresholds: bool = True,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Create bar chart of fairness metrics with threshold indicators.

    Args:
        metrics: Dictionary with fairness metric names and values.
        title: Chart title.
        figsize: Figure size.
        show_thresholds: Whether to show fairness thresholds.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib figure.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=figsize)

    metric_labels = {
        "statistical_parity_difference": "Statistical\nParity Diff",
        "disparate_impact": "Disparate\nImpact",
        "equal_opportunity_difference": "Equal\nOpportunity Diff",
        "average_odds_difference": "Average\nOdds Diff",
    }

    # Filter and order metrics
    ordered_metrics = []
    for key in metric_labels:
        if key in metrics:
            ordered_metrics.append((metric_labels[key], metrics[key], key))

    if not ordered_metrics:
        ax.text(0.5, 0.5, "No fairness metrics available", ha="center", va="center")
        return fig

    labels = [m[0] for m in ordered_metrics]
    values = [m[1] for m in ordered_metrics]
    keys = [m[2] for m in ordered_metrics]

    # Determine colors based on fairness
    colors = []
    for val, key in zip(values, keys):
        if key == "disparate_impact":
            # DI should be between 0.8 and 1.25
            if 0.8 <= val <= 1.25:
                colors.append("#2ecc71")  # Green - fair
            else:
                colors.append("#e74c3c")  # Red - unfair
        else:
            # Other metrics should be close to 0
            if abs(val) <= 0.1:
                colors.append("#2ecc71")  # Green - fair
            else:
                colors.append("#e74c3c")  # Red - unfair

    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=1.2)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        va = "bottom" if height >= 0 else "top"
        offset = 0.02 if height >= 0 else -0.02
        ax.annotate(
            f"{val:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5 if height >= 0 else -5),
            textcoords="offset points",
            ha="center",
            va=va,
            fontweight="bold",
        )

    # Add threshold lines
    if show_thresholds:
        ax.axhline(y=0.1, color=COLORS["threshold"], linestyle="--", linewidth=2, label="Fair threshold (±0.1)")
        ax.axhline(y=-0.1, color=COLORS["threshold"], linestyle="--", linewidth=2)
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Metric Value")
    ax.set_title(title, fontweight="bold", pad=20)

    # Legend
    legend_elements = [
        Patch(facecolor="#2ecc71", edgecolor="black", label="Within fair range"),
        Patch(facecolor="#e74c3c", edgecolor="black", label="Outside fair range"),
    ]
    if show_thresholds:
        legend_elements.append(plt.Line2D([0], [0], color=COLORS["threshold"], linestyle="--", label="Threshold (±0.1)"))
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_fairness_radar(
    metrics: Dict[str, float],
    title: str = "Fairness Profile",
    figsize: Tuple[int, int] = (8, 8),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Create radar/spider chart of fairness metrics.

    Args:
        metrics: Dictionary with fairness metric names and values.
        title: Chart title.
        figsize: Figure size.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib figure.
    """
    setup_style()

    # Normalize metrics to 0-1 scale for radar chart
    metric_labels = {
        "statistical_parity_difference": "Statistical Parity",
        "disparate_impact": "Disparate Impact",
        "equal_opportunity_difference": "Equal Opportunity",
        "average_odds_difference": "Average Odds",
    }

    # Compute fairness scores (1 = perfectly fair, 0 = very unfair)
    scores = []
    labels = []
    for key, label in metric_labels.items():
        if key in metrics:
            val = metrics[key]
            if key == "disparate_impact":
                # DI: 1.0 is perfect, further from 1 is worse
                score = max(0, 1 - abs(val - 1) / 0.5)  # Normalize
            else:
                # Others: 0 is perfect, further from 0 is worse
                score = max(0, 1 - abs(val) / 0.3)  # Normalize
            scores.append(score)
            labels.append(label)

    if len(scores) < 3:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Need at least 3 metrics for radar chart", ha="center", va="center")
        return fig

    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    scores += scores[:1]  # Complete the loop
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    # Plot data
    ax.fill(angles, scores, color=COLORS["privileged"], alpha=0.25)
    ax.plot(angles, scores, color=COLORS["privileged"], linewidth=2, marker="o", markersize=8)

    # Add fair threshold circle
    fair_threshold = [0.7] * (len(labels) + 1)
    ax.plot(angles, fair_threshold, color=COLORS["threshold"], linestyle="--", linewidth=2, label="Fair threshold")

    # Configure chart
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"])
    ax.set_title(title, fontweight="bold", pad=20, size=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# =============================================================================
# GROUP COMPARISON VISUALIZATIONS
# =============================================================================


def plot_group_metrics_comparison(
    group_metrics: Dict[str, Dict[str, float]],
    title: str = "Performance by Group",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Create grouped bar chart comparing metrics across groups.

    Args:
        group_metrics: Dictionary with 'privileged' and 'unprivileged' keys,
                       each containing performance metrics.
        title: Chart title.
        figsize: Figure size.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib figure.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=figsize)

    # Get metrics present in both groups
    priv_metrics = group_metrics.get("privileged", {})
    unpriv_metrics = group_metrics.get("unprivileged", {})
    overall_metrics = group_metrics.get("overall", {})

    common_metrics = set(priv_metrics.keys()) & set(unpriv_metrics.keys())
    metric_names = sorted(common_metrics)

    if not metric_names:
        ax.text(0.5, 0.5, "No common metrics found", ha="center", va="center")
        return fig

    x = np.arange(len(metric_names))
    width = 0.25

    # Plot bars for each group
    priv_values = [priv_metrics.get(m, 0) for m in metric_names]
    unpriv_values = [unpriv_metrics.get(m, 0) for m in metric_names]
    overall_values = [overall_metrics.get(m, 0) for m in metric_names]

    bars1 = ax.bar(x - width, priv_values, width, label="Privileged", color=COLORS["privileged"], edgecolor="black")
    bars2 = ax.bar(x, unpriv_values, width, label="Unprivileged", color=COLORS["unprivileged"], edgecolor="black")
    bars3 = ax.bar(x + width, overall_values, width, label="Overall", color=COLORS["neutral"], edgecolor="black", alpha=0.7)

    # Add value labels
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)

    # Format metric names for display
    formatted_names = [m.replace("_", " ").title() for m in metric_names]
    ax.set_xticks(x)
    ax.set_xticklabels(formatted_names, rotation=45, ha="right")
    ax.set_ylabel("Metric Value")
    ax.set_title(title, fontweight="bold", pad=20)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.15)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_outcome_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected_attr: np.ndarray,
    privileged_value: int = 1,
    group_names: Tuple[str, str] = ("Unprivileged", "Privileged"),
    title: str = "Outcome Distribution by Group",
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Create stacked bar charts showing outcome distribution by group.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        protected_attr: Protected attribute values.
        privileged_value: Value indicating privileged group.
        group_names: Names for (unprivileged, privileged) groups.
        title: Chart title.
        figsize: Figure size.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib figure.
    """
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    privileged_mask = protected_attr == privileged_value

    for ax, (labels, label_type) in zip(axes, [(y_true, "Actual"), (y_pred, "Predicted")]):
        # Calculate proportions
        data = []
        for mask, name in [(~privileged_mask, group_names[0]), (privileged_mask, group_names[1])]:
            group_labels = labels[mask]
            pos_rate = np.mean(group_labels)
            neg_rate = 1 - pos_rate
            data.append({"Group": name, "Positive": pos_rate, "Negative": neg_rate, "N": len(group_labels)})

        df = pd.DataFrame(data)

        # Create stacked bar
        x = np.arange(len(df))
        width = 0.5

        ax.bar(x, df["Negative"], width, label="Negative (0)", color="#e74c3c", alpha=0.8)
        ax.bar(x, df["Positive"], width, bottom=df["Negative"], label="Positive (1)", color="#2ecc71", alpha=0.8)

        # Add percentage labels
        for i, row in df.iterrows():
            ax.text(i, row["Negative"] / 2, f"{row['Negative']:.1%}", ha="center", va="center", fontweight="bold", color="white")
            ax.text(i, row["Negative"] + row["Positive"] / 2, f"{row['Positive']:.1%}", ha="center", va="center", fontweight="bold", color="white")
            ax.text(i, 1.02, f"n={row['N']}", ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{row['Group']}" for _, row in df.iterrows()])
        ax.set_ylabel("Proportion")
        ax.set_title(f"{label_type} Outcomes", fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.legend(loc="upper right")

    fig.suptitle(title, fontweight="bold", size=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# =============================================================================
# MITIGATION COMPARISON VISUALIZATIONS
# =============================================================================


def plot_mitigation_comparison(
    results: List[Dict[str, Any]],
    metric_type: str = "fairness",
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Create comparison chart showing metrics across mitigation strategies.

    Args:
        results: List of result dictionaries from audit runs.
        metric_type: 'fairness' or 'performance'.
        figsize: Figure size.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib figure.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=figsize)

    if not results:
        ax.text(0.5, 0.5, "No results provided", ha="center", va="center")
        return fig

    # Extract metrics from results
    mitigations = []
    all_metrics = set()

    for result in results:
        mitigations.append(result.get("mitigation", "unknown"))
        metrics = result.get("metrics", {}).get(metric_type, {})
        all_metrics.update(metrics.keys())

    metric_names = sorted(all_metrics)
    if not metric_names:
        ax.text(0.5, 0.5, f"No {metric_type} metrics found", ha="center", va="center")
        return fig

    x = np.arange(len(metric_names))
    width = 0.8 / len(mitigations)

    colors = plt.cm.Set2(np.linspace(0, 1, len(mitigations)))

    for i, (result, color) in enumerate(zip(results, colors)):
        metrics = result.get("metrics", {}).get(metric_type, {})
        values = [metrics.get(m, 0) for m in metric_names]
        mitigation = result.get("mitigation", "unknown").replace("_", " ").title()

        offset = (i - len(mitigations) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=mitigation, color=color, edgecolor="black")

        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(
                f"{val:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90 if len(mitigations) > 3 else 0,
            )

    # Add threshold lines for fairness metrics
    if metric_type == "fairness":
        ax.axhline(y=0.1, color=COLORS["threshold"], linestyle="--", linewidth=2, alpha=0.7)
        ax.axhline(y=-0.1, color=COLORS["threshold"], linestyle="--", linewidth=2, alpha=0.7)
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    # Format labels
    formatted_names = [m.replace("_", "\n").title() for m in metric_names]
    ax.set_xticks(x)
    ax.set_xticklabels(formatted_names)
    ax.set_ylabel("Metric Value")
    ax.set_title(f"{metric_type.title()} Metrics by Mitigation Strategy", fontweight="bold", pad=20)
    ax.legend(loc="upper right", title="Mitigation")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_fairness_accuracy_tradeoff(
    results: List[Dict[str, Any]],
    fairness_metric: str = "statistical_parity_difference",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Create scatter plot showing fairness vs accuracy tradeoff.

    Args:
        results: List of result dictionaries from audit runs.
        fairness_metric: Which fairness metric to plot.
        figsize: Figure size.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib figure.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=figsize)

    if not results:
        ax.text(0.5, 0.5, "No results provided", ha="center", va="center")
        return fig

    # Extract data
    points = []
    for result in results:
        mitigation = result.get("mitigation", "unknown")
        metrics = result.get("metrics", {})
        accuracy = metrics.get("performance", {}).get("accuracy", np.nan)
        fairness = metrics.get("fairness", {}).get(fairness_metric, np.nan)
        if not np.isnan(accuracy) and not np.isnan(fairness):
            points.append(
                {
                    "mitigation": mitigation.replace("_", " ").title(),
                    "accuracy": accuracy,
                    "fairness": abs(fairness),  # Use absolute value
                }
            )

    if not points:
        ax.text(0.5, 0.5, "No valid data points", ha="center", va="center")
        return fig

    df = pd.DataFrame(points)

    # Create scatter plot
    colors = plt.cm.Set1(np.linspace(0, 1, len(df)))
    for i, (_, row) in enumerate(df.iterrows()):
        ax.scatter(
            row["fairness"],
            row["accuracy"],
            s=300,
            c=[colors[i]],
            label=row["mitigation"],
            edgecolors="black",
            linewidths=2,
            zorder=3,
        )
        ax.annotate(
            row["mitigation"],
            (row["fairness"], row["accuracy"]),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
        )

    # Add reference lines
    ax.axvline(x=0.1, color=COLORS["threshold"], linestyle="--", linewidth=2, label="Fair threshold (0.1)")

    # Shade fair region
    ax.axvspan(0, 0.1, alpha=0.1, color="green", label="Fair region")

    ax.set_xlabel(f"|{fairness_metric.replace('_', ' ').title()}|", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Fairness vs Accuracy Trade-off", fontweight="bold", pad=20)
    ax.legend(loc="lower left", bbox_to_anchor=(1.02, 0))

    # Set axis limits with padding
    ax.set_xlim(0, max(df["fairness"]) * 1.2)
    ax.set_ylim(min(df["accuracy"]) * 0.95, max(df["accuracy"]) * 1.02)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# =============================================================================
# CONFUSION MATRIX VISUALIZATIONS
# =============================================================================


def plot_confusion_matrices_by_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected_attr: np.ndarray,
    privileged_value: int = 1,
    group_names: Tuple[str, str] = ("Unprivileged", "Privileged"),
    title: str = "Confusion Matrices by Group",
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Create side-by-side confusion matrices for each group.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        protected_attr: Protected attribute values.
        privileged_value: Value indicating privileged group.
        group_names: Names for (unprivileged, privileged) groups.
        title: Chart title.
        figsize: Figure size.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib figure.
    """
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    privileged_mask = protected_attr == privileged_value
    masks = [~privileged_mask, privileged_mask, np.ones(len(y_true), dtype=bool)]
    names = [group_names[0], group_names[1], "Overall"]

    for ax, mask, name in zip(axes, masks, names):
        cm = confusion_matrix(y_true[mask], y_pred[mask], labels=[0, 1])

        # Normalize
        cm_normalized = cm.astype("float") / cm.sum() * 100

        # Create heatmap
        sns.heatmap(
            cm_normalized,
            annot=False,
            fmt=".1f",
            cmap="Blues",
            ax=ax,
            cbar=False,
            square=True,
            linewidths=2,
            linecolor="white",
        )

        # Add custom annotations with counts and percentages
        for i in range(2):
            for j in range(2):
                text = f"{cm[i, j]}\n({cm_normalized[i, j]:.1f}%)"
                color = "white" if cm_normalized[i, j] > 50 else "black"
                ax.text(j + 0.5, i + 0.5, text, ha="center", va="center", fontsize=11, fontweight="bold", color=color)

        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual", fontsize=11)
        ax.set_xticklabels(["Negative", "Positive"])
        ax.set_yticklabels(["Negative", "Positive"], rotation=0)
        ax.set_title(f"{name}\n(n={mask.sum()})", fontweight="bold")

    fig.suptitle(title, fontweight="bold", size=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# =============================================================================
# ROC CURVE VISUALIZATIONS
# =============================================================================


def plot_roc_curves_by_group(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    protected_attr: np.ndarray,
    privileged_value: int = 1,
    group_names: Tuple[str, str] = ("Unprivileged", "Privileged"),
    title: str = "ROC Curves by Group",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Create ROC curves for each group.

    Args:
        y_true: True labels.
        y_proba: Predicted probabilities for positive class.
        protected_attr: Protected attribute values.
        privileged_value: Value indicating privileged group.
        group_names: Names for (unprivileged, privileged) groups.
        title: Chart title.
        figsize: Figure size.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib figure.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=figsize)

    privileged_mask = protected_attr == privileged_value
    colors = [COLORS["unprivileged"], COLORS["privileged"], COLORS["neutral"]]

    for mask, name, color in zip(
        [~privileged_mask, privileged_mask, np.ones(len(y_true), dtype=bool)],
        [group_names[0], group_names[1], "Overall"],
        colors,
    ):
        fpr, tpr, _ = roc_curve(y_true[mask], y_proba[mask])
        auc_score = np.trapz(tpr, fpr)

        linestyle = "--" if name == "Overall" else "-"
        linewidth = 2 if name == "Overall" else 2.5
        ax.plot(fpr, tpr, color=color, linestyle=linestyle, linewidth=linewidth, label=f"{name} (AUC = {auc_score:.3f})")

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random (AUC = 0.500)")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontweight="bold", pad=20)
    ax.legend(loc="lower right", fontsize=10)

    # Add grid
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# =============================================================================
# DISTRIBUTION VISUALIZATIONS
# =============================================================================


def plot_score_distribution(
    y_proba: np.ndarray,
    protected_attr: np.ndarray,
    privileged_value: int = 1,
    group_names: Tuple[str, str] = ("Unprivileged", "Privileged"),
    title: str = "Score Distribution by Group",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Create overlapping histograms of prediction scores by group.

    Args:
        y_proba: Predicted probabilities.
        protected_attr: Protected attribute values.
        privileged_value: Value indicating privileged group.
        group_names: Names for (unprivileged, privileged) groups.
        title: Chart title.
        figsize: Figure size.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib figure.
    """
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    privileged_mask = protected_attr == privileged_value

    # Left plot: Overlapping histograms
    ax1 = axes[0]
    ax1.hist(
        y_proba[~privileged_mask],
        bins=30,
        alpha=0.6,
        color=COLORS["unprivileged"],
        label=group_names[0],
        density=True,
        edgecolor="black",
    )
    ax1.hist(
        y_proba[privileged_mask],
        bins=30,
        alpha=0.6,
        color=COLORS["privileged"],
        label=group_names[1],
        density=True,
        edgecolor="black",
    )
    ax1.axvline(x=0.5, color="black", linestyle="--", linewidth=2, label="Threshold (0.5)")
    ax1.set_xlabel("Predicted Probability", fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.set_title("Score Distribution", fontweight="bold")
    ax1.legend()

    # Right plot: KDE comparison
    ax2 = axes[1]
    try:
        sns.kdeplot(y_proba[~privileged_mask], ax=ax2, color=COLORS["unprivileged"], label=group_names[0], linewidth=2.5, fill=True, alpha=0.3)
        sns.kdeplot(y_proba[privileged_mask], ax=ax2, color=COLORS["privileged"], label=group_names[1], linewidth=2.5, fill=True, alpha=0.3)
    except Exception:
        # Fallback if KDE fails
        ax2.hist(y_proba[~privileged_mask], bins=30, alpha=0.5, color=COLORS["unprivileged"], label=group_names[0], density=True)
        ax2.hist(y_proba[privileged_mask], bins=30, alpha=0.5, color=COLORS["privileged"], label=group_names[1], density=True)

    ax2.axvline(x=0.5, color="black", linestyle="--", linewidth=2, label="Threshold (0.5)")
    ax2.set_xlabel("Predicted Probability", fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.set_title("Kernel Density Estimate", fontweight="bold")
    ax2.legend()

    fig.suptitle(title, fontweight="bold", size=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# =============================================================================
# COMPREHENSIVE DASHBOARD
# =============================================================================


def create_audit_dashboard(
    results: List[Dict[str, Any]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    protected_attr: np.ndarray,
    privileged_value: int = 1,
    group_names: Tuple[str, str] = ("Unprivileged", "Privileged"),
    title: str = "Credit Bias Audit Dashboard",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Create comprehensive dashboard with multiple visualizations.

    Args:
        results: List of result dictionaries from audit runs.
        y_true: True labels.
        y_pred: Predicted labels (from baseline or selected model).
        y_proba: Predicted probabilities.
        protected_attr: Protected attribute values.
        privileged_value: Value indicating privileged group.
        group_names: Names for groups.
        title: Dashboard title.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib figure.
    """
    setup_style()
    fig = plt.figure(figsize=(20, 16))

    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # 1. Fairness Metrics (top-left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    if results:
        baseline = results[0]
        metrics = baseline.get("metrics", {}).get("fairness", {})
        if metrics:
            metric_labels = ["SPD", "DI", "EOD", "AOD"]
            values = [
                metrics.get("statistical_parity_difference", 0),
                metrics.get("disparate_impact", 1) - 1,  # Center around 0
                metrics.get("equal_opportunity_difference", 0),
                metrics.get("average_odds_difference", 0),
            ]
            colors = ["#e74c3c" if abs(v) > 0.1 else "#2ecc71" for v in values]
            bars = ax1.bar(metric_labels, values, color=colors, edgecolor="black", linewidth=1.5)
            ax1.axhline(y=0.1, color=COLORS["threshold"], linestyle="--", linewidth=2)
            ax1.axhline(y=-0.1, color=COLORS["threshold"], linestyle="--", linewidth=2)
            ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
            for bar, val in zip(bars, values):
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.3f}", ha="center", va="bottom" if val >= 0 else "top", fontweight="bold")
    ax1.set_title("Baseline Fairness Metrics", fontweight="bold", size=12)
    ax1.set_ylabel("Metric Value")

    # 2. Group Outcome Distribution (top-right)
    ax2 = fig.add_subplot(gs[0, 2])
    privileged_mask = protected_attr == privileged_value
    for i, (mask, name, color) in enumerate(
        [
            (~privileged_mask, group_names[0], COLORS["unprivileged"]),
            (privileged_mask, group_names[1], COLORS["privileged"]),
        ]
    ):
        pos_rate = np.mean(y_pred[mask])
        ax2.bar(i, pos_rate, color=color, edgecolor="black", linewidth=1.5)
        ax2.text(i, pos_rate + 0.02, f"{pos_rate:.1%}", ha="center", fontweight="bold")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(group_names)
    ax2.set_ylabel("Positive Outcome Rate")
    ax2.set_title("Predicted Positive Rate by Group", fontweight="bold", size=12)
    ax2.set_ylim(0, 1)

    # 3. Confusion Matrices (middle row, spans 2 columns)
    ax3_1 = fig.add_subplot(gs[1, 0])
    ax3_2 = fig.add_subplot(gs[1, 1])

    for ax, mask, name in [
        (ax3_1, ~privileged_mask, group_names[0]),
        (ax3_2, privileged_mask, group_names[1]),
    ]:
        cm = confusion_matrix(y_true[mask], y_pred[mask], labels=[0, 1])
        cm_normalized = cm.astype("float") / cm.sum() * 100
        sns.heatmap(cm_normalized, annot=False, cmap="Blues", ax=ax, cbar=False, square=True, linewidths=2, linecolor="white")
        for i in range(2):
            for j in range(2):
                text = f"{cm[i, j]}\n({cm_normalized[i, j]:.1f}%)"
                color = "white" if cm_normalized[i, j] > 50 else "black"
                ax.text(j + 0.5, i + 0.5, text, ha="center", va="center", fontsize=9, fontweight="bold", color=color)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticklabels(["Neg", "Pos"])
        ax.set_yticklabels(["Neg", "Pos"], rotation=0)
        ax.set_title(f"{name}", fontweight="bold", size=11)

    # 4. ROC Curves (middle-right)
    ax4 = fig.add_subplot(gs[1, 2])
    if y_proba is not None:
        for mask, name, color in [
            (~privileged_mask, group_names[0], COLORS["unprivileged"]),
            (privileged_mask, group_names[1], COLORS["privileged"]),
        ]:
            try:
                fpr, tpr, _ = roc_curve(y_true[mask], y_proba[mask])
                auc_score = np.trapz(tpr, fpr)
                ax4.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} ({auc_score:.2f})")
            except Exception:
                pass
        ax4.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
        ax4.set_xlabel("FPR")
        ax4.set_ylabel("TPR")
        ax4.legend(loc="lower right", fontsize=9)
    ax4.set_title("ROC Curves by Group", fontweight="bold", size=12)

    # 5. Mitigation Comparison (bottom row, spans 2 columns)
    ax5 = fig.add_subplot(gs[2, :2])
    if len(results) > 1:
        metrics_to_plot = ["accuracy", "statistical_parity_difference", "disparate_impact"]
        x = np.arange(len(metrics_to_plot))
        width = 0.8 / len(results)
        colors = plt.cm.Set2(np.linspace(0, 1, len(results)))

        for i, (result, color) in enumerate(zip(results, colors)):
            values = []
            for m in metrics_to_plot:
                val = result.get("metrics", {}).get("performance", {}).get(m)
                if val is None:
                    val = result.get("metrics", {}).get("fairness", {}).get(m, 0)
                values.append(val)

            offset = (i - len(results) / 2 + 0.5) * width
            mitigation = result.get("mitigation", "unknown").replace("_", " ").title()
            ax5.bar(x + offset, values, width, label=mitigation, color=color, edgecolor="black")

        ax5.set_xticks(x)
        ax5.set_xticklabels(["Accuracy", "SPD", "DI"])
        ax5.legend(loc="upper right", fontsize=9)
    ax5.set_title("Metrics Comparison Across Mitigations", fontweight="bold", size=12)
    ax5.set_ylabel("Value")

    # 6. Trade-off Summary (bottom-right)
    ax6 = fig.add_subplot(gs[2, 2])
    if len(results) >= 2:
        for i, result in enumerate(results):
            acc = result.get("metrics", {}).get("performance", {}).get("accuracy", 0)
            spd = abs(result.get("metrics", {}).get("fairness", {}).get("statistical_parity_difference", 0))
            mitigation = result.get("mitigation", "unknown").replace("_", " ").title()
            color = plt.cm.Set1(i / len(results))
            ax6.scatter(spd, acc, s=200, c=[color], label=mitigation, edgecolors="black", linewidths=2)

        ax6.axvline(x=0.1, color=COLORS["threshold"], linestyle="--", linewidth=2)
        ax6.axvspan(0, 0.1, alpha=0.1, color="green")
        ax6.set_xlabel("|Statistical Parity Diff|")
        ax6.set_ylabel("Accuracy")
        ax6.legend(loc="lower left", fontsize=9)
    ax6.set_title("Fairness-Accuracy Trade-off", fontweight="bold", size=12)

    fig.suptitle(title, fontweight="bold", size=16, y=0.98)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def generate_all_visualizations(
    results: List[Dict[str, Any]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    protected_attr: np.ndarray,
    privileged_value: int = 1,
    group_names: Tuple[str, str] = ("Unprivileged", "Privileged"),
    output_dir: Path = Path("reports/figures"),
) -> Dict[str, Path]:
    """Generate all visualization figures and save to output directory.

    Args:
        results: List of result dictionaries from audit runs.
        y_true: True labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities.
        protected_attr: Protected attribute values.
        privileged_value: Value indicating privileged group.
        group_names: Names for groups.
        output_dir: Directory to save figures.

    Returns:
        Dictionary mapping figure names to saved paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_figures = {}

    # 1. Fairness metrics bar chart
    if results:
        baseline_fairness = results[0].get("metrics", {}).get("fairness", {})
        if baseline_fairness:
            path = output_dir / "fairness_metrics.png"
            plot_fairness_metrics_bar(baseline_fairness, save_path=path)
            saved_figures["fairness_metrics"] = path
            plt.close()

    # 2. Fairness radar chart
    if results:
        baseline_fairness = results[0].get("metrics", {}).get("fairness", {})
        if baseline_fairness:
            path = output_dir / "fairness_radar.png"
            plot_fairness_radar(baseline_fairness, save_path=path)
            saved_figures["fairness_radar"] = path
            plt.close()

    # 3. Group metrics comparison
    if results:
        group_perf = results[0].get("metrics", {}).get("group_performance", {})
        if group_perf:
            path = output_dir / "group_comparison.png"
            plot_group_metrics_comparison(group_perf, save_path=path)
            saved_figures["group_comparison"] = path
            plt.close()

    # 4. Outcome distribution
    path = output_dir / "outcome_distribution.png"
    plot_outcome_distribution(y_true, y_pred, protected_attr, privileged_value, group_names, save_path=path)
    saved_figures["outcome_distribution"] = path
    plt.close()

    # 5. Mitigation comparison - fairness
    if len(results) > 1:
        path = output_dir / "mitigation_fairness.png"
        plot_mitigation_comparison(results, metric_type="fairness", save_path=path)
        saved_figures["mitigation_fairness"] = path
        plt.close()

        # 6. Mitigation comparison - performance
        path = output_dir / "mitigation_performance.png"
        plot_mitigation_comparison(results, metric_type="performance", save_path=path)
        saved_figures["mitigation_performance"] = path
        plt.close()

        # 7. Fairness-accuracy tradeoff
        path = output_dir / "tradeoff.png"
        plot_fairness_accuracy_tradeoff(results, save_path=path)
        saved_figures["tradeoff"] = path
        plt.close()

    # 8. Confusion matrices
    path = output_dir / "confusion_matrices.png"
    plot_confusion_matrices_by_group(y_true, y_pred, protected_attr, privileged_value, group_names, save_path=path)
    saved_figures["confusion_matrices"] = path
    plt.close()

    # 9. ROC curves
    if y_proba is not None:
        path = output_dir / "roc_curves.png"
        plot_roc_curves_by_group(y_true, y_proba, protected_attr, privileged_value, group_names, save_path=path)
        saved_figures["roc_curves"] = path
        plt.close()

    # 10. Score distribution
    if y_proba is not None:
        path = output_dir / "score_distribution.png"
        plot_score_distribution(y_proba, protected_attr, privileged_value, group_names, save_path=path)
        saved_figures["score_distribution"] = path
        plt.close()

    # 11. Full dashboard
    path = output_dir / "dashboard.png"
    create_audit_dashboard(
        results, y_true, y_pred, y_proba, protected_attr, privileged_value, group_names, save_path=path
    )
    saved_figures["dashboard"] = path
    plt.close()

    return saved_figures
