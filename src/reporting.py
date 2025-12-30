"""Report generation utilities for fairness audits."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def generate_metrics_csv(
    results: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Export metrics to CSV file.

    Args:
        results: List of result dictionaries from audit runs.
        output_path: Path for output CSV file.
    """
    rows = []

    for result in results:
        model_name = result.get("model", "unknown")
        mitigation = result.get("mitigation", "none")
        metrics = result.get("metrics", {})

        # Performance metrics
        for metric_name, value in metrics.get("performance", {}).items():
            rows.append(
                {
                    "model": model_name,
                    "mitigation": mitigation,
                    "category": "performance",
                    "metric": metric_name,
                    "value": value,
                }
            )

        # Fairness metrics
        for metric_name, value in metrics.get("fairness", {}).items():
            rows.append(
                {
                    "model": model_name,
                    "mitigation": mitigation,
                    "category": "fairness",
                    "metric": metric_name,
                    "value": value,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)


def format_metric_value(value: float, precision: int = 4) -> str:
    """Format metric value for display.

    Args:
        value: Metric value.
        precision: Decimal precision.

    Returns:
        Formatted string.
    """
    if pd.isna(value):
        return "N/A"
    return f"{value:.{precision}f}"


def generate_comparison_table(
    results: List[Dict[str, Any]],
    metric_names: Optional[List[str]] = None,
) -> str:
    """Generate markdown comparison table.

    Args:
        results: List of result dictionaries.
        metric_names: Optional list of metrics to include.

    Returns:
        Markdown table string.
    """
    if metric_names is None:
        metric_names = [
            "accuracy",
            "balanced_accuracy",
            "auc",
            "statistical_parity_difference",
            "disparate_impact",
            "equal_opportunity_difference",
            "average_odds_difference",
        ]

    # Build header
    headers = ["Mitigation"] + [m.replace("_", " ").title() for m in metric_names]
    separator = ["-" * len(h) for h in headers]

    header_row = "| " + " | ".join(headers) + " |"
    sep_row = "| " + " | ".join(separator) + " |"

    rows = [header_row, sep_row]

    # Build data rows
    for result in results:
        mitigation = result.get("mitigation", "none")
        metrics = result.get("metrics", {})

        row_values = [mitigation]
        for metric_name in metric_names:
            # Search in performance and fairness
            value = metrics.get("performance", {}).get(metric_name)
            if value is None:
                value = metrics.get("fairness", {}).get(metric_name)
            row_values.append(format_metric_value(value) if value is not None else "N/A")

        rows.append("| " + " | ".join(row_values) + " |")

    return "\n".join(rows)


def interpret_fairness_metrics(metrics: Dict[str, float]) -> List[str]:
    """Generate interpretations of fairness metrics.

    Args:
        metrics: Dictionary of fairness metrics.

    Returns:
        List of interpretation strings.
    """
    interpretations = []

    # Statistical Parity Difference
    spd = metrics.get("statistical_parity_difference", 0)
    if abs(spd) < 0.1:
        interpretations.append(
            f"- **Statistical Parity Difference** ({spd:.4f}): Near parity in positive outcome rates."
        )
    elif spd < -0.1:
        interpretations.append(
            f"- **Statistical Parity Difference** ({spd:.4f}): Unprivileged group receives "
            f"fewer positive outcomes. Consider mitigation."
        )
    else:
        interpretations.append(
            f"- **Statistical Parity Difference** ({spd:.4f}): Unprivileged group receives "
            f"more positive outcomes than privileged group."
        )

    # Disparate Impact
    di = metrics.get("disparate_impact", 1)
    if 0.8 <= di <= 1.25:
        interpretations.append(
            f"- **Disparate Impact** ({di:.4f}): Within acceptable range (0.8-1.25)."
        )
    elif di < 0.8:
        interpretations.append(
            f"- **Disparate Impact** ({di:.4f}): Below 0.8 threshold (80% rule). "
            f"Indicates potential adverse impact on unprivileged group."
        )
    else:
        interpretations.append(
            f"- **Disparate Impact** ({di:.4f}): Above 1.25, unprivileged group "
            f"disproportionately receives positive outcomes."
        )

    # Equal Opportunity Difference
    eod = metrics.get("equal_opportunity_difference", 0)
    if abs(eod) < 0.1:
        interpretations.append(
            f"- **Equal Opportunity Difference** ({eod:.4f}): Near equal true positive rates."
        )
    elif eod < -0.1:
        interpretations.append(
            f"- **Equal Opportunity Difference** ({eod:.4f}): Unprivileged group has lower "
            f"true positive rate. Qualified individuals may be denied."
        )
    else:
        interpretations.append(
            f"- **Equal Opportunity Difference** ({eod:.4f}): Unprivileged group has higher "
            f"true positive rate."
        )

    # Average Odds Difference
    aod = metrics.get("average_odds_difference", 0)
    if abs(aod) < 0.1:
        interpretations.append(f"- **Average Odds Difference** ({aod:.4f}): Near equalized odds.")
    else:
        interpretations.append(
            f"- **Average Odds Difference** ({aod:.4f}): Significant difference in "
            f"error rates between groups."
        )

    return interpretations


def generate_markdown_report(
    results: List[Dict[str, Any]],
    config: Dict[str, Any],
    output_path: Path,
) -> None:
    """Generate full markdown report.

    Args:
        results: List of result dictionaries from audit runs.
        config: Audit configuration dictionary.
        output_path: Path for output markdown file.
    """
    lines = []

    # Header
    lines.append("# Credit Bias Audit Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Configuration
    lines.append("## Audit Configuration")
    lines.append("")
    lines.append(f"- **Dataset:** {config.get('dataset', 'N/A')}")
    lines.append(f"- **Protected Attribute:** {config.get('protected_attr', 'N/A')}")
    lines.append(f"- **Model:** {config.get('model', 'N/A')}")
    lines.append(f"- **Random Seed:** {config.get('seed', 'N/A')}")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append("This report presents the results of a fairness audit on a credit risk model. ")
    lines.append("We evaluate both predictive performance and fairness metrics, comparing ")
    lines.append("baseline results with bias-mitigated versions.")
    lines.append("")

    # Results comparison table
    lines.append("## Results Comparison")
    lines.append("")
    lines.append(generate_comparison_table(results))
    lines.append("")

    # Detailed analysis by mitigation
    lines.append("## Detailed Analysis")
    lines.append("")

    for result in results:
        mitigation = result.get("mitigation", "none")
        metrics = result.get("metrics", {})

        lines.append(f"### {mitigation.replace('_', ' ').title()}")
        lines.append("")

        # Performance
        lines.append("**Performance Metrics:**")
        lines.append("")
        perf = metrics.get("performance", {})
        lines.append(f"- Accuracy: {format_metric_value(perf.get('accuracy'))}")
        lines.append(f"- Balanced Accuracy: {format_metric_value(perf.get('balanced_accuracy'))}")
        lines.append(f"- AUC: {format_metric_value(perf.get('auc'))}")
        lines.append(f"- F1 Score: {format_metric_value(perf.get('f1'))}")
        lines.append("")

        # Fairness
        lines.append("**Fairness Metrics:**")
        lines.append("")
        fairness = metrics.get("fairness", {})
        lines.extend(interpret_fairness_metrics(fairness))
        lines.append("")

    # Trade-off analysis
    lines.append("## Trade-off Analysis")
    lines.append("")

    if len(results) >= 2:
        baseline = results[0]
        mitigated = results[-1]

        baseline_acc = baseline.get("metrics", {}).get("performance", {}).get("accuracy", 0)
        mitigated_acc = mitigated.get("metrics", {}).get("performance", {}).get("accuracy", 0)
        acc_diff = mitigated_acc - baseline_acc

        baseline_spd = abs(
            baseline.get("metrics", {}).get("fairness", {}).get("statistical_parity_difference", 0)
        )
        mitigated_spd = abs(
            mitigated.get("metrics", {}).get("fairness", {}).get("statistical_parity_difference", 0)
        )
        spd_improvement = baseline_spd - mitigated_spd

        lines.append(f"Comparing baseline to {mitigated.get('mitigation', 'mitigated')}:")
        lines.append("")
        lines.append(f"- **Accuracy change:** {acc_diff:+.4f} ({acc_diff * 100:+.2f}%)")
        lines.append(f"- **SPD improvement:** {spd_improvement:.4f} (closer to 0 is better)")
        lines.append("")

        if acc_diff < -0.05 and spd_improvement > 0.1:
            lines.append(
                "The mitigation significantly improved fairness but at a notable cost to accuracy. "
            )
            lines.append("Consider whether this trade-off is acceptable for your use case.")
        elif acc_diff >= -0.02 and spd_improvement > 0.05:
            lines.append("The mitigation improved fairness with minimal impact on accuracy. ")
            lines.append("This represents a favorable trade-off.")
        else:
            lines.append("The trade-off between fairness and performance should be evaluated ")
            lines.append(
                "in the context of specific business requirements and regulatory constraints."
            )

    lines.append("")

    # Limitations
    lines.append("## Limitations")
    lines.append("")
    lines.append("- This audit focuses on a single protected attribute. Real-world scenarios may ")
    lines.append("  require intersectional analysis across multiple attributes.")
    lines.append("- Fairness metrics capture different aspects of fairness; no single metric ")
    lines.append("  provides a complete picture.")
    lines.append("- Post-processing mitigations may not generalize well to new data distributions.")
    lines.append("- The analysis assumes the protected attribute is accurately recorded.")
    lines.append("")

    # Recommendations
    lines.append("## Recommendations")
    lines.append("")
    lines.append("1. **Review business context:** Consider which fairness definition aligns ")
    lines.append("   with organizational values and regulatory requirements.")
    lines.append("2. **Monitor over time:** Fairness metrics should be tracked continuously ")
    lines.append("   as model and data evolve.")
    lines.append("3. **Stakeholder input:** Engage affected communities in defining acceptable ")
    lines.append("   trade-offs between fairness and performance.")
    lines.append("4. **Document decisions:** Maintain records of fairness audits and mitigation ")
    lines.append("   choices for accountability.")
    lines.append("")

    # Write file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def create_report_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create a summary dictionary from results.

    Args:
        results: List of result dictionaries.

    Returns:
        Summary dictionary.
    """
    summary = {
        "n_experiments": len(results),
        "mitigations_tested": [r.get("mitigation") for r in results],
    }

    if results:
        # Best fairness (closest SPD to 0)
        best_fairness = min(
            results,
            key=lambda r: abs(
                r.get("metrics", {})
                .get("fairness", {})
                .get("statistical_parity_difference", float("inf"))
            ),
        )
        summary["best_fairness_mitigation"] = best_fairness.get("mitigation")

        # Best accuracy
        best_accuracy = max(
            results,
            key=lambda r: r.get("metrics", {}).get("performance", {}).get("accuracy", 0),
        )
        summary["best_accuracy_mitigation"] = best_accuracy.get("mitigation")

    return summary
