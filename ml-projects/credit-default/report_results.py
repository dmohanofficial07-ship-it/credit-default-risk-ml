"""Display saved evaluation results without retraining the model."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def format_results(report):
    """Keep accuracy, ranking metrics, and threshold tradeoffs clearly labeled."""
    test_size = report["split"]["test"]
    rows = [
        ("Baseline", report["test_baseline"]),
        ("Standard threshold", report["test_at_0_5"]),
        ("Recall-focused threshold", report["test_at_selected_threshold"]),
    ]
    lines = [
        "CREDIT DEFAULT: SAVED TEST RESULTS",
        f"Model: {report['selected_model']} | Test records: {test_size:,}",
        f"Model selection: {report['selection_metric']}",
        f"Threshold selection: {report['threshold_objective']}",
        "",
        f"{'Setting':26} {'Cutoff':>7} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'Missed':>8} {'False +':>8}",
        "-" * 95,
    ]
    for label, metrics in rows:
        tn, fp = metrics["confusion_matrix"][0]
        fn, tp = metrics["confusion_matrix"][1]
        if tn + fp + fn + tp != test_size:
            raise ValueError(f"Confusion matrix count differs from test size for {label}.")
        lines.append(
            f"{label:26} {metrics['threshold']:7.3f} {metrics['accuracy']:9.2%} "
            f"{metrics['precision']:9.2%} {metrics['recall']:9.2%} {fn:8,d} {fp:8,d}"
        )
    standard = report["test_at_0_5"]
    selected = report["test_at_selected_threshold"]
    lines.extend([
        "",
        f"Standard threshold: {sum(standard['confusion_matrix'][i][i] for i in (0, 1)):,} "
        f"of {test_size:,} labels correct.",
        f"Recall-focused threshold: {selected['confusion_matrix'][1][1]:,} "
        f"of {sum(selected['confusion_matrix'][1]):,} actual defaults caught.",
        "",
        f"ROC-AUC: {standard['roc_auc']:.3f} | Average precision: {standard['average_precision']:.3f}",
        f"Brier score: {standard['brier_score']:.4f} (lower is better)",
        "ROC-AUC and average precision are ranking measures, not accuracy percentages.",
        "Missed = actual defaults predicted as no default; False + = non-defaults flagged as default.",
        "These results describe this historical test cohort, not present-day lending performance.",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=ROOT / "artifacts" / "metrics.json")
    args = parser.parse_args()
    try:
        report = json.loads(args.metrics.read_text(encoding="utf-8"))
        summary = format_results(report)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        parser.error(f"Cannot read evaluation results: {exc}. Run train.py to generate them.")
    print(summary)
    print(f"\nMetrics: {args.metrics.resolve()}")


if __name__ == "__main__":
    main()
