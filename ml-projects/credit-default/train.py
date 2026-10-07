"""Compare historical default models; select using validation, evaluate on test."""

import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
os.environ.setdefault("MPLCONFIGDIR", str(ARTIFACTS / ".matplotlib"))
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score,
                             brier_score_loss, confusion_matrix, fbeta_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_source import FEATURES, load_data


def preprocessing():
    # Status codes are categories, including undocumented -2 and 0 values.
    return ColumnTransformer([
        ("amounts", StandardScaler(), [0, *range(7, 19)]),
        ("status", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(range(1, 7))),
    ])


def score(y, probability, threshold):
    predicted = (probability >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "brier_score": float(brier_score_loss(y, probability)),
        "accuracy": float(accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f2": float(fbeta_score(y, predicted, beta=2, zero_division=0)),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
    }


def choose_threshold(y, probability):
    precision, recall, thresholds = precision_recall_curve(y, probability)
    # An illustrative recall-oriented objective; this is not an economic cost model.
    denominator = 4 * precision[:-1] + recall[:-1]
    f2 = np.divide(5 * precision[:-1] * recall[:-1], denominator,
                   out=np.zeros_like(denominator), where=denominator > 0)
    return float(thresholds[np.argmax(f2)])


def main():
    X, y, ids, provenance = load_data()
    indices = np.arange(len(y))
    development, test = train_test_split(indices, test_size=0.2, stratify=y, random_state=42)
    train, validation = train_test_split(
        development, test_size=0.25, stratify=y[development], random_state=42
    )
    candidates = {
        "baseline": DummyClassifier(strategy="prior"),
        "logistic_regression": make_pipeline(preprocessing(), LogisticRegression(max_iter=1000)),
        "gradient_boosting": make_pipeline(preprocessing(), HistGradientBoostingClassifier(
            max_iter=120, max_leaf_nodes=15, l2_regularization=1.0,
            early_stopping=False, random_state=42,
        )),
    }
    validation_metrics = {}
    for name, model in candidates.items():
        print(f"Training {name} on {len(train):,} records...", flush=True)
        model.fit(X[train], y[train])
        p = model.predict_proba(X[validation])[:, 1]
        validation_metrics[name] = score(y[validation], p, 0.5)
        print(f"  Validation average precision: {validation_metrics[name]['average_precision']:.3f}")

    # Select model and threshold entirely on validation data. Test remains untouched.
    winner = max(validation_metrics, key=lambda name: validation_metrics[name]["average_precision"])
    model = candidates[winner]
    threshold = choose_threshold(y[validation], model.predict_proba(X[validation])[:, 1])
    test_probability = model.predict_proba(X[test])[:, 1]
    baseline_probability = candidates["baseline"].predict_proba(X[test])[:, 1]
    test_default = score(y[test], test_probability, 0.5)
    test_selected = score(y[test], test_probability, threshold)
    report = {
        "dataset": provenance,
        "scikit_learn_version": sklearn.__version__,
        "rows": len(y), "feature_names": FEATURES, "overall_default_rate": float(y.mean()),
        "excluded_fields": ["ID", "SEX", "EDUCATION", "MARRIAGE", "AGE"],
        "split": {"method": "stratified random 60/20/20", "seed": 42,
                  "train": len(train), "validation": len(validation), "test": len(test)},
        "selection_metric": "validation average precision",
        "validation": validation_metrics,
        "selected_model": winner, "threshold_objective": "maximize validation F2",
        "test_at_0_5": test_default, "test_at_selected_threshold": test_selected,
        "test_baseline": score(y[test], baseline_probability, 0.5),
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    joblib.dump({"model": model, "features": FEATURES, "threshold": threshold,
                 "source_sha256": provenance["sha256"]}, ARTIFACTS / "model.joblib")
    split_ids = {name: ids[index].tolist() for name, index in
                 [("train", train), ("validation", validation), ("test", test)]}
    (ARTIFACTS / "split_ids.json").write_text(json.dumps(split_ids), encoding="utf-8")
    # A real held-out record for experimenting; no invented customer information.
    example = dict(zip(FEATURES, X[test[0]].tolist()))
    (ARTIFACTS / "example_customer.json").write_text(json.dumps(example, indent=2), encoding="utf-8")
    with (ARTIFACTS / "test_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["source_id", "observed_default", "model_probability", "predicted_default"])
        writer.writerows(zip(ids[test].tolist(), y[test].tolist(), test_probability.tolist(),
                             (test_probability >= threshold).astype(int).tolist()))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    for ax, cutoff, title in [(axes[0, 0], 0.5, "Default threshold: 0.50"),
                              (axes[0, 1], threshold, f"Validation-selected threshold: {threshold:.2f}")]:
        ConfusionMatrixDisplay.from_predictions(
            y[test], test_probability >= cutoff, display_labels=["No default", "Default"],
            cmap="Blues", colorbar=False, ax=ax,
        )
        ax.set_title(title)
    precision, recall, _ = precision_recall_curve(y[test], test_probability)
    axes[1, 0].plot(recall, precision, label=f"Model AP: {test_selected['average_precision']:.3f}")
    axes[1, 0].axhline(y[test].mean(), color="gray", linestyle="--", label="Default prevalence")
    axes[1, 0].set(xlabel="Recall: share of defaults caught", ylabel="Precision: share of flags correct",
                   title="Held-out precision-recall curve", xlim=(0, 1), ylim=(0, 1))
    axes[1, 0].legend()
    observed, predicted = calibration_curve(y[test], test_probability, n_bins=10, strategy="quantile")
    axes[1, 1].plot(predicted, observed, "o-", label="Model")
    axes[1, 1].plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    axes[1, 1].set(xlabel="Mean predicted probability", ylabel="Observed default rate",
                   title="Do probabilities match observed frequencies?", xlim=(0, 1), ylim=(0, 1))
    axes[1, 1].legend()
    fig.suptitle("Historical credit-card default | UCI Taiwan dataset | 6,000 test records", fontsize=14)
    fig.savefig(ARTIFACTS / "evaluation.png", dpi=150)
    plt.close(fig)

    print(f"\nSelected model: {winner}")
    print(f"Observed default rate: {y.mean():.2%}")
    print(f"Test ROC-AUC: {test_selected['roc_auc']:.3f} | Average precision: {test_selected['average_precision']:.3f}")
    for label, result in [("Threshold 0.50", test_default), (f"Threshold {threshold:.3f}", test_selected)]:
        tn, fp, fn, tp = np.array(result["confusion_matrix"]).ravel()
        print(f"{label}: precision {result['precision']:.1%}, recall {result['recall']:.1%}; "
              f"missed defaults {fn}, false alarms {fp}")
    print(f"Saved evaluation chart, metrics, and model to {ARTIFACTS}")


if __name__ == "__main__":
    main()
