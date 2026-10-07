"""Score a JSON record using the saved historical research model."""

import argparse
import json
import math
from pathlib import Path

import joblib

from data_source import FEATURES, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "artifacts" / "example_customer.json")
    args = parser.parse_args()
    model_path = ROOT / "artifacts" / "model.joblib"
    if not model_path.exists():
        parser.error("Run train.py first.")
    try:
        record = json.loads(args.input.read_text(encoding="utf-8"))
        if not isinstance(record, dict) or set(record) != set(FEATURES):
            raise ValueError("Input must contain exactly the 19 fields in example_customer.json.")
        if any(isinstance(record[name], bool) or not isinstance(record[name], (int, float)) for name in FEATURES):
            raise ValueError("Every feature must be a numeric value.")
        row = [float(record[name]) for name in FEATURES]
        if not all(math.isfinite(value) for value in row):
            raise ValueError("All measurements must be finite.")
        if row[0] <= 0 or any(value < 0 for value in row[13:]):
            raise ValueError("Credit limit must be positive; payment amounts cannot be negative.")
        if any(value != int(value) or not -2 <= value <= 9 for value in row[1:7]):
            raise ValueError("Repayment status must be an integer code between -2 and 9.")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    # This is the trusted artifact produced by this project's training script.
    bundle = joblib.load(model_path)
    probability = float(bundle["model"].predict_proba([row])[0, 1])
    print(f"Historical-model default probability estimate: {probability:.1%}")
    print(f"Validation-selected threshold: {bundle['threshold']:.3f}")
    print(f"Predicted label: {int(probability >= bundle['threshold'])} (1 = default, 0 = no default)")
    print("Research example using 2005 Taiwan data; this is not a lending decision.")


if __name__ == "__main__":
    main()
