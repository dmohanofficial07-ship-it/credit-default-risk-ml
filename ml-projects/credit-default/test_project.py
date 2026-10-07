"""Checks for the completed research workflow. Run after train.py."""

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from data_source import FEATURES, ROOT, load_data


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, cls.ids, cls.provenance = load_data()
        cls.artifacts = ROOT / "artifacts"
        cls.bundle = joblib.load(cls.artifacts / "model.joblib")
        cls.splits = json.loads((cls.artifacts / "split_ids.json").read_text())

    def test_splits_are_disjoint_and_cover_source(self):
        train, validation, test = (set(self.splits[name]) for name in ["train", "validation", "test"])
        self.assertEqual((len(train), len(validation), len(test)), (18000, 6000, 6000))
        self.assertFalse(train & validation or train & test or validation & test)
        self.assertEqual(train | validation | test, set(self.ids.tolist()))

    def test_preprocessing_fitted_on_training_only(self):
        model = self.bundle["model"]
        if not hasattr(model, "steps"):
            self.skipTest("The selected baseline has no preprocessing.")
        scaler = model.steps[0][1].named_transformers_["amounts"]
        self.assertEqual(int(scaler.n_samples_seen_), 18000)
        mask = np.isin(self.ids, self.splits["train"])
        np.testing.assert_allclose(scaler.mean_, self.X[mask][:, [0, *range(7, 19)]].mean(axis=0))
        self.assertEqual(self.bundle["features"], FEATURES)

    def test_saved_predictions_match_model_and_source(self):
        with (self.artifacts / "test_predictions.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 6000)
        lookup = {int(identifier): i for i, identifier in enumerate(self.ids)}
        indices = [lookup[int(row["source_id"])] for row in rows]
        np.testing.assert_array_equal([int(row["observed_default"]) for row in rows], self.y[indices])
        expected = self.bundle["model"].predict_proba(self.X[indices])[:, 1]
        np.testing.assert_allclose([float(row["model_probability"]) for row in rows], expected)
        np.testing.assert_array_equal([int(row["predicted_default"]) for row in rows],
                                      expected >= self.bundle["threshold"])
        self.assertEqual(self.bundle["source_sha256"], self.provenance["sha256"])

    def test_prediction_cli_valid_and_invalid_input(self):
        script = str(ROOT / "predict.py")
        valid = subprocess.run([sys.executable, script], capture_output=True, text=True)
        self.assertEqual(valid.returncode, 0, valid.stderr)
        self.assertIn("default probability estimate", valid.stdout)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text('{"LIMIT_BAL": 10000}', encoding="utf-8")
            invalid = subprocess.run([sys.executable, script, "--input", str(path)], capture_output=True, text=True)
            self.assertEqual(invalid.returncode, 2)
            self.assertIn("exactly the 19 fields", invalid.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
