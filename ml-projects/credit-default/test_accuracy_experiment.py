"""Verify grouped evaluation, threshold selection, and experiment artifacts."""

import csv
import json
import unittest

import joblib
import numpy as np
from sklearn.metrics import accuracy_score

from data_source import load_data
from improve_accuracy import OUTPUT, accuracy_threshold, make_splits


class ThresholdTests(unittest.TestCase):
    def test_matches_exhaustive_search_with_ties(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            truth = rng.integers(0, 2, 30)
            probability = rng.integers(0, 11, 30) / 10
            cutoff = accuracy_threshold(truth, probability)
            candidates = np.r_[probability, 0.5, np.nextafter(probability.max(), np.inf)]
            expected = max(accuracy_score(truth, probability >= t) for t in candidates)
            self.assertEqual(accuracy_score(truth, probability >= cutoff), expected)

    def test_can_predict_no_defaults(self):
        truth, probability = np.array([0, 0, 1]), np.array([.95, .9, .85])
        self.assertGreater(accuracy_threshold(truth, probability), probability.max())


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, cls.ids, _ = load_data()
        cls.development, cls.test, cls.purged, cls.groups, cls.folds = make_splits(cls.X, cls.y)

    def test_no_profile_overlap_and_complete_partition(self):
        self.assertFalse(set(self.groups[self.development]) & set(self.groups[self.test]))
        partitions = [set(self.development), set(self.test), set(self.purged)]
        self.assertEqual(sum(map(len, partitions)), len(self.y))
        self.assertEqual(set.union(*partitions), set(range(len(self.y))))

    def test_each_cv_record_validated_once_without_profile_overlap(self):
        seen = np.zeros(len(self.development), dtype=int)
        for train, valid in self.folds:
            self.assertFalse(set(self.groups[self.development[train]]) & set(self.groups[self.development[valid]]))
            seen[valid] += 1
        np.testing.assert_array_equal(seen, 1)

    def test_saved_model_and_threshold_match_recorded_results(self):
        report = json.loads((OUTPUT / 'metrics.json').read_text())
        bundle = joblib.load(OUTPUT / 'model.joblib')
        with (OUTPUT / 'development_predictions.csv').open(newline='') as stream:
            oof = list(csv.DictReader(stream))
        self.assertEqual({int(row['source_id']) for row in oof}, set(self.ids[self.development]))
        cutoff = accuracy_threshold(np.array([int(row['observed_default']) for row in oof]),
                                    np.array([float(row['out_of_fold_probability']) for row in oof]))
        self.assertEqual(cutoff, bundle['threshold'])
        self.assertEqual(cutoff, report['threshold'])
        probability = bundle['model'].predict_proba(self.X[self.test])[:, 1]
        actual = accuracy_score(self.y[self.test], probability >= cutoff)
        self.assertEqual(actual, report['test']['selected_accuracy_threshold']['accuracy'])
        winner = max(report['candidates'], key=lambda item: item['cv_accuracy_mean'])
        self.assertEqual(winner['config']['name'], report['selected_model'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
