import unittest

import numpy as np

from financial_features import FinancialFeatures


class FinancialFeatureTests(unittest.TestCase):
    def setUp(self):
        self.row = np.array([[2000, 2, 1, 0, -1, -2, 0,
                              1000, 900, 800, 700, 600, 500,
                              100, 80, 60, 40, 20, 0]], dtype=float)

    def test_known_financial_history(self):
        transformer = FinancialFeatures().fit(self.row)
        result = transformer.transform(self.row)
        named = dict(zip(transformer.get_feature_names_out(), result[0]))
        self.assertEqual(result.shape, (1, 36))
        self.assertEqual(named['bill_to_limit_1'], 0.5)
        self.assertEqual(named['late_month_count'], 2)
        self.assertEqual(named['worst_delay'], 2)
        self.assertEqual(named['recent_delay_change'], 1)
        self.assertEqual(named['mean_payment'], 50)
        self.assertEqual(named['recent_payment_change'], 20)
        self.assertEqual(named['recent_bill_change'], 100)
        self.assertEqual(named['bill_change_over_limit'], 0.25)
        self.assertEqual(named['total_payments_over_limit'], 0.15)

    def test_row_independence_and_input_preservation(self):
        original = self.row.copy()
        other = self.row.copy()
        other[:, 7:] *= 100
        transformer = FinancialFeatures().fit(np.vstack([self.row, other]))
        alone = transformer.transform(self.row)
        together = transformer.transform(np.vstack([self.row, other]))[:1]
        np.testing.assert_array_equal(alone, together)
        np.testing.assert_array_equal(self.row, original)
        np.testing.assert_array_equal(alone[:, :19], self.row)

    def test_negative_bills_are_preserved(self):
        self.row[0, 7] = -100
        result = FinancialFeatures().fit_transform(self.row)
        self.assertEqual(result[0, 7], -100)
        self.assertEqual(result[0, 19], -0.05)

    def test_invalid_limits_fail(self):
        for limit in [0, -1]:
            self.row[0, 0] = limit
            with self.assertRaisesRegex(ValueError, 'positive'):
                FinancialFeatures().fit_transform(self.row)


if __name__ == '__main__':
    unittest.main(verbosity=2)
