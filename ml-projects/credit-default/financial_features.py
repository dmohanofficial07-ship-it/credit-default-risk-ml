"""Financial-history features derived independently for each record."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

from data_source import FEATURES

DERIVED_NAMES = [
    *[f"bill_to_limit_{i}" for i in range(1, 7)],
    "late_month_count", "worst_delay", "recent_delay_change",
    "mean_payment", "std_payment", "recent_payment_change",
    "mean_bill", "std_bill", "recent_bill_change",
    "bill_change_over_limit", "total_payments_over_limit",
]


class FinancialFeatures(TransformerMixin, BaseEstimator):
    """Append 17 features without using labels or statistics from other records."""

    def fit(self, X, y=None):
        values = self._validate(X)
        self.n_features_in_ = values.shape[1]
        return self

    @staticmethod
    def _validate(X):
        values = check_array(X, dtype=float)
        if values.shape[1] != len(FEATURES):
            raise ValueError(f"Expected {len(FEATURES)} financial input fields.")
        if np.any(values[:, 0] <= 0):
            raise ValueError("Credit limits must be positive to calculate utilization ratios.")
        return values

    def transform(self, X):
        check_is_fitted(self, "n_features_in_")
        values = self._validate(X)
        limit = values[:, 0]
        delay = np.maximum(values[:, 1:7], 0)
        bills, payments = values[:, 7:13], values[:, 13:19]
        changes = np.column_stack([
            bills / limit[:, None],
            np.count_nonzero(delay, axis=1), delay.max(axis=1), delay[:, 0] - delay[:, 1],
            payments.mean(axis=1), payments.std(axis=1), payments[:, 0] - payments[:, 1],
            bills.mean(axis=1), bills.std(axis=1), bills[:, 0] - bills[:, 1],
            (bills[:, 0] - bills[:, -1]) / limit,
            payments.sum(axis=1) / limit,
        ])
        return np.column_stack([values, changes])

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "n_features_in_")
        return np.array([*FEATURES, *DERIVED_NAMES], dtype=object)
