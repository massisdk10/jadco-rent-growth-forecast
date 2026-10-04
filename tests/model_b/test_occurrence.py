"""Synthetic contract tests for Model B occurrence interfaces."""

import unittest

import pandas as pd

from src.models.model_b.occurrence import (
    HierarchicalOccurrenceEstimator,
    OccurrenceConfig,
)


class OccurrenceInterfaceTests(unittest.TestCase):
    def test_public_interface_and_default_configuration(self):
        config = OccurrenceConfig()
        estimator = HierarchicalOccurrenceEstimator(config)

        self.assertEqual(estimator.config, config)
        self.assertTrue(callable(estimator.fit))
        self.assertTrue(callable(estimator.predict_proba))

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            OccurrenceConfig(shrinkage_strength=0)
        with self.assertRaises(ValueError):
            OccurrenceConfig(minimum_segment_size=0)
        with self.assertRaises(ValueError):
            OccurrenceConfig(hierarchy=(("province",), ("building",)))
        with self.assertRaises(ValueError):
            OccurrenceConfig(hierarchy=(("new_sRenewal",),))

    def test_prediction_interface_rejects_non_allowlisted_fields(self):
        estimator = HierarchicalOccurrenceEstimator()
        candidates = pd.DataFrame({"new_sRenewal": [1]})

        with self.assertRaises(ValueError):
            estimator.predict_proba(candidates)

    def test_estimation_is_an_explicit_placeholder(self):
        estimator = HierarchicalOccurrenceEstimator()
        features = pd.DataFrame({"province": ["Quebec"]})
        labels = pd.Series([1.0], index=features.index)

        with self.assertRaises(NotImplementedError):
            estimator.fit(features, labels)


if __name__ == "__main__":
    unittest.main()