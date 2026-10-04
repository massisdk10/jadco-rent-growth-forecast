"""Synthetic contract tests for Model B conditional-growth interfaces."""

import unittest

import pandas as pd

from src.models.model_b.growth import GrowthConfig, HierarchicalGrowthEstimator


class GrowthInterfaceTests(unittest.TestCase):
    def test_public_interface_and_default_configuration(self):
        config = GrowthConfig()
        estimator = HierarchicalGrowthEstimator(config)

        self.assertEqual(estimator.config, config)
        self.assertTrue(callable(estimator.fit))
        self.assertTrue(callable(estimator.predict))

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            GrowthConfig(shrinkage_strength=float("inf"))
        with self.assertRaises(ValueError):
            GrowthConfig(minimum_segment_size=-1)
        with self.assertRaises(ValueError):
            GrowthConfig(hierarchy=(("province",), ("property_code",)))
        with self.assertRaises(ValueError):
            GrowthConfig(hierarchy=(("old_sRentEffective",),))

    def test_prediction_interface_rejects_non_allowlisted_fields(self):
        estimator = HierarchicalGrowthEstimator()
        candidates = pd.DataFrame({"old_sRentEffective": [1000.0]})

        with self.assertRaises(ValueError):
            estimator.predict(candidates)

    def test_estimation_is_an_explicit_placeholder(self):
        estimator = HierarchicalGrowthEstimator()
        features = pd.DataFrame({"province": ["Ontario"]})
        labels = pd.Series([0.03], index=features.index)

        with self.assertRaises(NotImplementedError):
            estimator.fit(features, labels)


if __name__ == "__main__":
    unittest.main()