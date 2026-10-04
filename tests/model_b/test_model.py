"""Synthetic tests for Model B output assembly and leakage boundaries."""

import unittest

import numpy as np
import pandas as pd

from src.models.model_b.model import (
    PREDICTION_COLUMNS,
    ModelB,
    aggregate_p1,
    assemble_predictions,
)
from src.models.model_b.growth import HierarchicalGrowthEstimator
from src.models.model_b.occurrence import HierarchicalOccurrenceEstimator


class ModelBTests(unittest.TestCase):
    def test_p1_is_median_of_rowwise_probability_growth_products(self):
        predictions = assemble_predictions([0.0, 0.9, 0.4], [0.2, 0.03, 0.1])

        np.testing.assert_allclose(
            predictions["contribution_i"], [0.0, 0.027, 0.04]
        )
        self.assertAlmostEqual(aggregate_p1(predictions), 0.027)
        self.assertNotAlmostEqual(
            aggregate_p1(predictions),
            float(predictions["p_i"].median() * predictions["g_i"].median()),
        )

    def test_wrapper_combines_component_predictions(self):
        class OccurrenceStub(HierarchicalOccurrenceEstimator):
            def predict_proba(self, candidates: pd.DataFrame) -> np.ndarray:
                return np.array([0.5, 0.25])

        class GrowthStub(HierarchicalGrowthEstimator):
            def predict(self, candidates: pd.DataFrame) -> np.ndarray:
                return np.array([0.04, 0.08])

        index = pd.Index(["unit-a", "unit-b"])
        features = pd.DataFrame({"province": ["Quebec", "Ontario"]}, index=index)
        model = ModelB(OccurrenceStub(), GrowthStub())

        predictions = model.predict(features, features)

        self.assertEqual(tuple(predictions.columns), PREDICTION_COLUMNS)
        np.testing.assert_allclose(predictions["contribution_i"], [0.02, 0.02])
        self.assertEqual(predictions.index.tolist(), index.tolist())

    def test_prediction_lengths_and_domains_are_validated(self):
        with self.assertRaises(ValueError):
            assemble_predictions([0.2, 0.3], [0.01])
        with self.assertRaises(ValueError):
            assemble_predictions([1.1], [0.02])
        with self.assertRaises(ValueError):
            assemble_predictions([0.5], [np.inf])
        with self.assertRaises(ValueError):
            assemble_predictions([0.5], [0.02], index=pd.RangeIndex(2))

    def test_prediction_output_contains_no_feature_or_future_fields(self):
        predictions = assemble_predictions(
            [0.4], [0.025], index=pd.Index(["synthetic-row"])
        )

        self.assertEqual(tuple(predictions.columns), PREDICTION_COLUMNS)
        self.assertEqual(predictions.index.tolist(), ["synthetic-row"])
        self.assertTrue(
            set(predictions.columns).isdisjoint(
                {"new_sRent", "new_sRenewal", "occurrence_target", "unit_growth"}
            )
        )

    def test_candidate_matrices_reject_future_information(self):
        model = ModelB()
        occurrence = pd.DataFrame(
            {"new_sRenewal": [1]}, index=pd.Index(["synthetic-row"])
        )
        growth = pd.DataFrame(
            {"province": ["Quebec"]}, index=pd.Index(["synthetic-row"])
        )

        with self.assertRaises(ValueError):
            model.predict(occurrence, growth)

    def test_candidate_matrices_must_be_aligned(self):
        model = ModelB()
        occurrence = pd.DataFrame({"province": ["Quebec"]}, index=[0])
        growth = pd.DataFrame({"province": ["Quebec"]}, index=[1])

        with self.assertRaises(ValueError):
            model.predict(occurrence, growth)

    def test_fitting_remains_unimplemented(self):
        model = ModelB()
        empty = pd.DataFrame()
        labels = pd.Series(dtype=float)

        with self.assertRaises(NotImplementedError):
            model.fit(empty, labels, empty, labels)


if __name__ == "__main__":
    unittest.main()