"""Tests for the fixed-weight, aggregate-only backup ensemble wrapper."""

import inspect
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.models import ensemble_50_50 as ensemble
from src.models.model_b.experiments import (
    P1_COMBINATIONS,
    P1_GROWTH_CONFIG,
    P1_OCCURRENCE_CONFIG,
)


class FixedEnsembleTests(unittest.TestCase):
    def test_weights_are_exactly_fixed_at_one_half_each(self):
        self.assertEqual(dict(ensemble.ENSEMBLE_WEIGHTS), {"model_a": 0.5, "model_b": 0.5})
        with self.assertRaises(TypeError):
            ensemble.ENSEMBLE_WEIGHTS["model_a"] = 0.9

    def test_2026_ensemble_is_exact_arithmetic_mean_and_deterministic(self):
        forecast_a = 2.4436448124776575
        forecast_b = 2.924144
        expected = 0.5 * forecast_a + 0.5 * forecast_b

        with (
            patch.object(
                ensemble,
                "model_a_estimate_2026",
                return_value=forecast_a,
            ) as model_a,
            patch.object(
                ensemble,
                "_model_b_p1_g1_2026",
                return_value={"forecast": forecast_b, "candidate_count": 931},
            ),
        ):
            first = ensemble.estimate_2026_ensemble(pd.DataFrame(), asking=None)
            second = ensemble.estimate_2026_ensemble(pd.DataFrame(), asking=None)

        self.assertEqual(first["ensemble_50_50"], expected)
        self.assertEqual(first, second)
        self.assertEqual(first["weights"], {"model_a": 0.5, "model_b": 0.5})
        self.assertEqual(first["prediction_origin"], "2025-12-31")
        self.assertEqual(model_a.call_count, 2)
        self.assertTrue(
            all(np.isfinite(value) for value in (first["model_a"], first["model_b"], first["ensemble_50_50"]))
        )

    def test_non_finite_model_forecast_is_rejected(self):
        with (
            patch.object(ensemble, "model_a_estimate_2026", return_value=np.nan),
            patch.object(
                ensemble,
                "_model_b_p1_g1_2026",
                return_value={"forecast": 1.0, "candidate_count": 1},
            ),
        ):
            with self.assertRaisesRegex(ValueError, "Model A forecast must be finite"):
                ensemble.estimate_2026_ensemble(pd.DataFrame())

    def test_wrapper_has_no_weight_or_optimization_controls(self):
        parameters = inspect.signature(ensemble.estimate_2026_ensemble).parameters
        self.assertEqual(tuple(parameters), ("leases", "asking"))
        self.assertFalse(
            any("weight" in name or "optim" in name for name in parameters)
        )
        self.assertFalse(
            any("optim" in name or "tune" in name for name in dir(ensemble))
        )

    def test_forecast_result_is_aggregate_only(self):
        with (
            patch.object(ensemble, "model_a_estimate_2026", return_value=1.0),
            patch.object(
                ensemble,
                "_model_b_p1_g1_2026",
                return_value={"forecast": 2.0, "candidate_count": 3},
            ),
        ):
            result = ensemble.estimate_2026_ensemble(pd.DataFrame())

        self.assertTrue(
            set(result).isdisjoint(
                {
                    "probabilities",
                    "contributions",
                    "unit_key",
                    "sPropCode",
                    "sUnitCode",
                    "candidate_predictions",
                }
            )
        )
        self.assertEqual(
            set(result),
            {
                "model_a",
                "model_b",
                "ensemble_50_50",
                "weights",
                "forecast_year",
                "prediction_origin",
                "model_b_candidate_count",
            },
        )

    def test_backtest_uses_existing_a_and_b_results_and_fixed_average(self):
        model_a_results = {
            2023: {
                "prediction_origin": "2022-12-31",
                "candidate_units": 10,
                "predicted_P1_pct": 2.0,
                "realized_P1_pct": 1.0,
            },
            2024: {
                "prediction_origin": "2023-12-31",
                "candidate_units": 20,
                "predicted_P1_pct": 3.0,
                "realized_P1_pct": 2.0,
            },
            2025: {
                "prediction_origin": "2024-12-31",
                "candidate_units": 30,
                "predicted_P1_pct": 4.0,
                "realized_P1_pct": 3.0,
            },
        }
        b_rows = pd.DataFrame(
            {
                "year": [2023, 2024, 2025],
                "model_name": ["P1_G1"] * 3,
                "candidate_count": [10, 20, 30],
                "predicted_P1_percent": [1.0, 2.0, 3.0],
                "observed_P1_percent": [1.0, 2.0, 3.0],
            }
        )
        with (
            patch.object(
                ensemble,
                "model_a_backtest",
                side_effect=lambda leases, year: model_a_results[year],
            ) as model_a,
            patch.object(
                ensemble,
                "run_p1_combination_tournament",
                return_value=(b_rows, pd.DataFrame()),
            ) as model_b,
        ):
            folds, metrics = ensemble.backtest_ensemble_50_50(pd.DataFrame())

        self.assertEqual(model_a.call_count, 3)
        model_b.assert_called_once()
        self.assertEqual(
            folds["ensemble_50_50"].tolist(),
            [0.5 * 2.0 + 0.5 * 1.0, 0.5 * 3.0 + 0.5 * 2.0, 0.5 * 4.0 + 0.5 * 3.0],
        )
        self.assertEqual(folds["ensemble_error"].tolist(), [0.5, 0.5, 0.5])
        ensemble_metrics = metrics.loc[
            metrics["model"].eq("Ensemble 50/50")
        ].iloc[0]
        self.assertEqual(ensemble_metrics["MAE"], 0.5)
        self.assertEqual(ensemble_metrics["RMSE"], 0.5)
        self.assertEqual(ensemble_metrics["bias"], 0.5)
        self.assertEqual(ensemble_metrics["worst_absolute_error"], 0.5)

    def test_model_b_p1_g1_configuration_is_reused_unchanged(self):
        self.assertEqual(
            P1_COMBINATIONS["P1_G1"],
            ("hierarchical_province_expiry_month", "hierarchical_province"),
        )
        self.assertEqual(
            P1_OCCURRENCE_CONFIG.hierarchy,
            (("province",), ("province", "expiry_month")),
        )
        self.assertEqual(P1_OCCURRENCE_CONFIG.shrinkage_strength, 20.0)
        self.assertEqual(P1_GROWTH_CONFIG.hierarchy, (("province",),))
        self.assertEqual(P1_GROWTH_CONFIG.shrinkage_strength, 1.0)


if __name__ == "__main__":
    unittest.main()
