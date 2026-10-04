"""Synthetic end-to-end tests for the occurrence tournament."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.evaluation.backtest import (
    BACKTEST_YEARS,
    baseline_previous_year,
    baseline_recent_history,
    evaluate_predictions,
    occurrence_history,
    predict_occurrence,
    split_year,
)
from src.features.model_dataset import (
    MAIN_TARGET,
    OCCURRENCE_FEATURES,
    UNIT_KEY,
    build_model_dataset,
    known_leases,
    prediction_origin,
)
from src.models.model_b.experiments import (
    GROWTH_BASELINES,
    GROWTH_HIERARCHIES,
    GROWTH_SHRINKAGE_CANDIDATES,
    GROWTH_SUMMARY_COLUMNS,
    GROWTH_TOURNAMENT_COLUMNS,
    HISTORICAL_GLOBAL_BASELINE,
    OCCURRENCE_HIERARCHIES,
    OCCURRENCE_SHRINKAGE_CANDIDATES,
    OCCURRENCE_SUMMARY_COLUMNS,
    OCCURRENCE_TOURNAMENT_COLUMNS,
    run_growth_tournament,
    run_occurrence_tournament,
)
from src.models.model_b.growth import HierarchicalGrowthEstimator
from src.models.model_b.occurrence import HierarchicalOccurrenceEstimator


def synthetic_leases() -> pd.DataFrame:
    """Create a small point-in-time lease history without private records."""
    units = [
        ("P1", "001", "B1", "Quebec", {2023, 2024, 2025}),
        ("P1", "002", "B1", "Quebec", {2023, 2025}),
        ("P2", "003", "B2", "Ontario", {2024, 2025}),
    ]
    rows = []
    for prop, unit, building, province, transition_years in units:
        for year in range(2019, 2026):
            if year > 2022 and year not in transition_years:
                continue
            rows.append(
                {
                    "sPropCode": prop,
                    "sUnitCode": unit,
                    "sBuilding": building,
                    "sState": province,
                    "sBeds": 1,
                    "sBaths": 1,
                    "sSqft": 700,
                    "sFloor": 2,
                    "sUnitSubtype": "A",
                    "sLeaseFrom": f"{year}-01-01",
                    "sLeaseTo": f"{year + 1}-12-31",
                    "sSignDate": f"{year - 1}-12-15",
                    "sRent": 1000.0 * 1.03 ** (year - 2019),
                    "sRentEffective": 900.0 * 1.02 ** (year - 2019),
                    "sTermMonths": 12,
                    "sRenewal": 1,
                }
            )
    return pd.DataFrame(rows)


class OccurrenceTournamentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.leases = synthetic_leases()

    def test_result_and_summary_schemas_cover_fixed_tournament(self):
        results, summary = run_occurrence_tournament(self.leases)
        expected_model_rows = (
            len(OCCURRENCE_HIERARCHIES)
            * len(OCCURRENCE_SHRINKAGE_CANDIDATES)
        )

        self.assertEqual(tuple(results.columns), OCCURRENCE_TOURNAMENT_COLUMNS)
        self.assertEqual(tuple(summary.columns), OCCURRENCE_SUMMARY_COLUMNS)
        self.assertEqual(len(results), (expected_model_rows + 1) * len(BACKTEST_YEARS))
        self.assertEqual(
            set(results["year"]),
            {2023, 2024, 2025},
        )
        self.assertFalse(results["year"].eq(2026).any())
        self.assertTrue(
            results["candidate_count"].map(lambda count: count >= 0).all()
        )

        for year in BACKTEST_YEARS:
            fold = results.loc[results["year"].eq(year)]
            self.assertEqual(len(fold), expected_model_rows + 1)
            baseline = fold.loc[
                fold["hierarchy_name"].eq(HISTORICAL_GLOBAL_BASELINE)
            ]
            self.assertEqual(len(baseline), 1)
            self.assertTrue(baseline["shrinkage_strength"].isna().all())

    def test_each_shrinkage_candidate_is_fitted_independently(self):
        fitted_strengths = []
        original_fit = HierarchicalOccurrenceEstimator.fit

        def capture_fit(estimator, features, target):
            fitted_strengths.append(estimator.config.shrinkage_strength)
            return original_fit(estimator, features, target)

        with patch.object(HierarchicalOccurrenceEstimator, "fit", capture_fit):
            results, _ = run_occurrence_tournament(self.leases)

        self.assertEqual(
            len(fitted_strengths),
            len(BACKTEST_YEARS)
            * len(OCCURRENCE_HIERARCHIES)
            * len(OCCURRENCE_SHRINKAGE_CANDIDATES),
        )
        for offset in range(0, len(fitted_strengths), len(OCCURRENCE_SHRINKAGE_CANDIDATES)):
            self.assertEqual(
                fitted_strengths[offset : offset + len(OCCURRENCE_SHRINKAGE_CANDIDATES)],
                [float(value) for value in OCCURRENCE_SHRINKAGE_CANDIDATES],
            )
        model_rows = results.loc[
            ~results["hierarchy_name"].eq(HISTORICAL_GLOBAL_BASELINE)
        ]
        self.assertEqual(
            set(model_rows["shrinkage_strength"]),
            {float(value) for value in OCCURRENCE_SHRINKAGE_CANDIDATES},
        )

    def test_only_frozen_cutoff_qualified_history_is_used_for_training(self):
        fitted = []
        original_fit = HierarchicalOccurrenceEstimator.fit

        def capture_fit(estimator, features, target):
            fitted.append((features.copy(), target.copy()))
            return original_fit(estimator, features, target)

        with patch.object(HierarchicalOccurrenceEstimator, "fit", capture_fit):
            run_occurrence_tournament(self.leases)

        fits_per_fold = (
            len(OCCURRENCE_HIERARCHIES)
            * len(OCCURRENCE_SHRINKAGE_CANDIDATES)
        )
        for fold_index, year in enumerate(BACKTEST_YEARS):
            history = occurrence_history(self.leases, year)
            qualified = history["occurrence_target"].notna()
            expected_features = history.loc[
                qualified, list(OCCURRENCE_FEATURES)
            ]
            expected_target = history.loc[qualified, "occurrence_target"].astype(float)
            actual_features, actual_target = fitted[fold_index * fits_per_fold]
            assert_frame_equal(actual_features, expected_features)
            pd.testing.assert_series_equal(actual_target, expected_target)
            self.assertTrue(prediction_origin(year).year < year)

    def test_target_outcomes_are_revealed_after_all_predictions(self):
        events = []
        from src.models.model_b import experiments

        original_predict = HierarchicalOccurrenceEstimator.predict_proba
        original_baseline = predict_occurrence
        original_reveal = experiments.reveal_occurrence

        def capture_model_prediction(estimator, candidates):
            events.append("model_prediction")
            return original_predict(estimator, candidates)

        def capture_baseline(*args, **kwargs):
            events.append("baseline_prediction")
            return original_baseline(*args, **kwargs)

        def capture_target_reveal(*args, **kwargs):
            events.append("target_reveal")
            return original_reveal(*args, **kwargs)

        with (
            patch.object(
                HierarchicalOccurrenceEstimator,
                "predict_proba",
                capture_model_prediction,
            ),
            patch.object(experiments, "predict_occurrence", capture_baseline),
            patch.object(experiments, "reveal_occurrence", capture_target_reveal),
        ):
            run_occurrence_tournament(self.leases)

        first_reveal = events.index("target_reveal")
        self.assertTrue(all(event != "target_reveal" for event in events[:first_reveal]))
        self.assertTrue(
            all(event in {"model_prediction", "baseline_prediction"} for event in events[:first_reveal])
        )
        self.assertEqual(events.count("target_reveal"), len(BACKTEST_YEARS))

    def test_future_records_cannot_change_fold_predictions_or_training_counts(self):
        before, _ = run_occurrence_tournament(self.leases)
        changed = self.leases.copy()
        future = changed.loc[changed["sLeaseFrom"].eq("2025-01-01")].copy()
        future["sLeaseFrom"] = "2026-01-01"
        future["sLeaseTo"] = "2027-12-31"
        future["sSignDate"] = "2025-12-15"
        future["sRent"] = 999999.0
        future["sRentEffective"] = 888888.0
        future["sRenewal"] = 0
        changed = pd.concat([changed, future], ignore_index=True)

        after, _ = run_occurrence_tournament(changed)
        columns = [
            "year",
            "hierarchy_name",
            "shrinkage_strength",
            "candidate_count",
            "qualified_training_count",
            "predicted_transition_rate",
            "brier_score",
            "calibration_gap",
            "p1_predicted_percent",
        ]
        assert_frame_equal(before[columns], after[columns])

    def test_results_are_deterministic(self):
        first_results, first_summary = run_occurrence_tournament(self.leases)
        second_results, second_summary = run_occurrence_tournament(self.leases)

        assert_frame_equal(first_results, second_results)
        assert_frame_equal(first_summary, second_summary)

    def test_p1_and_occurrence_metrics_are_filled_when_computable(self):
        results, _ = run_occurrence_tournament(self.leases)

        self.assertTrue(results["brier_score"].notna().all())
        self.assertTrue(results["observed_transition_rate"].notna().all())
        self.assertTrue(results["predicted_transition_rate"].notna().all())
        self.assertTrue(results["p1_predicted_percent"].notna().all())
        self.assertTrue(results["p1_observed_percent"].notna().all())
        self.assertTrue(results["p1_error_pp"].notna().all())

    def test_frozen_historical_global_baseline_is_used(self):
        results, _ = run_occurrence_tournament(self.leases)
        for year in BACKTEST_YEARS:
            history = occurrence_history(self.leases, year)
            cutoff_candidates = history.loc[:, list(OCCURRENCE_FEATURES)]
            expected = float(
                predict_occurrence(
                    history, cutoff_candidates, method="historical_global"
                )[0]
            )
            row = results.loc[
                results["year"].eq(year)
                & results["hierarchy_name"].eq(HISTORICAL_GLOBAL_BASELINE)
            ].iloc[0]
            self.assertAlmostEqual(row["predicted_transition_rate"], expected)


class GrowthTournamentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.leases = synthetic_leases()

    def test_result_and_summary_schemas_cover_fixed_tournament(self):
        results, summary = run_growth_tournament(self.leases)
        expected_model_rows = (
            len(GROWTH_HIERARCHIES) * len(GROWTH_SHRINKAGE_CANDIDATES)
        )
        expected_configs = expected_model_rows + len(GROWTH_BASELINES)

        self.assertEqual(tuple(results.columns), GROWTH_TOURNAMENT_COLUMNS)
        self.assertEqual(tuple(summary.columns), GROWTH_SUMMARY_COLUMNS)
        self.assertEqual(len(results), expected_configs * len(BACKTEST_YEARS))
        self.assertEqual(len(summary), expected_configs)
        self.assertEqual(set(results["year"]), {2023, 2024, 2025})
        self.assertFalse(results["year"].eq(2026).any())
        self.assertFalse(
            results.duplicated(
                ["year", "hierarchy_name", "shrinkage_strength"]
            ).any()
        )

        for year in BACKTEST_YEARS:
            fold = results.loc[results["year"].eq(year)]
            self.assertEqual(len(fold), expected_configs)
            for baseline_name in (
                "previous_year_median_baseline",
                "rolling_historical_median_baseline",
            ):
                row = fold.loc[fold["hierarchy_name"].eq(baseline_name)]
                self.assertEqual(len(row), 1)
                self.assertTrue(row["shrinkage_strength"].isna().all())

    def test_each_hierarchy_and_shrinkage_is_fitted_independently(self):
        fitted = []
        original_fit = HierarchicalGrowthEstimator.fit

        def capture_fit(estimator, features, target):
            fitted.append(
                (
                    estimator.config.hierarchy,
                    estimator.config.shrinkage_strength,
                )
            )
            return original_fit(estimator, features, target)

        with patch.object(HierarchicalGrowthEstimator, "fit", capture_fit):
            results, _ = run_growth_tournament(self.leases)

        expected_per_fold = {
            (hierarchy, float(strength))
            for hierarchy in GROWTH_HIERARCHIES.values()
            for strength in GROWTH_SHRINKAGE_CANDIDATES
        }
        self.assertEqual(
            len(fitted), len(BACKTEST_YEARS) * len(expected_per_fold)
        )
        for fold_index in range(len(BACKTEST_YEARS)):
            start = fold_index * len(expected_per_fold)
            self.assertEqual(set(fitted[start : start + len(expected_per_fold)]), expected_per_fold)
        model_rows = results.loc[
            results["hierarchy_name"].isin(GROWTH_HIERARCHIES)
        ]
        self.assertEqual(
            len(model_rows),
            len(BACKTEST_YEARS)
            * len(GROWTH_HIERARCHIES)
            * len(GROWTH_SHRINKAGE_CANDIDATES),
        )

    def test_only_foundation_eligible_pre_cutoff_labels_enter_fitting(self):
        fitted = []
        original_fit = HierarchicalGrowthEstimator.fit

        def capture_fit(estimator, features, target):
            fitted.append((features.copy(), target.copy()))
            return original_fit(estimator, features, target)

        with patch.object(HierarchicalGrowthEstimator, "fit", capture_fit):
            run_growth_tournament(self.leases)

        fits_per_fold = (
            len(GROWTH_HIERARCHIES) * len(GROWTH_SHRINKAGE_CANDIDATES)
        )
        for fold_index, year in enumerate(BACKTEST_YEARS):
            historical, _ = build_model_dataset(
                known_leases(self.leases, prediction_origin(year))
            )
            train, _ = split_year(historical, year, MAIN_TARGET)
            self.assertTrue(train["year"].lt(year).all())
            self.assertTrue(
                pd.to_datetime(train["label_available_date"])
                .le(prediction_origin(year))
                .all()
            )
            for fit_index in range(fits_per_fold):
                actual_features, actual_target = fitted[
                    fold_index * fits_per_fold + fit_index
                ]
                self.assertEqual(len(actual_features), len(train))
                self.assertEqual(actual_target.name, MAIN_TARGET)
                np.testing.assert_allclose(
                    np.sort(actual_target.to_numpy(dtype=float)),
                    np.sort(train[MAIN_TARGET].to_numpy(dtype=float)),
                )

    def test_all_predictions_precede_target_evaluation(self):
        from src.models.model_b import experiments

        events = []
        original_predict = HierarchicalGrowthEstimator.predict
        original_evaluate_experiments = experiments.evaluate_predictions
        original_previous_baseline = experiments.baseline_previous_year
        original_rolling_baseline = experiments.baseline_recent_history

        def capture_prediction(estimator, candidates):
            events.append("prediction")
            return original_predict(estimator, candidates)

        def capture_previous_baseline(*args, **kwargs):
            events.append("prediction")
            return original_previous_baseline(*args, **kwargs)

        def capture_rolling_baseline(*args, **kwargs):
            events.append("prediction")
            return original_rolling_baseline(*args, **kwargs)

        def capture_experiment_evaluation(*args, **kwargs):
            events.append("target_evaluation")
            return original_evaluate_experiments(*args, **kwargs)

        with (
            patch.object(
                HierarchicalGrowthEstimator, "predict", capture_prediction
            ),
            patch.object(
                experiments,
                "baseline_previous_year",
                capture_previous_baseline,
            ),
            patch.object(
                experiments,
                "baseline_recent_history",
                capture_rolling_baseline,
            ),
            patch.object(
                experiments,
                "evaluate_predictions",
                capture_experiment_evaluation,
            ),
        ):
            run_growth_tournament(self.leases)

        first_evaluation = events.index("target_evaluation")
        expected_predictions = (
            len(BACKTEST_YEARS)
            * (
                len(GROWTH_HIERARCHIES)
                * len(GROWTH_SHRINKAGE_CANDIDATES)
                + len(GROWTH_BASELINES)
            )
        )
        self.assertEqual(events[:first_evaluation].count("prediction"), expected_predictions)
        self.assertFalse(
            any(event == "prediction" for event in events[first_evaluation:])
        )

    def test_frozen_baselines_are_reused(self):
        from src.models.model_b import experiments

        previous_calls = []
        rolling_calls = []
        original_previous = baseline_previous_year
        original_rolling = baseline_recent_history

        def capture_previous(train, year, target):
            previous_calls.append((year, target))
            return original_previous(train, year, target)

        def capture_rolling(train, year, target):
            rolling_calls.append((year, target))
            return original_rolling(train, year, target)

        with (
            patch.object(
                experiments, "baseline_previous_year", capture_previous
            ),
            patch.object(
                experiments, "baseline_recent_history", capture_rolling
            ),
        ):
            results, _ = run_growth_tournament(self.leases)

        expected_calls = [(year, MAIN_TARGET) for year in BACKTEST_YEARS]
        self.assertEqual(previous_calls, expected_calls)
        self.assertEqual(rolling_calls, expected_calls)
        for year in BACKTEST_YEARS:
            historical, _ = build_model_dataset(
                known_leases(self.leases, prediction_origin(year))
            )
            train, _ = split_year(historical, year, MAIN_TARGET)
            expected_previous = train.loc[
                train["year"].eq(year - 1), MAIN_TARGET
            ].median()
            expected_rolling = train.loc[
                train["year"].between(year - 3, year - 1), MAIN_TARGET
            ].median()
            for name, expected in (
                ("previous_year_median_baseline", expected_previous),
                ("rolling_historical_median_baseline", expected_rolling),
            ):
                row = results.loc[
                    results["year"].eq(year)
                    & results["hierarchy_name"].eq(name)
                ].iloc[0]
                if pd.isna(expected):
                    self.assertTrue(
                        pd.isna(row["predicted_median_growth_percent"])
                    )
                else:
                    self.assertAlmostEqual(
                        row["predicted_median_growth_percent"], 100 * expected
                    )
            model_fold = results.loc[
                results["year"].eq(year)
                & results["hierarchy_name"].eq("G0_global_only")
            ].iloc[0]
            for name in (
                "previous_year_median_baseline",
                "rolling_historical_median_baseline",
            ):
                baseline_fold = results.loc[
                    results["year"].eq(year)
                    & results["hierarchy_name"].eq(name)
                ].iloc[0]
                self.assertEqual(
                    baseline_fold["test_count"], model_fold["test_count"]
                )

    def test_results_are_deterministic(self):
        first_results, first_summary = run_growth_tournament(self.leases)
        second_results, second_summary = run_growth_tournament(self.leases)

        assert_frame_equal(first_results, second_results)
        assert_frame_equal(first_summary, second_summary)


if __name__ == "__main__":
    unittest.main()
