"""Fixed-weight 50/50 backup ensemble for the finalized Model A and B APIs."""

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np
import pandas as pd

from src.evaluation.backtest import (
    BACKTEST_YEARS,
    occurrence_history,
    split_year,
)
from src.features.model_dataset import (
    MAIN_TARGET,
    UNIT_KEY,
    build_features,
    build_model_dataset,
    build_occurrence_features,
    build_prediction_cohort,
    known_leases,
    prediction_origin,
)
from src.models.model_a import backtest as model_a_backtest
from src.models.model_a import estimate_2026 as model_a_estimate_2026
from src.models.model_b.experiments import (
    P1_COMBINATIONS,
    P1_GROWTH_CONFIG,
    P1_OCCURRENCE_CONFIG,
    _p1_growth_training,
    _qualified_training,
    run_p1_combination_tournament,
)
from src.models.model_b.growth import HierarchicalGrowthEstimator
from src.models.model_b.model import ModelB, aggregate_p1
from src.models.model_b.occurrence import HierarchicalOccurrenceEstimator


ENSEMBLE_WEIGHTS: Mapping[str, float] = MappingProxyType(
    {"model_a": 0.5, "model_b": 0.5}
)
ENSEMBLE_YEARS = BACKTEST_YEARS
_MODEL_B_COMBINATION = "P1_G1"


def _finite_scalar(value: object, name: str) -> float:
    """Convert one forecast to a finite scalar percentage."""
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} forecast must be finite.")
    return result


def _model_b_p1_g1_2026(leases: pd.DataFrame) -> dict[str, float | int]:
    """Apply the existing P1_G1 estimators at the frozen 2026 prediction origin."""
    if P1_COMBINATIONS.get(_MODEL_B_COMBINATION) != (
        "hierarchical_province_expiry_month",
        "hierarchical_province",
    ):
        raise RuntimeError("The frozen Model B P1_G1 definition has changed.")

    origin = prediction_origin(2026)
    cutoff_leases = known_leases(leases, origin)
    occurrence_history_rows = occurrence_history(cutoff_leases, 2026)
    occurrence_x, occurrence_y = _qualified_training(occurrence_history_rows)
    if occurrence_y.empty:
        raise ValueError(
            f"No qualified Model B occurrence labels available at {origin.date()}."
        )

    historical_pairs, _ = build_model_dataset(cutoff_leases)
    growth_train, _ = split_year(historical_pairs, 2026, MAIN_TARGET)
    if growth_train.empty:
        raise ValueError(
            f"No eligible Model B growth labels available at {origin.date()}."
        )

    cohort = build_prediction_cohort(cutoff_leases, 2026)
    if cohort.empty:
        raise ValueError("The frozen 2026 candidate cohort is empty.")
    candidate_index = pd.MultiIndex.from_frame(cohort.loc[:, list(UNIT_KEY)])

    occurrence_candidates = build_occurrence_features(
        cutoff_leases, cohort, 2026
    ).loc[:, ["province", "expiry_month"]]
    occurrence_candidates.index = candidate_index
    growth_candidates = build_features(
        cutoff_leases, cohort, 2026, features=("province",)
    )
    growth_candidates.index = candidate_index

    growth_training = _p1_growth_training(
        cutoff_leases, growth_train, 2026
    )
    growth_features = growth_training.loc[:, ["province"]]
    growth_target = growth_training[MAIN_TARGET].rename(MAIN_TARGET)

    model = ModelB(
        HierarchicalOccurrenceEstimator(P1_OCCURRENCE_CONFIG),
        HierarchicalGrowthEstimator(P1_GROWTH_CONFIG),
    )
    model.fit(occurrence_x, occurrence_y, growth_features, growth_target)
    predictions = model.predict(occurrence_candidates, growth_candidates)
    if len(predictions) != len(cohort):
        raise RuntimeError("Model B must return one prediction per candidate unit.")

    return {
        "forecast": 100.0 * aggregate_p1(predictions),
        "candidate_count": len(cohort),
    }


def estimate_2026_ensemble(
    leases: pd.DataFrame, asking: pd.DataFrame | None = None
) -> dict[str, object]:
    """Return aggregate A, B P1_G1, and fixed 50/50 2026 forecasts.

    The Model A API is called with its established ``asking`` argument and no
    external data. Model B reuses its frozen P1_G1 configurations and
    cutoff-aware foundation helpers. No candidate-level records are returned.
    """
    forecast_a = _finite_scalar(
        model_a_estimate_2026(leases, asking, external=None),
        "Model A",
    )
    model_b = _model_b_p1_g1_2026(leases)
    forecast_b = _finite_scalar(model_b["forecast"], "Model B P1_G1")
    ensemble = (
        ENSEMBLE_WEIGHTS["model_a"] * forecast_a
        + ENSEMBLE_WEIGHTS["model_b"] * forecast_b
    )
    return {
        "model_a": forecast_a,
        "model_b": forecast_b,
        "ensemble_50_50": ensemble,
        "weights": dict(ENSEMBLE_WEIGHTS),
        "forecast_year": 2026,
        "prediction_origin": "2025-12-31",
        "model_b_candidate_count": int(model_b["candidate_count"]),
    }


def backtest_ensemble_50_50(
    leases: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare A, B P1_G1, and their fixed 50/50 ensemble on 2023-2025."""
    model_a_rows = {
        year: model_a_backtest(leases, year) for year in ENSEMBLE_YEARS
    }
    model_b_results, _ = run_p1_combination_tournament(leases)
    model_b_rows = model_b_results.loc[
        model_b_results["model_name"].eq(_MODEL_B_COMBINATION)
    ].set_index("year")

    rows: list[dict[str, float | int]] = []
    for year in ENSEMBLE_YEARS:
        result_a = model_a_rows[year]
        if year not in model_b_rows.index:
            raise ValueError(f"Model B P1_G1 result is missing for {year}.")
        result_b = model_b_rows.loc[year]
        if isinstance(result_b, pd.DataFrame):
            raise ValueError(f"Duplicate Model B P1_G1 rows for {year}.")

        expected_origin = prediction_origin(year).date().isoformat()
        if result_a.get("prediction_origin") != expected_origin:
            raise ValueError(
                f"Model A prediction origin differs from foundation cutoff in {year}."
            )
        if int(result_a["candidate_units"]) != int(result_b["candidate_count"]):
            raise ValueError(
                f"Model A and B candidate cohorts differ in {year}."
            )
        realized_a = _finite_scalar(result_a["realized_P1_pct"], "Model A realized")
        realized_b = _finite_scalar(result_b["observed_P1_percent"], "Model B realized")
        if not np.isclose(realized_a, realized_b, rtol=0.0, atol=1e-10):
            raise ValueError(
                f"Model A and B realized P1 differ in {year}; "
                "their historical cohorts/targets are not aligned."
            )
        prediction_a = _finite_scalar(
            result_a["predicted_P1_pct"], "Model A prediction"
        )
        prediction_b = _finite_scalar(
            result_b["predicted_P1_percent"], "Model B prediction"
        )
        prediction_ensemble = (
            ENSEMBLE_WEIGHTS["model_a"] * prediction_a
            + ENSEMBLE_WEIGHTS["model_b"] * prediction_b
        )
        rows.append(
            {
                "year": year,
                "realized": realized_a,
                "A_prediction": prediction_a,
                "B_prediction": prediction_b,
                "ensemble_50_50": prediction_ensemble,
                "A_error": prediction_a - realized_a,
                "B_error": prediction_b - realized_a,
                "ensemble_error": prediction_ensemble - realized_a,
            }
        )

    fold_results = pd.DataFrame(rows)
    metric_rows = []
    for model_name, error_column in (
        ("Model A", "A_error"),
        ("Model B P1_G1", "B_error"),
        ("Ensemble 50/50", "ensemble_error"),
    ):
        errors = fold_results[error_column].to_numpy(dtype=float)
        metric_rows.append(
            {
                "model": model_name,
                "MAE": float(np.mean(np.abs(errors))),
                "RMSE": float(np.sqrt(np.mean(np.square(errors)))),
                "bias": float(np.mean(errors)),
                "worst_absolute_error": float(np.max(np.abs(errors))),
            }
        )
    metrics = pd.DataFrame(metric_rows)
    return fold_results, metrics
