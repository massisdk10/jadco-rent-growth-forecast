"""Configuration placeholders for the Model B experiment ladder."""

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from src.evaluation.backtest import (
    BACKTEST_YEARS,
    aggregate_expected_contribution,
    baseline_previous_year,
    baseline_recent_history,
    evaluate_predictions,
    occurrence_history,
    occurrence_metrics,
    predict_occurrence,
    split_year,
)
from src.features.model_dataset import (
    ADMISSIBLE_FEATURES,
    MAIN_TARGET,
    OCCURRENCE_FEATURES,
    UNIT_KEY,
    build_features,
    build_model_dataset,
    build_occurrence_features,
    build_prediction_cohort,
    known_leases,
    prediction_origin,
    reveal_occurrence,
    validate_features,
)
from src.models.model_b.growth import GrowthConfig, HierarchicalGrowthEstimator
from src.models.model_b.model import ModelB, aggregate_p1, assemble_predictions
from src.models.model_b.occurrence import (
    HierarchicalOccurrenceEstimator,
    OccurrenceConfig,
)


class ExperimentVariant(str, Enum):
    """Planned experiment families; this module does not run them."""

    B0_BASELINE = "B0"
    B1_HIERARCHICAL_INTERNAL = "B1"
    B2_ELIGIBLE_EXTERNAL = "B2"
    B3_EXTERNAL_SENSITIVITY = "B3"


# Signal identifiers present in the current public external-data schema.
# These remain separate context identifiers, never model feature names.
KNOWN_EXTERNAL_SIGNALS = frozenset(
    {
        "average_rent",
        "average_rent_growth",
        "non_turnover_rent_growth",
        "ontario_guideline",
        "rent_cpi_annual_growth",
        "rent_cpi_annual_mean",
        "rent_cpi_ytd_growth",
        "rent_cpi_ytd_mean",
        "tal_legacy_building_services",
        "tal_legacy_capital_expenditure",
        "tal_legacy_electricity",
        "tal_legacy_fuel_oil_other_energy",
        "tal_legacy_gas",
        "tal_legacy_maintenance",
        "tal_legacy_management",
        "tal_legacy_net_income",
        "tal_legacy_personal_services_rpa",
        "tal_new_base_rent",
        "tal_new_capital_expenditure",
        "tal_new_personal_services",
        "turnover_average_rent",
        "turnover_rent_growth",
        "vacancy_rate",
    }
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Declare a planned comparison without executing or joining any data.

    ``feature_names`` always refers to foundation-approved internal features.
    ``external_signals`` is an optional selection from the current public
    signal schema and is not an expansion of that feature allowlist.
    """

    variant: ExperimentVariant
    feature_names: tuple[str, ...] = ADMISSIBLE_FEATURES
    external_signals: tuple[str, ...] = ()
    scenario_name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.variant, ExperimentVariant):
            raise ValueError("variant must be an ExperimentVariant.")
        if len(self.external_signals) != len(set(self.external_signals)):
            raise ValueError("external_signals cannot contain duplicates.")
        unknown_signals = set(self.external_signals) - KNOWN_EXTERNAL_SIGNALS
        if unknown_signals:
            raise ValueError(f"Unknown external signals: {sorted(unknown_signals)}")

        if self.variant is ExperimentVariant.B0_BASELINE:
            if self.feature_names or self.external_signals or self.scenario_name:
                raise ValueError("B0 baseline cannot declare features or external context.")
            return

        try:
            validate_features(self.feature_names)
        except (TypeError, ValueError) as exc:
            raise ValueError("Experiment features must follow the foundation allowlist.") from exc
        if not self.feature_names:
            raise ValueError("Model experiments must declare at least one internal feature.")

        if self.variant is ExperimentVariant.B1_HIERARCHICAL_INTERNAL:
            if self.external_signals or self.scenario_name:
                raise ValueError("B1 is internal-only.")
        elif self.variant is ExperimentVariant.B2_ELIGIBLE_EXTERNAL:
            if not self.external_signals:
                raise ValueError("B2 must explicitly select eligible external signals.")
            if self.scenario_name:
                raise ValueError("B2 uses only cutoff-eligible external context, not scenarios.")
        elif self.variant is ExperimentVariant.B3_EXTERNAL_SENSITIVITY:
            if not self.external_signals:
                raise ValueError("B3 must explicitly select external signals for sensitivity.")
            if self.scenario_name is None or not self.scenario_name.strip():
                raise ValueError("B3 must declare a non-empty scenario_name.")


def run_experiment(config: ExperimentConfig) -> None:
    """Reserved entry point for later fold-by-fold Model B comparisons."""
    raise NotImplementedError(
        f"Experiment {config.variant.value} is configured only; execution is deferred."
    )


OCCURRENCE_HIERARCHIES: dict[str, tuple[tuple[str, ...], ...]] = {
    "global": (),
    "province": (("province",),),
    "building": (("building",),),
    "province_expiry_month": (("province",), ("province", "expiry_month")),
    "building_expiry_month": (("building",), ("building", "expiry_month")),
    "province_building": (("province",), ("province", "building")),
    "province_building_expiry_month": (
        ("province",),
        ("province", "building"),
        ("province", "building", "expiry_month"),
    ),
}
OCCURRENCE_SHRINKAGE_CANDIDATES = (1, 2, 5, 10, 20, 50)
HISTORICAL_GLOBAL_BASELINE = "historical_global_baseline"
OCCURRENCE_TOURNAMENT_COLUMNS = (
    "year",
    "hierarchy_name",
    "shrinkage_strength",
    "candidate_count",
    "qualified_training_count",
    "observed_transition_rate",
    "predicted_transition_rate",
    "brier_score",
    "calibration_gap",
    "p1_predicted_percent",
    "p1_observed_percent",
    "p1_error_pp",
    "accuracy",
    "precision",
    "recall",
)
OCCURRENCE_SUMMARY_COLUMNS = (
    "hierarchy_name",
    "shrinkage_strength",
    "mean_brier",
    "mean_abs_calibration_gap",
    "p1_mae_pp",
    "p1_rmse_pp",
    "p1_bias_pp",
    "worst_abs_p1_error_pp",
)

GROWTH_HIERARCHIES: dict[str, tuple[tuple[str, ...], ...]] = {
    "G0_global_only": (),
    "G1_province": (("province",),),
    "G2_building": (("building",),),
    "G3_bedrooms": (("bedrooms",),),
    "G4_province_building": (("province",), ("province", "building")),
    "G5_building_bedrooms": (("building",), ("building", "bedrooms")),
    "G6_province_building_bedrooms": (
        ("province",),
        ("province", "building"),
        ("province", "building", "bedrooms"),
    ),
}
GROWTH_SHRINKAGE_CANDIDATES = (1, 2, 5, 10, 20, 50)
GROWTH_BASELINES = (
    "mediane_annee_precedente",
    "mediane_trois_ans",
)
GROWTH_TOURNAMENT_COLUMNS = (
    "year",
    "hierarchy_name",
    "shrinkage_strength",
    "train_count",
    "test_count",
    "mae_pp",
    "rmse_pp",
    "bias_pp",
    "predicted_median_growth_percent",
    "observed_median_growth_percent",
    "portfolio_error_pp",
)
GROWTH_SUMMARY_COLUMNS = (
    "hierarchy_name",
    "shrinkage_strength",
    "mean_mae_pp",
    "mean_rmse_pp",
    "mean_abs_bias_pp",
    "portfolio_mae_pp",
    "portfolio_rmse_pp",
    "portfolio_bias_pp",
    "worst_abs_portfolio_error_pp",
    "mean_mae_rank",
    "mean_rmse_rank",
    "portfolio_mae_rank",
    "worst_portfolio_error_rank",
)


def _growth_features_for(
    hierarchy: tuple[tuple[str, ...], ...],
) -> tuple[str, ...]:
    return tuple(dict.fromkeys(feature for level in hierarchy for feature in level))


def _growth_model_train(
    leases: pd.DataFrame,
    train: pd.DataFrame,
    year: int,
    features: tuple[str, ...],
) -> pd.DataFrame:
    """Build each historical row's features at its own frozen origin."""
    parts = []
    for historical_year, part in train.groupby("year", sort=True):
        x = build_features(leases, part, int(historical_year), features)
        x[MAIN_TARGET] = part[MAIN_TARGET].to_numpy()
        parts.append(x)
    if not parts:
        return pd.DataFrame(columns=[*features, MAIN_TARGET])
    return pd.concat(parts, ignore_index=True)


def _growth_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (name, strength), group in results.groupby(
        ["hierarchy_name", "shrinkage_strength"], dropna=False, sort=False
    ):
        portfolio_errors = pd.to_numeric(
            group["portfolio_error_pp"], errors="coerce"
        ).dropna()
        rows.append(
            {
                "hierarchy_name": name,
                "shrinkage_strength": strength,
                "mean_mae_pp": group["mae_pp"].mean(),
                "mean_rmse_pp": group["rmse_pp"].mean(),
                "mean_abs_bias_pp": group["bias_pp"].abs().mean(),
                "portfolio_mae_pp": portfolio_errors.abs().mean(),
                "portfolio_rmse_pp": (
                    float(np.sqrt(np.mean(np.square(portfolio_errors))))
                    if not portfolio_errors.empty
                    else np.nan
                ),
                "portfolio_bias_pp": portfolio_errors.mean(),
                "worst_abs_portfolio_error_pp": portfolio_errors.abs().max(),
            }
        )
    summary = pd.DataFrame(rows)
    rank_columns = {
        "mean_mae_pp": "mean_mae_rank",
        "mean_rmse_pp": "mean_rmse_rank",
        "portfolio_mae_pp": "portfolio_mae_rank",
        "worst_abs_portfolio_error_pp": "worst_portfolio_error_rank",
    }
    for metric, rank in rank_columns.items():
        summary[rank] = summary[metric].rank(
            method="min", ascending=True, na_option="bottom"
        ).astype("Int64")
    return summary.loc[:, list(GROWTH_SUMMARY_COLUMNS)]


def run_growth_tournament(
    leases: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the fixed, cutoff-safe 2023-2025 conditional-growth tournament.

    Training eligibility, historical feature origins, prediction cohorts,
    target-year evaluation, metrics, and baseline definitions are delegated
    to the frozen foundation helpers. All Model B predictions for all folds
    are fixed before any target-year evaluation rows are selected.
    """
    evaluation_data, _ = build_model_dataset(leases)
    prediction_rows: list[dict[str, object]] = []
    baseline_functions = {
        "previous_year_median_baseline": baseline_previous_year,
        "rolling_historical_median_baseline": baseline_recent_history,
    }

    for year in BACKTEST_YEARS:
        historical, _ = build_model_dataset(
            known_leases(leases, prediction_origin(year))
        )
        train, _ = split_year(historical, year, MAIN_TARGET)
        if train.empty:
            raise ValueError(
                f"No eligible historical growth labels for prediction year {year}."
            )
        cohort = build_prediction_cohort(leases, year)

        for hierarchy_name, hierarchy in GROWTH_HIERARCHIES.items():
            features = _growth_features_for(hierarchy)
            model_train = _growth_model_train(leases, train, year, features)
            candidate_features = build_features(leases, cohort, year, features)
            for strength in GROWTH_SHRINKAGE_CANDIDATES:
                estimator = HierarchicalGrowthEstimator(
                    GrowthConfig(
                        hierarchy=hierarchy,
                        shrinkage_strength=float(strength),
                    )
                )
                estimator.fit(
                    model_train.loc[:, list(features)],
                    model_train[MAIN_TARGET].rename(MAIN_TARGET),
                )
                predictions = estimator.predict(candidate_features)
                prediction_rows.append(
                    {
                        "year": year,
                        "hierarchy_name": hierarchy_name,
                        "shrinkage_strength": float(strength),
                        "train_count": len(model_train),
                        "cohort": cohort.loc[:, list(UNIT_KEY)].copy(),
                        "predictions": predictions,
                    }
                )

        for baseline_name, baseline_function in baseline_functions.items():
            baseline_value = baseline_function(train, year, MAIN_TARGET)
            prediction_rows.append(
                {
                    "year": year,
                    "hierarchy_name": baseline_name,
                    "shrinkage_strength": np.nan,
                    "train_count": len(train),
                    "cohort": cohort.loc[:, list(UNIT_KEY)].copy(),
                    "predictions": np.full(len(cohort), baseline_value, dtype=float),
                }
            )

    result_rows: list[dict[str, object]] = []
    for prediction in prediction_rows:
        year = int(prediction["year"])
        cohort = prediction["cohort"]
        predicted = cohort.copy()
        predicted["_prediction"] = prediction["predictions"]
        _, test = split_year(evaluation_data, year, MAIN_TARGET)
        observed = test.merge(
            predicted,
            on=list(UNIT_KEY),
            how="inner",
            validate="many_to_one",
        )
        metrics = (
            evaluate_predictions(
                observed, observed["_prediction"].to_numpy(dtype=float), MAIN_TARGET
            )
            if not observed.empty
            and np.isfinite(
                observed["_prediction"].to_numpy(dtype=float)
            ).all()
            else {}
        )
        result_rows.append(
            {
                "year": year,
                "hierarchy_name": prediction["hierarchy_name"],
                "shrinkage_strength": prediction["shrinkage_strength"],
                "train_count": prediction["train_count"],
                "test_count": len(observed),
                "mae_pp": metrics.get("MAE_pp", np.nan),
                "rmse_pp": metrics.get("RMSE_pp", np.nan),
                "bias_pp": metrics.get("bias_pp", np.nan),
                "predicted_median_growth_percent": metrics.get(
                    "predicted_growth_pct", np.nan
                ),
                "observed_median_growth_percent": metrics.get(
                    "observed_growth_pct", np.nan
                ),
                "portfolio_error_pp": metrics.get("portfolio_error_pp", np.nan),
            }
        )

    results = pd.DataFrame(result_rows, columns=GROWTH_TOURNAMENT_COLUMNS)
    return results, _growth_summary(results)


def _qualified_training(history: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select only frozen-helper labels that are non-missing at this cutoff."""
    if "occurrence_target" not in history:
        raise ValueError("Occurrence history is missing its frozen label column.")
    qualified = history["occurrence_target"].notna()
    if not qualified.any():
        return (
            pd.DataFrame(columns=OCCURRENCE_FEATURES),
            pd.Series(dtype=float, name="occurrence_target"),
        )
    features = history.loc[qualified, list(OCCURRENCE_FEATURES)].copy()
    labels = pd.to_numeric(
        history.loc[qualified, "occurrence_target"], errors="coerce"
    ).astype(float)
    return features, labels


def _metric_row(
    *,
    year: int,
    hierarchy_name: str,
    shrinkage_strength: float,
    candidate_count: int,
    qualified_training_count: int,
    predictions: np.ndarray,
    observed: pd.DataFrame,
    conditional_growth: float,
) -> dict[str, object]:
    row: dict[str, object] = {
        "year": year,
        "hierarchy_name": hierarchy_name,
        "shrinkage_strength": shrinkage_strength,
        "candidate_count": candidate_count,
        "qualified_training_count": qualified_training_count,
        "observed_transition_rate": np.nan,
        "predicted_transition_rate": np.nan,
        "brier_score": np.nan,
        "calibration_gap": np.nan,
        "p1_predicted_percent": np.nan,
        "p1_observed_percent": np.nan,
        "p1_error_pp": np.nan,
        "accuracy": np.nan,
        "precision": np.nan,
        "recall": np.nan,
    }
    valid_labels = observed["occurrence_target"].notna().to_numpy()
    if valid_labels.any():
        metrics = occurrence_metrics(
            observed.loc[valid_labels, "occurrence_target"].to_numpy(dtype=float),
            predictions[valid_labels],
        )
        row.update(
            observed_transition_rate=metrics["observed_rate"],
            predicted_transition_rate=metrics["predicted_rate"],
            brier_score=metrics["Brier"],
            calibration_gap=metrics["calibration_gap"],
            accuracy=metrics["accuracy"],
            precision=metrics["precision"],
            recall=metrics["recall"],
        )

    observed_growth = pd.to_numeric(
        observed["unit_growth"], errors="coerce"
    ).to_numpy(dtype=float)
    observed_labels = pd.to_numeric(
        observed["occurrence_target"], errors="coerce"
    ).to_numpy(dtype=float)
    observed_contribution = np.where(
        observed_labels == 0.0, 0.0, observed_growth
    )
    if (
        candidate_count
        and np.isfinite(conditional_growth)
        and np.isfinite(predictions).all()
        and np.isfinite(observed_contribution).all()
        and valid_labels.all()
    ):
        predicted_p1 = aggregate_expected_contribution(
            predictions,
            np.full(candidate_count, conditional_growth, dtype=float),
        )
        observed_p1 = aggregate_expected_contribution(
            np.ones(candidate_count, dtype=float),
            observed_contribution,
        )
        row.update(
            p1_predicted_percent=100.0 * predicted_p1,
            p1_observed_percent=100.0 * observed_p1,
            p1_error_pp=100.0 * (predicted_p1 - observed_p1),
        )
    return row


def run_occurrence_tournament(
    leases: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the fixed 2023-2025 occurrence hierarchy/shrinkage tournament.

    For each fold, historical labels and candidate features come exclusively
    from the frozen foundation's cutoff-aware helpers. Predictions for every
    candidate configuration and the frozen historical-global baseline are
    made before target-year transitions are built or occurrence outcomes are
    revealed. P1 uses the shared aggregation with the qualified historical
    conditional-growth mean as a fixed comparator; no growth model is fit.

    Returns the tidy fold-level result table and a per-configuration summary.
    The function does not select a winner.
    """
    result_rows: list[dict[str, object]] = []
    for year in BACKTEST_YEARS:
        origin = prediction_origin(year)
        history = occurrence_history(leases, year)
        train_features, train_labels = _qualified_training(history)
        if train_labels.empty:
            raise ValueError(
                f"No qualified historical occurrence labels for cutoff {origin.date()}."
            )

        cohort = build_prediction_cohort(leases, year)
        candidate_features = build_occurrence_features(leases, cohort, year)
        qualified_growth = pd.to_numeric(
            history.loc[
                history["occurrence_target"].eq(1.0), "unit_growth"
            ],
            errors="coerce",
        )
        qualified_growth = qualified_growth[np.isfinite(qualified_growth)]
        conditional_growth = (
            float(qualified_growth.mean()) if not qualified_growth.empty else np.nan
        )

        fold_predictions: list[
            tuple[str, float, np.ndarray]
        ] = []
        for hierarchy_name, hierarchy in OCCURRENCE_HIERARCHIES.items():
            for strength in OCCURRENCE_SHRINKAGE_CANDIDATES:
                config = OccurrenceConfig(
                    hierarchy=hierarchy,
                    shrinkage_strength=float(strength),
                )
                estimator = HierarchicalOccurrenceEstimator(config).fit(
                    train_features, train_labels
                )
                probabilities = estimator.predict_proba(candidate_features)
                fold_predictions.append((hierarchy_name, float(strength), probabilities))

        baseline_probabilities = predict_occurrence(
            history, candidate_features, method="historical_global"
        )
        fold_predictions.append(
            (
                HISTORICAL_GLOBAL_BASELINE,
                np.nan,
                np.asarray(baseline_probabilities, dtype=float),
            )
        )

        # Only after all cutoff-time predictions are fixed do we construct
        # target-year transitions and reveal their outcomes.
        target_year_leases = known_leases(
            leases, pd.Timestamp(year=year, month=12, day=31)
        )
        target_year_transitions, _ = build_model_dataset(target_year_leases)
        observed = reveal_occurrence(cohort, target_year_transitions, year)

        for hierarchy_name, strength, probabilities in fold_predictions:
            row = _metric_row(
                year=year,
                hierarchy_name=hierarchy_name,
                shrinkage_strength=strength,
                candidate_count=len(cohort),
                qualified_training_count=len(train_labels),
                predictions=probabilities,
                observed=observed,
                conditional_growth=conditional_growth,
            )
            result_rows.append(row)

    results = pd.DataFrame(result_rows, columns=OCCURRENCE_TOURNAMENT_COLUMNS)
    summary_rows: list[dict[str, object]] = []
    for (hierarchy_name, strength), group in results.groupby(
        ["hierarchy_name", "shrinkage_strength"], dropna=False, sort=False
    ):
        p1_errors = pd.to_numeric(group["p1_error_pp"], errors="coerce").dropna()
        summary_rows.append(
            {
                "hierarchy_name": hierarchy_name,
                "shrinkage_strength": strength,
                "mean_brier": group["brier_score"].mean(),
                "mean_abs_calibration_gap": group["calibration_gap"].abs().mean(),
                "p1_mae_pp": p1_errors.abs().mean(),
                "p1_rmse_pp": np.sqrt(np.mean(np.square(p1_errors)))
                if not p1_errors.empty
                else np.nan,
                "p1_bias_pp": p1_errors.mean(),
                "worst_abs_p1_error_pp": p1_errors.abs().max(),
            }
        )
    summary = pd.DataFrame(summary_rows, columns=OCCURRENCE_SUMMARY_COLUMNS)
    return results, summary


P1_OCCURRENCE_CONFIG = OccurrenceConfig(
    hierarchy=(("province",), ("province", "expiry_month")),
    shrinkage_strength=20.0,
)
P1_GROWTH_CONFIG = GrowthConfig(
    hierarchy=(("province",),),
    shrinkage_strength=1.0,
)
P1_COMBINATIONS = {
    "P0_G0": ("historical_global", "previous_year_median"),
    "P0_G1": ("historical_global", "hierarchical_province"),
    "P1_G0": ("hierarchical_province_expiry_month", "previous_year_median"),
    "P1_G1": ("hierarchical_province_expiry_month", "hierarchical_province"),
}
P1_TOURNAMENT_COLUMNS = (
    "year",
    "model_name",
    "candidate_count",
    "predicted_P1_percent",
    "observed_P1_percent",
    "error_pp",
    "absolute_error_pp",
    "predicted_occurrence_rate",
    "observed_occurrence_rate",
    "predicted_conditional_growth_percent",
    "observed_conditional_growth_percent",
)
P1_SUMMARY_COLUMNS = (
    "model_name",
    "p1_mae_pp",
    "p1_rmse_pp",
    "p1_bias_pp",
    "worst_absolute_yearly_error_pp",
    "p1_mae_rank",
    "p1_rmse_rank",
    "p1_bias_rank",
    "worst_absolute_error_rank",
)


def _p1_growth_training(
    leases: pd.DataFrame, train: pd.DataFrame, year: int
) -> pd.DataFrame:
    """Build historical growth features at each transition's own origin."""
    return _growth_model_train(leases, train, year, ("province",))


def _p1_tournament_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, group in results.groupby("model_name", sort=False):
        errors = pd.to_numeric(group["error_pp"], errors="coerce").dropna()
        rows.append(
            {
                "model_name": model_name,
                "p1_mae_pp": errors.abs().mean(),
                "p1_rmse_pp": (
                    float(np.sqrt(np.mean(np.square(errors))))
                    if not errors.empty
                    else np.nan
                ),
                "p1_bias_pp": errors.mean(),
                "worst_absolute_yearly_error_pp": errors.abs().max(),
            }
        )
    summary = pd.DataFrame(rows)
    for metric, rank in (
        ("p1_mae_pp", "p1_mae_rank"),
        ("p1_rmse_pp", "p1_rmse_rank"),
        ("p1_bias_pp", "p1_bias_rank"),
        ("worst_absolute_yearly_error_pp", "worst_absolute_error_rank"),
    ):
        values = summary[metric].abs() if metric == "p1_bias_pp" else summary[metric]
        summary[rank] = values.rank(method="min", ascending=True).astype("Int64")
    return summary.loc[:, list(P1_SUMMARY_COLUMNS)]


def run_p1_combination_tournament(
    leases: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate the four shortlisted occurrence-growth combinations on 2023-25.

    All cutoff-time fits and candidate predictions are completed before the
    first target-year outcome is revealed. Cohorts, cutoff snapshots, labels,
    baselines, and P1 aggregation are delegated to the frozen foundation.
    """
    evaluation_data, _ = build_model_dataset(leases)
    predictions_by_year: list[dict[str, object]] = []

    for year in BACKTEST_YEARS:
        origin = prediction_origin(year)
        history = occurrence_history(leases, year)
        occurrence_features, occurrence_target = _qualified_training(history)
        if occurrence_target.empty:
            raise ValueError(
                f"No qualified occurrence history for cutoff {origin.date()}."
            )

        historical, _ = build_model_dataset(known_leases(leases, origin))
        growth_train, _ = split_year(historical, year, MAIN_TARGET)
        if growth_train.empty:
            raise ValueError(f"No eligible growth history for prediction year {year}.")

        cohort = build_prediction_cohort(leases, year)
        if cohort.empty:
            raise ValueError(f"No candidate units for prediction year {year}.")
        candidate_index = pd.MultiIndex.from_frame(cohort.loc[:, list(UNIT_KEY)])

        occurrence_candidates = build_occurrence_features(leases, cohort, year)
        occurrence_candidates.index = candidate_index
        growth_candidates = build_features(
            leases, cohort, year, features=("province",)
        )
        growth_candidates.index = candidate_index

        growth_model_train = _p1_growth_training(leases, growth_train, year)
        growth_model_train.index = pd.RangeIndex(len(growth_model_train))
        growth_target = growth_model_train[MAIN_TARGET].rename(MAIN_TARGET)
        growth_features = growth_model_train.loc[:, ["province"]]

        # Fix all component predictions before any target-year labels are built.
        occurrence_model = HierarchicalOccurrenceEstimator(P1_OCCURRENCE_CONFIG).fit(
            occurrence_features, occurrence_target
        )
        occurrence_hierarchical = occurrence_model.predict_proba(
            occurrence_candidates
        )
        occurrence_baseline = predict_occurrence(
            history,
            occurrence_candidates,
            method="historical_global",
        )

        growth_model = HierarchicalGrowthEstimator(P1_GROWTH_CONFIG).fit(
            growth_features, growth_target
        )
        growth_hierarchical = growth_model.predict(growth_candidates)
        previous_year_growth = baseline_previous_year(
            growth_train, year, MAIN_TARGET
        )
        if not np.isfinite(previous_year_growth):
            raise ValueError(
                f"Previous-year growth baseline is unavailable for {year}."
            )
        growth_baseline = np.full(len(cohort), previous_year_growth, dtype=float)

        component_predictions = {
            "P0_G0": (occurrence_baseline, growth_baseline),
            "P0_G1": (occurrence_baseline, growth_hierarchical),
            "P1_G0": (occurrence_hierarchical, growth_baseline),
            "P1_G1": (occurrence_hierarchical, growth_hierarchical),
        }
        for model_name, (probabilities, growth_values) in component_predictions.items():
            assembled = assemble_predictions(
                probabilities,
                growth_values,
                index=candidate_index,
            )
            predictions_by_year.append(
                {
                    "year": year,
                    "model_name": model_name,
                    "cohort": cohort.loc[:, list(UNIT_KEY)].copy(),
                    "predictions": assembled,
                    "candidate_count": len(cohort),
                    "predicted_occurrence_rate": float(np.mean(probabilities)),
                    "predicted_conditional_growth_percent": float(
                        100.0 * np.median(growth_values)
                    ),
                }
            )

    rows: list[dict[str, object]] = []
    for year in BACKTEST_YEARS:
        fold_predictions = [
            prediction
            for prediction in predictions_by_year
            if prediction["year"] == year
        ]
        if len(fold_predictions) != len(P1_COMBINATIONS):
            raise ValueError(f"Expected four frozen combinations for {year}.")
        cohort = fold_predictions[0]["cohort"]
        expected_index = pd.MultiIndex.from_frame(cohort.loc[:, list(UNIT_KEY)])
        for prediction in fold_predictions:
            if not prediction["predictions"].index.equals(expected_index):
                raise ValueError(
                    "Predictions are not aligned to candidate unit keys."
                )

        target_year_transitions, _ = build_model_dataset(
            known_leases(leases, pd.Timestamp(year=year, month=12, day=31))
        )
        observed = reveal_occurrence(cohort, target_year_transitions, year)
        labels = observed["occurrence_target"].to_numpy(dtype=float)
        observed_growth = pd.to_numeric(
            observed["unit_growth"], errors="coerce"
        ).to_numpy(dtype=float)
        observed_contribution = np.where(
            labels == 1.0, observed_growth, 0.0
        )
        observed_p1 = aggregate_expected_contribution(
            np.ones(len(cohort), dtype=float), observed_contribution
        )
        qualified_labels = np.isfinite(labels)
        observed_occurrence_rate = (
            float(np.mean(labels[qualified_labels]))
            if qualified_labels.any()
            else np.nan
        )
        positive_growth = observed_growth[(labels == 1.0) & np.isfinite(observed_growth)]
        observed_conditional_growth = (
            float(100.0 * np.median(positive_growth))
            if positive_growth.size
            else np.nan
        )

        for prediction in fold_predictions:
            model_predictions = prediction["predictions"]
            model_name = str(prediction["model_name"])
            predicted_p1 = 100.0 * aggregate_p1(model_predictions)
            error = predicted_p1 - 100.0 * observed_p1
            rows.append(
                {
                    "year": year,
                    "model_name": model_name,
                    "candidate_count": int(prediction["candidate_count"]),
                    "predicted_P1_percent": predicted_p1,
                    "observed_P1_percent": 100.0 * observed_p1,
                    "error_pp": error,
                    "absolute_error_pp": abs(error),
                    "predicted_occurrence_rate": prediction[
                        "predicted_occurrence_rate"
                    ],
                    "observed_occurrence_rate": observed_occurrence_rate,
                    "predicted_conditional_growth_percent": prediction[
                        "predicted_conditional_growth_percent"
                    ],
                    "observed_conditional_growth_percent": observed_conditional_growth,
                }
            )

    results = pd.DataFrame(rows, columns=P1_TOURNAMENT_COLUMNS)
    return results, _p1_tournament_summary(results)