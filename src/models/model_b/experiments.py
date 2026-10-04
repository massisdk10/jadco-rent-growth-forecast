"""Configuration placeholders for the Model B experiment ladder."""

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from src.evaluation.backtest import (
    BACKTEST_YEARS,
    aggregate_expected_contribution,
    occurrence_history,
    occurrence_metrics,
    predict_occurrence,
)
from src.features.model_dataset import (
    ADMISSIBLE_FEATURES,
    OCCURRENCE_FEATURES,
    build_model_dataset,
    build_occurrence_features,
    build_prediction_cohort,
    known_leases,
    prediction_origin,
    reveal_occurrence,
    validate_features,
)
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