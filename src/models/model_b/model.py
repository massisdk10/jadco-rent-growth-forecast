"""Model B wrapper and P1 prediction assembly."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.evaluation.backtest import aggregate_expected_contribution
from src.features.model_dataset import ADMISSIBLE_FEATURES, OCCURRENCE_FEATURES

from .growth import HierarchicalGrowthEstimator
from .occurrence import HierarchicalOccurrenceEstimator


PREDICTION_COLUMNS = ("p_i", "g_i", "contribution_i")


def _as_vector(values: Sequence[float] | np.ndarray | pd.Series, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return vector


def assemble_predictions(
    probabilities: Sequence[float] | np.ndarray | pd.Series,
    conditional_growth: Sequence[float] | np.ndarray | pd.Series,
    *,
    index: pd.Index | None = None,
) -> pd.DataFrame:
    """Combine aligned component outputs without carrying feature/label columns."""
    p = _as_vector(probabilities, "probabilities")
    g = _as_vector(conditional_growth, "conditional_growth")
    if len(p) != len(g):
        raise ValueError("Occurrence and growth prediction lengths must match.")
    if index is not None and len(index) != len(p):
        raise ValueError("Prediction index length must match prediction vectors.")
    if not len(p) or not np.isfinite(p).all() or not np.isfinite(g).all():
        raise ValueError("Predictions must be non-empty and finite.")
    if not ((p >= 0) & (p <= 1)).all():
        raise ValueError("Occurrence probabilities must be in [0, 1].")
    return pd.DataFrame(
        {
            "p_i": p,
            "g_i": g,
            "contribution_i": p * g,
        },
        index=index,
    )


def aggregate_p1(predictions: pd.DataFrame) -> float:
    """Return the foundation P1 aggregate: median of ``p_i * g_i``."""
    if not set(PREDICTION_COLUMNS).issubset(predictions.columns):
        raise ValueError(f"predictions must contain {PREDICTION_COLUMNS}.")
    expected_contributions = (
        predictions["p_i"].to_numpy(dtype=float)
        * predictions["g_i"].to_numpy(dtype=float)
    )
    contributions = predictions["contribution_i"].to_numpy(dtype=float)
    if not np.isfinite(expected_contributions).all() or not np.array_equal(
        contributions, expected_contributions
    ):
        raise ValueError("contribution_i must equal p_i * g_i for every candidate.")
    return aggregate_expected_contribution(
        predictions["p_i"].to_numpy(),
        predictions["g_i"].to_numpy(),
    )


class ModelB:
    """Combine hierarchical occurrence and conditional-growth components.

    Component fitting remains the caller's responsibility so each estimator
    receives its own foundation-qualified, cutoff-safe training sample.
    """

    def __init__(
        self,
        occurrence_estimator: HierarchicalOccurrenceEstimator | None = None,
        growth_estimator: HierarchicalGrowthEstimator | None = None,
    ) -> None:
        self.occurrence_estimator = (
            occurrence_estimator
            if occurrence_estimator is not None
            else HierarchicalOccurrenceEstimator()
        )
        self.growth_estimator = (
            growth_estimator
            if growth_estimator is not None
            else HierarchicalGrowthEstimator()
        )

    def fit(
        self,
        occurrence_features: pd.DataFrame,
        occurrence_target: pd.Series,
        growth_features: pd.DataFrame,
        growth_target: pd.Series,
    ) -> "ModelB":
        """Fit both estimators on separately prepared, cutoff-safe samples."""
        self.occurrence_estimator.fit(occurrence_features, occurrence_target)
        self.growth_estimator.fit(growth_features, growth_target)
        return self

    def predict(
        self,
        occurrence_candidates: pd.DataFrame,
        growth_candidates: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return per-row ``p_i``, ``g_i``, and ``p_i * g_i`` contributions."""
        if not occurrence_candidates.index.equals(growth_candidates.index):
            raise ValueError("Occurrence and growth candidate rows must be aligned.")
        if not occurrence_candidates.columns.is_unique or not growth_candidates.columns.is_unique:
            raise ValueError("Candidate feature columns must be unique.")
        unexpected_occurrence = set(occurrence_candidates.columns) - set(
            OCCURRENCE_FEATURES
        )
        unexpected_growth = set(growth_candidates.columns) - set(ADMISSIBLE_FEATURES)
        if unexpected_occurrence or unexpected_growth:
            unexpected = sorted(unexpected_occurrence | unexpected_growth)
            raise ValueError(f"Candidate columns outside the foundation allowlists: {unexpected}")
        p = self.occurrence_estimator.predict_proba(occurrence_candidates)
        g = self.growth_estimator.predict(growth_candidates)
        return assemble_predictions(p, g, index=occurrence_candidates.index)