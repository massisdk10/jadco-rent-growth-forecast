"""Interfaces for hierarchical estimates of transition occurrence."""

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Sequence

import numpy as np
import pandas as pd

from src.features.model_dataset import OCCURRENCE_FEATURES


_SEGMENT_FEATURES = frozenset(
    {
        "property_code",
        "building",
        "province",
        "bedrooms",
        "bathrooms",
        "floor",
        "unit_subtype",
        "expiry_month",
    }
)


@dataclass(frozen=True)
class OccurrenceConfig:
    """Configuration for segment hierarchy and partial-pooling strength.

    Each hierarchy entry is a grouping level; later entries must extend the
    preceding level. Values are restricted to existing categorical cutoff
    features. Estimation is intentionally not implemented in this skeleton.
    """

    hierarchy: tuple[tuple[str, ...], ...] = (
        ("province",),
        ("province", "building"),
    )
    shrinkage_strength: float = 10.0
    minimum_segment_size: int = 10

    def __post_init__(self) -> None:
        if not self.hierarchy or any(not level for level in self.hierarchy):
            raise ValueError("hierarchy must contain non-empty grouping levels.")
        previous: tuple[str, ...] = ()
        for level in self.hierarchy:
            if len(level) != len(set(level)):
                raise ValueError("A hierarchy level cannot repeat a feature.")
            if any(feature not in _SEGMENT_FEATURES for feature in level):
                raise ValueError("hierarchy features must be approved categorical cutoff features.")
            if previous and level[: len(previous)] != previous:
                raise ValueError("Each hierarchy level must extend its parent level.")
            previous = level
        if (
            isinstance(self.shrinkage_strength, bool)
            or not isinstance(self.shrinkage_strength, Real)
            or not isfinite(self.shrinkage_strength)
            or self.shrinkage_strength <= 0
        ):
            raise ValueError("shrinkage_strength must be finite and greater than zero.")
        if (
            isinstance(self.minimum_segment_size, bool)
            or not isinstance(self.minimum_segment_size, int)
            or self.minimum_segment_size < 1
        ):
            raise ValueError("minimum_segment_size must be a positive integer.")


def _validate_features(features: pd.DataFrame) -> None:
    if not features.columns.is_unique:
        raise ValueError("Feature columns must be unique.")
    unexpected = set(features.columns) - set(OCCURRENCE_FEATURES)
    if unexpected:
        raise ValueError(
            f"Features outside the occurrence allowlist: {sorted(unexpected)}"
        )


class HierarchicalOccurrenceEstimator:
    """Fit and predict cutoff-time transition probabilities ``p_i``.

    ``fit`` receives only an explicit allowlisted feature frame and separate
    occurrence labels. ``predict_proba`` returns one probability per row.
    The hierarchical statistical estimator is a deliberate Model B extension
    point and is not implemented here.
    """

    def __init__(self, config: OccurrenceConfig | None = None) -> None:
        self.config = config if config is not None else OccurrenceConfig()

    def fit(
        self, features: pd.DataFrame, target: pd.Series
    ) -> "HierarchicalOccurrenceEstimator":
        """Fit from known occurrence labels; implementation is deferred."""
        _validate_features(features)
        if len(features) != len(target) or not features.index.equals(target.index):
            raise ValueError("Occurrence features and labels must be row-aligned.")
        raise NotImplementedError("Hierarchical occurrence fitting is not implemented yet.")

    def predict_proba(self, candidates: pd.DataFrame) -> np.ndarray:
        """Predict one cutoff-time occurrence probability per candidate."""
        _validate_features(candidates)
        raise NotImplementedError(
            "Hierarchical occurrence prediction is not implemented yet."
        )