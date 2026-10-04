"""Interfaces for hierarchical conditional-growth estimates."""

from dataclasses import dataclass
from math import isfinite
from numbers import Real

import numpy as np
import pandas as pd

from src.features.model_dataset import ADMISSIBLE_FEATURES


_SEGMENT_FEATURES = frozenset(
    {
        "property_code",
        "building",
        "province",
        "bedrooms",
        "bathrooms",
        "floor",
        "unit_subtype",
    }
)


@dataclass(frozen=True)
class GrowthConfig:
    """Configuration for segment hierarchy and partial-pooling strength."""

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
    unexpected = set(features.columns) - set(ADMISSIBLE_FEATURES)
    if unexpected:
        raise ValueError(
            f"Features outside the growth allowlist: {sorted(unexpected)}"
        )


class HierarchicalGrowthEstimator:
    """Fit and predict conditional annualized growth estimates ``g_i``.

    Fitting and prediction accept only explicit internal cutoff features;
    growth targets are passed separately from the feature frame. The
    hierarchical statistical estimator is not implemented in this skeleton.
    """

    def __init__(self, config: GrowthConfig | None = None) -> None:
        self.config = config if config is not None else GrowthConfig()

    def fit(
        self, features: pd.DataFrame, target: pd.Series
    ) -> "HierarchicalGrowthEstimator":
        """Fit from qualified positive-occurrence growth labels."""
        _validate_features(features)
        if len(features) != len(target) or not features.index.equals(target.index):
            raise ValueError("Growth features and labels must be row-aligned.")
        raise NotImplementedError("Hierarchical growth fitting is not implemented yet.")

    def predict(self, candidates: pd.DataFrame) -> np.ndarray:
        """Predict one conditional-growth fraction per candidate."""
        _validate_features(candidates)
        raise NotImplementedError("Hierarchical growth prediction is not implemented yet.")