"""Hierarchical robust estimates of conditional annualized rent growth."""

from collections.abc import Hashable
from dataclasses import dataclass
from math import isfinite
from numbers import Real

import numpy as np
import pandas as pd

from src.features.model_dataset import ADMISSIBLE_FEATURES, MAIN_TARGET


_SEGMENT_FEATURES = frozenset(
    {"province", "building", "bedrooms"}
)
_SUPPORTED_HIERARCHIES = frozenset(
    {
        (),
        (("province",),),
        (("building",),),
        (("bedrooms",),),
        (("province",), ("province", "building")),
        (("building",), ("building", "bedrooms")),
        (
            ("province",),
            ("province", "building"),
            ("province", "building", "bedrooms"),
        ),
    }
)


def _category(value: object) -> Hashable | None:
    """Return a hashable segment value, treating null/blank values as absent."""
    if not pd.api.types.is_scalar(value):
        raise ValueError("Growth segment values must be scalar and hashable.")
    if pd.isna(value):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    if not isinstance(value, Hashable):
        raise ValueError("Growth segment values must be scalar and hashable.")
    return value


@dataclass(frozen=True)
class GrowthConfig:
    """Configuration for robust segment hierarchy and partial pooling.

    At each level, the raw segment statistic is its median. It is blended
    with the already-shrunk parent estimate using ``n / (n + strength)``.
    Segments below ``minimum_segment_size`` back off fully to the parent.
    An empty hierarchy requests the global median only.
    """

    hierarchy: tuple[tuple[str, ...], ...] = (
        ("province",),
        ("province", "building"),
    )
    shrinkage_strength: float = 10.0
    minimum_segment_size: int = 10

    def __post_init__(self) -> None:
        if not isinstance(self.hierarchy, tuple) or any(
            not isinstance(level, tuple) for level in self.hierarchy
        ):
            raise ValueError("hierarchy must be a tuple of feature-level tuples.")
        if any(
            not isinstance(feature, str)
            for level in self.hierarchy
            for feature in level
        ):
            raise ValueError("hierarchy features must be strings.")
        if self.hierarchy not in _SUPPORTED_HIERARCHIES:
            raise ValueError(
                "hierarchy must be one of the supported province/building/bedrooms levels."
            )
        if any(
            feature not in _SEGMENT_FEATURES
            for level in self.hierarchy
            for feature in level
        ):
            raise ValueError("hierarchy features must be approved segment features.")
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


def _validate_features(
    features: pd.DataFrame, required_features: frozenset[str]
) -> None:
    if not isinstance(features, pd.DataFrame):
        raise TypeError("Growth features must be a pandas DataFrame.")
    if not features.columns.is_unique:
        raise ValueError("Feature columns must be unique.")
    unexpected = set(features.columns) - set(ADMISSIBLE_FEATURES)
    if unexpected:
        raise ValueError(
            f"Features outside the foundation allowlist: {sorted(unexpected)}"
        )
    missing = required_features - set(features.columns)
    if missing:
        raise ValueError(
            f"Required growth segment features are missing: {sorted(missing)}"
        )


class HierarchicalGrowthEstimator:
    """Fit median-based hierarchical estimates of conditional growth ``g_i``.

    ``fit`` expects already-qualified historical labels for the frozen
    ``MAIN_TARGET``. Same-unit construction, eligibility, and cutoff filtering
    remain the responsibility of the frozen foundation and caller. Only
    explicitly allowlisted cutoff features are accepted.
    """

    def __init__(self, config: GrowthConfig | None = None) -> None:
        self.config = config if config is not None else GrowthConfig()
        self._required_features = frozenset(
            feature for level in self.config.hierarchy for feature in level
        )
        self._global_median: float | None = None
        self._segment_estimates: list[
            dict[tuple[Hashable, ...], float]
        ] = []

    @staticmethod
    def _segment_key(
        row: pd.Series, level: tuple[str, ...]
    ) -> tuple[Hashable, ...] | None:
        values = tuple(_category(row[feature]) for feature in level)
        if any(value is None for value in values):
            return None
        return values  # type: ignore[return-value]

    def fit(
        self, features: pd.DataFrame, target: pd.Series
    ) -> "HierarchicalGrowthEstimator":
        """Fit from finite, qualified ``target_effective_annualized`` labels."""
        _validate_features(features, self._required_features)
        if not isinstance(target, pd.Series):
            raise TypeError("Growth target must be a pandas Series.")
        if target.name != MAIN_TARGET:
            raise ValueError(f"Growth target must be named {MAIN_TARGET!r}.")
        if len(features) != len(target) or not features.index.equals(target.index):
            raise ValueError("Growth features and labels must be row-aligned.")
        if len(features) == 0:
            raise ValueError("Growth training data must not be empty.")

        labels = pd.to_numeric(target, errors="coerce").astype(float).to_numpy()
        if not np.isfinite(labels).all():
            raise ValueError("Growth targets must contain only finite numeric values.")

        global_median = float(np.median(labels))
        segment_estimates: list[dict[tuple[Hashable, ...], float]] = []
        for level_index, level in enumerate(self.config.hierarchy):
            grouped: dict[tuple[Hashable, ...], list[float]] = {}
            for (_, row), label in zip(features.iterrows(), labels):
                key = self._segment_key(row, level)
                if key is not None:
                    grouped.setdefault(key, []).append(float(label))

            level_estimates: dict[tuple[Hashable, ...], float] = {}
            for key, values in grouped.items():
                count = len(values)
                if count < self.config.minimum_segment_size:
                    continue
                if level_index == 0:
                    parent_estimate = global_median
                else:
                    parent_length = len(self.config.hierarchy[level_index - 1])
                    parent_key = key[:parent_length]
                    parent_estimate = segment_estimates[-1].get(
                        parent_key, global_median
                    )
                child_median = float(np.median(values))
                weight = count / (count + self.config.shrinkage_strength)
                level_estimates[key] = (
                    weight * child_median + (1.0 - weight) * parent_estimate
                )
            segment_estimates.append(level_estimates)

        self._global_median = global_median
        self._segment_estimates = segment_estimates
        return self

    def predict(self, candidates: pd.DataFrame) -> np.ndarray:
        """Predict one finite conditional-growth estimate per candidate."""
        _validate_features(candidates, self._required_features)
        if self._global_median is None:
            raise RuntimeError("Growth estimator must be fitted before prediction.")
        predictions = np.full(len(candidates), self._global_median, dtype=float)
        if not self.config.hierarchy or candidates.empty:
            return predictions

        for row_position, (_, row) in enumerate(candidates.iterrows()):
            for level_index, level in enumerate(self.config.hierarchy):
                key = self._segment_key(row, level)
                if key is None:
                    break
                estimate = self._segment_estimates[level_index].get(key)
                if estimate is None:
                    break
                predictions[row_position] = estimate
        return predictions
