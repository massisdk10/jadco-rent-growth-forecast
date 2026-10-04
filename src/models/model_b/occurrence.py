"""Hierarchical, partially pooled estimates of transition occurrence."""

from collections.abc import Hashable
from dataclasses import dataclass
from math import isfinite
from numbers import Real

import numpy as np
import pandas as pd

from src.features.model_dataset import OCCURRENCE_FEATURES


_SEGMENT_FEATURES = frozenset({"province", "building", "expiry_month"})


def _category(value: object) -> Hashable | None:
    """Return a stable group key, treating missing values as no segment."""
    if not pd.api.types.is_scalar(value):
        raise ValueError("Occurrence segment values must be scalar and hashable.")
    if pd.isna(value):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    if not isinstance(value, Hashable):
        raise ValueError("Occurrence segment values must be scalar and hashable.")
    return value


@dataclass(frozen=True)
class OccurrenceConfig:
    """Configuration for segment hierarchy and partial-pooling strength.

    Each hierarchy entry is a grouping level. ``hierarchy=()`` uses the global
    rate only. Supported segment levels are province, building, province by
    expiry month, building by expiry month, province then building, and
    province then building then building by expiry month. In the nested
    building hierarchy, province remains part of the child key so a building
    is pooled only within its training/candidate province. All levels are
    optional and should be compared empirically before selecting a
    configuration.

    Segments with fewer than ``minimum_segment_size`` qualified labels back
    off fully to their parent. Otherwise their empirical rate is shrunk toward
    that parent's estimate using ``shrinkage_strength``.
    """

    hierarchy: tuple[tuple[str, ...], ...] = (("province",),)
    shrinkage_strength: float = 10.0
    minimum_segment_size: int = 1

    def __post_init__(self) -> None:
        if any(not level for level in self.hierarchy):
            raise ValueError("hierarchy levels must be non-empty.")
        if len(self.hierarchy) > 3:
            raise ValueError("hierarchy supports at most three segment levels.")
        previous: tuple[str, ...] | None = None
        for level in self.hierarchy:
            if len(level) != len(set(level)):
                raise ValueError("A hierarchy level cannot repeat a feature.")
            if (
                len(level) > 3
                or any(feature not in _SEGMENT_FEATURES for feature in level)
                or level
                not in (
                    ("province",),
                    ("building",),
                    ("province", "expiry_month"),
                    ("building", "expiry_month"),
                    ("province", "building"),
                    ("province", "building", "expiry_month"),
                )
            ):
                raise ValueError(
                    "hierarchy must be one of the supported province/building/expiry-month levels."
                )
            if previous is not None and (
                len(level) != len(previous) + 1
                or level[: len(previous)] != previous
            ):
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


def _validate_features(
    features: pd.DataFrame, required_features: frozenset[str]
) -> None:
    if not features.columns.is_unique:
        raise ValueError("Feature columns must be unique.")
    unexpected = set(features.columns) - set(OCCURRENCE_FEATURES)
    if unexpected:
        raise ValueError(
            f"Features outside the occurrence allowlist: {sorted(unexpected)}"
        )
    missing = required_features - set(features.columns)
    if missing:
        raise ValueError(
            f"Required occurrence segment features are missing: {sorted(missing)}"
        )


class HierarchicalOccurrenceEstimator:
    """Estimate occurrence rates through configurable empirical-rate pooling.

    At each configured level, a segment's observed rate is blended with its
    parent's already-shrunk rate. The blend weight is determined by the
    configured prior strength and the number of qualified labels in that
    segment. The only training labels are the supplied, already-qualified
    occurrence labels; cohort selection and label construction remain the
    responsibility of the frozen foundation.
    """

    def __init__(self, config: OccurrenceConfig | None = None) -> None:
        self.config = config if config is not None else OccurrenceConfig()
        self._required_features = frozenset(
            feature for level in self.config.hierarchy for feature in level
        )
        self._global_rate: float | None = None
        self._segment_rates: list[dict[tuple[Hashable, ...], float]] = []

    @staticmethod
    def _segment_key(row: pd.Series, level: tuple[str, ...]) -> tuple[Hashable, ...] | None:
        values = tuple(_category(row[feature]) for feature in level)
        if any(value is None for value in values):
            return None
        return values  # type: ignore[return-value]

    def fit(
        self, features: pd.DataFrame, target: pd.Series
    ) -> "HierarchicalOccurrenceEstimator":
        """Fit global and segment rates from qualified binary labels."""
        _validate_features(features, self._required_features)
        if len(features) != len(target) or not features.index.equals(target.index):
            raise ValueError("Occurrence features and labels must be row-aligned.")
        if len(features) == 0:
            raise ValueError("Occurrence training data must not be empty.")

        labels = pd.to_numeric(target, errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(labels).all() or not np.isin(labels, (0.0, 1.0)).all():
            raise ValueError("Occurrence targets must be finite binary labels (0 or 1).")

        global_rate = float(labels.mean())
        segment_rates: list[dict[tuple[Hashable, ...], float]] = []
        for level in self.config.hierarchy:
            counts: dict[tuple[Hashable, ...], int] = {}
            positives: dict[tuple[Hashable, ...], int] = {}
            for (_, row), label in zip(features.iterrows(), labels):
                key = self._segment_key(row, level)
                if key is None:
                    continue
                counts[key] = counts.get(key, 0) + 1
                positives[key] = positives.get(key, 0) + int(label)

            level_rates: dict[tuple[Hashable, ...], float] = {}
            for key, count in counts.items():
                if count < self.config.minimum_segment_size:
                    continue
                if not segment_rates:
                    parent_rate = global_rate
                else:
                    parent_length = len(self.config.hierarchy[len(segment_rates) - 1])
                    parent_key = key[:parent_length]
                    parent_rate = segment_rates[-1].get(parent_key, global_rate)
                level_rates[key] = (
                    positives[key] + self.config.shrinkage_strength * parent_rate
                ) / (count + self.config.shrinkage_strength)
            segment_rates.append(level_rates)
        self._global_rate = global_rate
        self._segment_rates = segment_rates
        return self

    def predict_proba(self, candidates: pd.DataFrame) -> np.ndarray:
        """Predict one cutoff-time occurrence probability per candidate."""
        _validate_features(candidates, self._required_features)
        if self._global_rate is None:
            raise RuntimeError("Occurrence estimator must be fitted before prediction.")
        predictions = np.full(len(candidates), self._global_rate, dtype=float)
        if not self.config.hierarchy or candidates.empty:
            return predictions

        for row_position, (_, row) in enumerate(candidates.iterrows()):
            for level_index, level in enumerate(self.config.hierarchy):
                key = self._segment_key(row, level)
                if key is None:
                    break
                estimate = self._segment_rates[level_index].get(key)
                if estimate is None:
                    break
                predictions[row_position] = estimate
        return np.clip(predictions, 0.0, 1.0)