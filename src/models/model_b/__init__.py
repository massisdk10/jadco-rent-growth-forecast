"""Model B: hierarchical, segment-based statistical forecasting interfaces."""

from src.features.model_dataset import (
    ADMISSIBLE_FEATURES,
    OCCURRENCE_FEATURES,
    prediction_origin,
)
from src.evaluation.backtest import aggregate_expected_contribution

from .growth import GrowthConfig, HierarchicalGrowthEstimator
from .model import ModelB, aggregate_p1, assemble_predictions
from .occurrence import HierarchicalOccurrenceEstimator, OccurrenceConfig

__all__ = [
    "ADMISSIBLE_FEATURES",
    "GrowthConfig",
    "HierarchicalGrowthEstimator",
    "HierarchicalOccurrenceEstimator",
    "ModelB",
    "OCCURRENCE_FEATURES",
    "OccurrenceConfig",
    "aggregate_expected_contribution",
    "aggregate_p1",
    "assemble_predictions",
    "prediction_origin",
]