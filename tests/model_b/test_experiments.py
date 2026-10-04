"""Synthetic tests for the Model B experiment configuration skeleton."""

import unittest

from src.models.model_b.experiments import (
    ExperimentConfig,
    ExperimentVariant,
    run_experiment,
)
from src.features.model_dataset import ADMISSIBLE_FEATURES


class ExperimentConfigTests(unittest.TestCase):
    def test_four_experiment_placeholders_exist(self):
        self.assertEqual(
            {variant.value for variant in ExperimentVariant},
            {"B0", "B1", "B2", "B3"},
        )
        b0 = ExperimentConfig(
            variant=ExperimentVariant.B0_BASELINE,
            feature_names=(),
        )
        b1 = ExperimentConfig(
            variant=ExperimentVariant.B1_HIERARCHICAL_INTERNAL,
            feature_names=ADMISSIBLE_FEATURES,
        )
        b2 = ExperimentConfig(
            variant=ExperimentVariant.B2_ELIGIBLE_EXTERNAL,
            external_signals=("rent_cpi_annual_growth",),
        )
        b3 = ExperimentConfig(
            variant=ExperimentVariant.B3_EXTERNAL_SENSITIVITY,
            external_signals=("vacancy_rate",),
            scenario_name="synthetic-sensitivity",
        )

        self.assertEqual((b0.variant, b1.variant, b2.variant, b3.variant), tuple(ExperimentVariant))

    def test_external_context_is_selected_from_known_public_schema(self):
        with self.assertRaises(ValueError):
            ExperimentConfig(
                variant=ExperimentVariant.B2_ELIGIBLE_EXTERNAL,
                external_signals=("unverified_custom_feature",),
            )
        with self.assertRaises(ValueError):
            ExperimentConfig(variant=ExperimentVariant.B2_ELIGIBLE_EXTERNAL)
        with self.assertRaises(ValueError):
            ExperimentConfig(
                variant=ExperimentVariant.B3_EXTERNAL_SENSITIVITY,
                external_signals=("vacancy_rate",),
            )

    def test_experiments_cannot_expand_internal_feature_allowlist(self):
        with self.assertRaises(ValueError):
            ExperimentConfig(
                variant=ExperimentVariant.B1_HIERARCHICAL_INTERNAL,
                feature_names=("new_sRent",),
            )

    def test_execution_is_an_explicit_placeholder(self):
        config = ExperimentConfig(
            variant=ExperimentVariant.B1_HIERARCHICAL_INTERNAL,
        )

        with self.assertRaises(NotImplementedError):
            run_experiment(config)


if __name__ == "__main__":
    unittest.main()