"""Synthetic tests for hierarchical occurrence-rate shrinkage."""

import unittest

import numpy as np
import pandas as pd

from src.models.model_b.occurrence import (
    HierarchicalOccurrenceEstimator,
    OccurrenceConfig,
)


def training_data(
    province: list[str],
    expiry_month: list[int],
    labels: list[int],
    building: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    features = pd.DataFrame(
        {
            "province": province,
            "building": building or ["b1"] * len(labels),
            "expiry_month": expiry_month,
        }
    )
    return features, pd.Series(labels, index=features.index, dtype=float)


class OccurrenceShrinkageTests(unittest.TestCase):
    def test_global_rate_only(self):
        features = pd.DataFrame(index=pd.RangeIndex(4))
        target = pd.Series([1, 1, 0, 0], dtype=float)
        estimator = HierarchicalOccurrenceEstimator(
            OccurrenceConfig(hierarchy=())
        ).fit(features, target)

        np.testing.assert_allclose(
            estimator.predict_proba(pd.DataFrame(index=pd.RangeIndex(3))),
            [0.5, 0.5, 0.5],
        )

    def test_child_rate_shrinks_toward_global_parent(self):
        features, target = training_data(
            province=["A", "A", "B", "B"],
            expiry_month=[1, 2, 1, 2],
            labels=[1, 1, 0, 0],
        )
        estimator = HierarchicalOccurrenceEstimator(
            OccurrenceConfig(
                hierarchy=(("province",),),
                shrinkage_strength=2,
            )
        ).fit(features, target)
        candidates = pd.DataFrame({"province": ["A", "B"]})

        np.testing.assert_allclose(estimator.predict_proba(candidates), [0.75, 0.25])

    def test_larger_segment_sample_has_more_weight(self):
        features, target = training_data(
            province=["small", *["large"] * 10, *["zero"] * 11],
            expiry_month=[1] * 22,
            labels=[1, *([1] * 10), *([0] * 11)],
        )
        estimator = HierarchicalOccurrenceEstimator(
            OccurrenceConfig(
                hierarchy=(("province",),),
                shrinkage_strength=2,
            )
        ).fit(features, target)
        small, large = estimator.predict_proba(
            pd.DataFrame({"province": ["small", "large"]})
        )

        self.assertAlmostEqual(small, 2 / 3)
        self.assertAlmostEqual(large, 11 / 12)
        self.assertGreater(large, small)

    def test_composite_level_shrinks_to_parent_and_unseen_child_falls_back(self):
        features, target = training_data(
            province=["A", "A", "A", "B", "B", "B"],
            expiry_month=[1, 1, 2, 1, 1, 2],
            labels=[1, 1, 0, 0, 0, 1],
        )
        estimator = HierarchicalOccurrenceEstimator(
            OccurrenceConfig(
                hierarchy=(("province",), ("province", "expiry_month")),
                shrinkage_strength=2,
            )
        ).fit(features, target)
        candidates = pd.DataFrame(
            {
                "province": ["A", "A", "unknown"],
                "expiry_month": [1, 12, 1],
            }
        )
        province_a_rate = 3 / 5
        expected_child_rate = (2 + 2 * province_a_rate) / 4

        np.testing.assert_allclose(
            estimator.predict_proba(candidates),
            [expected_child_rate, province_a_rate, 0.5],
        )

    def test_configurable_hierarchies_include_building_and_expiry(self):
        features, target = training_data(
            province=["A", "A", "B", "B"],
            building=["x", "x", "y", "y"],
            expiry_month=[1, 2, 1, 2],
            labels=[1, 1, 0, 0],
        )
        estimator = HierarchicalOccurrenceEstimator(
            OccurrenceConfig(
                hierarchy=(("building",), ("building", "expiry_month")),
                shrinkage_strength=2,
            )
        ).fit(features, target)

        np.testing.assert_allclose(
            estimator.predict_proba(
                pd.DataFrame(
                    {"building": ["x", "y"], "expiry_month": [1, 1]}
                )
            ),
            [5 / 6, 1 / 6],
        )

    def test_province_building_uses_parent_then_nested_building(self):
        features, target = training_data(
            province=["A"] * 5 + ["B"] * 3,
            building=["x", "x", "x", "y", "y", "z", "z", "z"],
            expiry_month=[1, 1, 2, 1, 2, 1, 2, 3],
            labels=[1, 1, 0, 0, 0, 0, 0, 0],
        )
        estimator = HierarchicalOccurrenceEstimator(
            OccurrenceConfig(
                hierarchy=(("province",), ("province", "building")),
                shrinkage_strength=1,
            )
        ).fit(features, target)
        candidates = pd.DataFrame(
            {
                "province": ["A", "A", "A", "unknown"],
                "building": ["x", "unseen", "y", "x"],
            }
        )

        # Global=.25; province A=(2 + .25)/6=.375.
        # A/x=(2 + .375)/4=.59375; unseen buildings fall back to province.
        np.testing.assert_allclose(
            estimator.predict_proba(candidates),
            [0.59375, 0.375, 0.125, 0.25],
        )

    def test_nested_building_month_falls_back_to_building_province_global(self):
        features, target = training_data(
            province=["A", "A", "A", "A", "A", "B", "B", "B"],
            building=["x", "x", "x", "y", "y", "z", "z", "z"],
            expiry_month=[1, 1, 2, 1, 2, 1, 2, 3],
            labels=[1, 1, 0, 0, 0, 0, 0, 0],
        )
        estimator = HierarchicalOccurrenceEstimator(
            OccurrenceConfig(
                hierarchy=(
                    ("province",),
                    ("province", "building"),
                    ("province", "building", "expiry_month"),
                ),
                shrinkage_strength=1,
            )
        ).fit(features, target)
        candidates = pd.DataFrame(
            {
                "province": ["A", "A", "A", "A", "unknown"],
                "building": ["x", "x", "unseen", "y", "x"],
                "expiry_month": [1, 12, 1, 1, 1],
            }
        )

        # A=.375; A/x=.59375; A/x/month1=(2+.59375)/3.
        # Unknown months/buildings fall back to their parent; the known A/y/month
        # segment shrinks its zero rate toward A/y's already-shrunk estimate.
        np.testing.assert_allclose(
            estimator.predict_proba(candidates),
            [2.59375 / 3, 0.59375, 0.375, 0.0625, 0.25],
        )

    def test_probabilities_stay_in_unit_interval(self):
        features, target = training_data(
            province=["all"] * 20,
            expiry_month=[1] * 20,
            labels=[1] * 20,
        )
        estimator = HierarchicalOccurrenceEstimator(
            OccurrenceConfig(
                hierarchy=(("province",),),
                shrinkage_strength=0.25,
            )
        ).fit(features, target)

        probabilities = estimator.predict_proba(
            pd.DataFrame({"province": ["all", "unseen"]})
        )

        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())

    def test_invalid_shrinkage_configuration_is_rejected(self):
        for strength in (0, -1, float("inf"), float("nan"), True):
            with self.subTest(strength=strength), self.assertRaises(ValueError):
                OccurrenceConfig(shrinkage_strength=strength)
        with self.assertRaises(ValueError):
            OccurrenceConfig(minimum_segment_size=0)

    def test_missing_required_segment_features_are_rejected(self):
        estimator = HierarchicalOccurrenceEstimator(
            OccurrenceConfig(hierarchy=(("province",),))
        )
        with self.assertRaisesRegex(ValueError, "missing"):
            estimator.fit(
                pd.DataFrame({"building": ["b1"]}),
                pd.Series([1]),
            )
        estimator.fit(
            pd.DataFrame({"province": ["A"]}),
            pd.Series([1]),
        )
        with self.assertRaisesRegex(ValueError, "missing"):
            estimator.predict_proba(pd.DataFrame(index=[0]))

    def test_forbidden_outcome_and_future_fields_are_rejected(self):
        estimator = HierarchicalOccurrenceEstimator(
            OccurrenceConfig(hierarchy=(("province",),))
        )
        for forbidden in ("new_sRenewal", "occurrence_target", "unit_growth"):
            with self.subTest(forbidden=forbidden), self.assertRaises(ValueError):
                estimator.fit(
                    pd.DataFrame({"province": ["A"], forbidden: [1]}),
                    pd.Series([1]),
                )
            with self.subTest(forbidden=forbidden), self.assertRaises(ValueError):
                estimator.predict_proba(
                    pd.DataFrame({"province": ["A"], forbidden: [1]})
                )

    def test_empty_or_invalid_training_labels_are_rejected(self):
        estimator = HierarchicalOccurrenceEstimator(
            OccurrenceConfig(hierarchy=())
        )
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            estimator.fit(pd.DataFrame(index=[]), pd.Series(dtype=float))
        for labels in ([0, np.nan], [0, 2], [0, np.inf]):
            with self.subTest(labels=labels), self.assertRaisesRegex(
                ValueError, "binary"
            ):
                estimator.fit(
                    pd.DataFrame(index=pd.RangeIndex(len(labels))),
                    pd.Series(labels),
                )

    def test_training_rows_must_be_aligned(self):
        estimator = HierarchicalOccurrenceEstimator(
            OccurrenceConfig(hierarchy=())
        )
        with self.assertRaisesRegex(ValueError, "row-aligned"):
            estimator.fit(
                pd.DataFrame(index=[0, 1]),
                pd.Series([0, 1], index=[1, 0]),
            )

    def test_predictions_are_deterministic_and_unfitted_use_fails_clearly(self):
        features, target = training_data(
            province=["A", "A", "B"],
            expiry_month=[1, 2, 1],
            labels=[1, 0, 1],
        )
        estimator = HierarchicalOccurrenceEstimator(
            OccurrenceConfig(hierarchy=(("province",),), shrinkage_strength=3)
        )
        candidates = pd.DataFrame({"province": ["A", "B", "unseen"]})

        with self.assertRaisesRegex(RuntimeError, "must be fitted"):
            estimator.predict_proba(candidates)

        estimator.fit(features, target)
        first = estimator.predict_proba(candidates)
        second = estimator.predict_proba(candidates)
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
