"""Synthetic tests for Model B's median-based growth estimator."""

import unittest

import numpy as np
import pandas as pd

from src.features.model_dataset import MAIN_TARGET
from src.models.model_b.growth import GrowthConfig, HierarchicalGrowthEstimator


def growth_target(values, index=None):
    return pd.Series(values, index=index, name=MAIN_TARGET)


class GrowthEstimatorTests(unittest.TestCase):
    def test_public_interface_and_supported_hierarchies(self):
        hierarchies = (
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
        )
        for hierarchy in hierarchies:
            with self.subTest(hierarchy=hierarchy):
                config = GrowthConfig(hierarchy=hierarchy)
                estimator = HierarchicalGrowthEstimator(config)
                self.assertEqual(estimator.config, config)
                self.assertTrue(callable(estimator.fit))
                self.assertTrue(callable(estimator.predict))

    def test_global_prediction_is_training_median(self):
        features = pd.DataFrame(index=range(5))
        target = growth_target([0.01, 0.02, 0.03, 0.04, 0.50], features.index)
        estimator = HierarchicalGrowthEstimator(
            GrowthConfig(hierarchy=(), shrinkage_strength=2)
        ).fit(features, target)

        predictions = estimator.predict(pd.DataFrame(index=[10, 11]))

        np.testing.assert_allclose(predictions, [0.03, 0.03])

    def test_child_median_is_shrunk_toward_global_parent(self):
        features = pd.DataFrame(
            {"province": ["A", "A", "B", "B", "B", "B"]}
        )
        target = growth_target([0.2, 0.4, 0.0, 0.0, 0.0, 0.0], features.index)
        estimator = HierarchicalGrowthEstimator(
            GrowthConfig(
                hierarchy=(("province",),),
                shrinkage_strength=2,
                minimum_segment_size=1,
            )
        ).fit(features, target)

        # Global median is zero; A's median is .3 with n=2, so its weight is 1/2.
        prediction = estimator.predict(pd.DataFrame({"province": ["A"]}))

        np.testing.assert_allclose(prediction, [0.15])

    def test_larger_segment_sample_has_more_influence(self):
        def prediction_for_child_size(child_size):
            features = pd.DataFrame(
                {
                    "province": ["other"] * 10 + ["child"] * child_size,
                }
            )
            target = growth_target(
                [0.0] * 10 + [0.4] * child_size,
                features.index,
            )
            estimator = HierarchicalGrowthEstimator(
                GrowthConfig(
                    hierarchy=(("province",),),
                    shrinkage_strength=2,
                    minimum_segment_size=1,
                )
            ).fit(features, target)
            return estimator.predict(pd.DataFrame({"province": ["child"]}))[0]

        one_observation = prediction_for_child_size(1)
        five_observations = prediction_for_child_size(5)

        self.assertAlmostEqual(one_observation, 0.4 / 3)
        self.assertAlmostEqual(five_observations, 0.4 * 5 / 7)
        self.assertGreater(five_observations, one_observation)

    def test_recursive_shrinkage_and_parent_fallback(self):
        rows = []
        values = []
        for bedroom, rent_growth in ((1, 0.0), (2, 0.8)):
            for _ in range(2):
                rows.append(("P", "b1", bedroom))
                values.append(rent_growth)
        for _ in range(4):
            rows.append(("P", "b2", 1))
            values.append(0.4)
        for _ in range(4):
            rows.append(("Q", "b3", 1))
            values.append(0.0)
        features = pd.DataFrame(rows, columns=["province", "building", "bedrooms"])
        target = growth_target(values, features.index)
        estimator = HierarchicalGrowthEstimator(
            GrowthConfig(
                hierarchy=(
                    ("province",),
                    ("province", "building"),
                    ("province", "building", "bedrooms"),
                ),
                shrinkage_strength=2,
                minimum_segment_size=1,
            )
        ).fit(features, target)
        candidates = pd.DataFrame(
            {
                "province": ["P", "P", "P", "unknown", "P"],
                "building": ["b1", "b1", "missing", "missing", None],
                "bedrooms": [1, 3, 1, 1, 1],
            }
        )

        # global=.2; P=.36; P/b1=.386666...; P/b1/bedroom1=.193333...
        np.testing.assert_allclose(
            estimator.predict(candidates),
            [0.19333333333333333, 0.38666666666666666, 0.36, 0.2, 0.36],
        )

    def test_unseen_segments_fall_back_to_parent_then_global(self):
        features = pd.DataFrame(
            {
                "province": ["A"] * 4 + ["B"] * 4,
                "building": ["a"] * 4 + ["b"] * 4,
            }
        )
        target = growth_target([0.4] * 4 + [0.0] * 4, features.index)
        estimator = HierarchicalGrowthEstimator(
            GrowthConfig(
                hierarchy=(("province",), ("province", "building")),
                shrinkage_strength=2,
                minimum_segment_size=1,
            )
        ).fit(features, target)

        predictions = estimator.predict(
            pd.DataFrame(
                {
                    "province": ["A", "A", "unknown"],
                    "building": ["unseen", None, "a"],
                }
            )
        )

        # Global=.2; province A=.4*4/6 + .2*2/6 = 1/3.
        np.testing.assert_allclose(predictions, [1 / 3, 1 / 3, 0.2])

    def test_global_median_is_robust_to_an_extreme_outlier(self):
        features = pd.DataFrame(index=range(5))
        target = growth_target([0.01, 0.02, 0.03, 0.04, 100.0], features.index)
        estimator = HierarchicalGrowthEstimator(
            GrowthConfig(hierarchy=())
        ).fit(features, target)

        prediction = estimator.predict(pd.DataFrame(index=[0]))

        np.testing.assert_allclose(prediction, [0.03])

    def test_invalid_configuration_is_rejected(self):
        invalid_configs = (
            {"shrinkage_strength": 0},
            {"shrinkage_strength": float("inf")},
            {"shrinkage_strength": float("nan")},
            {"minimum_segment_size": 0},
            {"minimum_segment_size": True},
            {"hierarchy": (("province",), ("property_code",))},
            {"hierarchy": (("province", "bedrooms"),)},
            {"hierarchy": (("old_sRentEffective",),)},
        )
        for kwargs in invalid_configs:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                GrowthConfig(**kwargs)

    def test_empty_or_invalid_growth_training_data_is_rejected(self):
        config = GrowthConfig(hierarchy=())
        empty_features = pd.DataFrame(index=pd.Index([], dtype=int))
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            HierarchicalGrowthEstimator(config).fit(
                empty_features, growth_target([], empty_features.index)
            )

        features = pd.DataFrame(index=range(3))
        for values in ([0.1, np.nan, 0.2], [0.1, np.inf, 0.2], [0.1, "bad", 0.2]):
            with self.subTest(values=values), self.assertRaisesRegex(
                ValueError, "finite numeric"
            ):
                HierarchicalGrowthEstimator(config).fit(
                    features, growth_target(values, features.index)
                )

    def test_growth_target_must_be_the_frozen_main_target(self):
        features = pd.DataFrame(index=[0])
        wrong_target = pd.Series([0.1], index=features.index, name="other_target")

        with self.assertRaisesRegex(ValueError, MAIN_TARGET):
            HierarchicalGrowthEstimator(GrowthConfig(hierarchy=())).fit(
                features, wrong_target
            )

    def test_features_must_be_allowlisted_and_complete(self):
        estimator = HierarchicalGrowthEstimator(
            GrowthConfig(hierarchy=(("province",),))
        )
        features = pd.DataFrame({"province": ["A"]})
        estimator.fit(features, growth_target([0.1], features.index))

        with self.assertRaisesRegex(ValueError, "Required growth segment features"):
            estimator.predict(pd.DataFrame({"building": ["b1"]}))

        forbidden_frames = (
            pd.DataFrame({"province": ["A"], "target_effective_annualized": [0.1]}),
            pd.DataFrame({"province": ["A"], "new_sRentEffective": [1000.0]}),
            pd.DataFrame({"province": ["A"], "sRenewal": [1]}),
        )
        for forbidden in forbidden_frames:
            with self.subTest(columns=list(forbidden.columns)):
                with self.assertRaises(ValueError):
                    estimator.predict(forbidden)
                with self.assertRaises(ValueError):
                    HierarchicalGrowthEstimator(estimator.config).fit(
                        forbidden, growth_target([0.1], forbidden.index)
                    )

    def test_deterministic_predictions_and_unfitted_use_fails_clearly(self):
        estimator = HierarchicalGrowthEstimator(
            GrowthConfig(
                hierarchy=(("province",),),
                shrinkage_strength=3,
                minimum_segment_size=1,
            )
        )
        candidates = pd.DataFrame({"province": ["A", "unseen", None]})
        with self.assertRaisesRegex(RuntimeError, "must be fitted"):
            estimator.predict(candidates)

        features = pd.DataFrame({"province": ["A", "A", "B"]})
        estimator.fit(features, growth_target([0.1, 0.3, 0.0], features.index))
        first = estimator.predict(candidates)
        second = estimator.predict(candidates)

        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.isfinite(first).all())


if __name__ == "__main__":
    unittest.main()
