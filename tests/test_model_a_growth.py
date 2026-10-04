"""Tests croissance CORE, métriques et déterminisme des modèles fixes."""
import unittest
import numpy as np
from src.features.model_dataset import ADMISSIBLE_FEATURES
from src.models.model_a_growth import predict_growth_models
from src.evaluation.model_a_evaluation import growth_metrics
from test_model_a_occurrence import frames


class GrowthTests(unittest.TestCase):
    def test_models_deterministic_finite_and_no_mutation(self):
        train,test=frames();train=train[list(ADMISSIBLE_FEATURES)];test=test[list(ADMISSIBLE_FEATURES)]
        before=train.copy(deep=True)
        y=.01+.0001*np.arange(len(train));o=np.ones(len(train))
        a,models=predict_growth_models(train,y,o,test,2023)
        b,_=predict_growth_models(train,y,o,test,2023)
        self.assertEqual(set(a),{'historical_mean','historical_median','previous_year_median','Ridge','ElasticNet','HGB'})
        for name,p in a.items():
            self.assertTrue(np.isfinite(p).all());np.testing.assert_allclose(p,b[name],rtol=0,atol=0)
        self.assertTrue(train.equals(before))
        for name,model in models.items():
            prep=model.named_steps['preprocessing'];i=prep.transformers_[0][2].index('sqft')
            self.assertAlmostEqual(prep.named_transformers_['num'].named_steps['imputation'].statistics_[i],train.sqft.median())
            self.assertNotIn('INCONNU',prep.named_transformers_['cat'].named_steps['encodage'].categories_[1])

    def test_only_positive_transitions_and_core_allowed(self):
        train,test=frames();train=train[list(ADMISSIBLE_FEATURES)];test=test[list(ADMISSIBLE_FEATURES)]
        y=np.full(len(train),.02);o=np.ones(len(train));o[0]=0
        with self.assertRaises(ValueError):predict_growth_models(train,y,o,test,2023)
        with self.assertRaises(ValueError):predict_growth_models(train.assign(expiry_month=1),y,np.ones(len(train)),test,2023)
        with self.assertRaises(ValueError):predict_growth_models(train.assign(previous_same_unit_growth=.01),y,np.ones(len(train)),test,2023)

    def test_no_2026_or_future_training(self):
        train,test=frames();train=train[list(ADMISSIBLE_FEATURES)];test=test[list(ADMISSIBLE_FEATURES)]
        y=np.full(len(train),.02);o=np.ones(len(train))
        with self.assertRaises(ValueError):predict_growth_models(train,y,o,test,2026)
        with self.assertRaises(ValueError):predict_growth_models(train.assign(forecast_year=2023),y,o,test,2023)

    def test_small_train_baselines_without_invented_ml(self):
        train,test=frames();train=train[list(ADMISSIBLE_FEATURES)].iloc[:10];test=test[list(ADMISSIBLE_FEATURES)]
        p,models=predict_growth_models(train,np.full(10,.03),np.ones(10),test,2023)
        self.assertFalse(models);self.assertEqual(set(p),{'historical_mean','historical_median','previous_year_median'})

    def test_metrics_in_percentage_points(self):
        m=growth_metrics([.01,.03],[.02,.01])
        self.assertAlmostEqual(m['mae_pp'],1.5)
        self.assertAlmostEqual(m['rmse_pp'],np.sqrt(2.5))
        self.assertAlmostEqual(m['bias_pp'],-.5)
        self.assertAlmostEqual(m['median_prediction'],1.5)
        with self.assertRaises(ValueError):growth_metrics([np.nan],[.01])

    def test_missing_train_columns_remain_predictable(self):
        train,test=frames();train=train[list(ADMISSIBLE_FEATURES)];test=test[list(ADMISSIBLE_FEATURES)]
        train['sqft']=np.nan;train['building']=np.nan
        p,_=predict_growth_models(train,np.full(len(train),.02),np.ones(len(train)),test,2023)
        self.assertTrue(all(np.isfinite(v).all() for v in p.values()))


if __name__=='__main__':unittest.main()
