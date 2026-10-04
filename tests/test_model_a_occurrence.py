"""Tests synthétiques occurrence, coupures et preprocessing train uniquement."""
import unittest
import numpy as np
import pandas as pd
from src.features.model_dataset import OCCURRENCE_FEATURES, ADMISSIBLE_FEATURES
from src.models.model_a_occurrence import predict_occurrence_models
from src.evaluation.model_a_evaluation import temporal_training, p1_diagnostic
from src.evaluation.backtest import occurrence_metrics


def frames(n=60):
    train=pd.DataFrame({f:np.arange(n,dtype=float) for f in OCCURRENCE_FEATURES})
    for f in ['property_code','building','province','unit_subtype']:
        train[f]=['A','B']*(n//2)
    train['forecast_year']=2022
    train['sqft']=np.where(np.arange(n)%7==0,np.nan,np.arange(n)+100.)
    test=train.iloc[:8].copy().reset_index(drop=True)
    test['forecast_year']=2023
    test['sqft']=1000000.
    test['building']='INCONNU'
    return train,test


class OccurrenceTests(unittest.TestCase):
    def test_probabilities_deterministic_and_core_only(self):
        train,test=frames(); y=np.array([0,1]*30)
        a,models=predict_occurrence_models(train,y,test,2023)
        b,_=predict_occurrence_models(train,y,test,2023)
        for name,p in a.items():
            self.assertTrue(((p>=0)&(p<=1)).all())
            np.testing.assert_allclose(p,b[name],rtol=0,atol=0)
        self.assertEqual(set(a),{'historical_global','Logistic'})
        with self.assertRaises(ValueError):
            predict_occurrence_models(train.assign(conditional_growth=1),y,test,2023)
        with self.assertRaises(ValueError):
            predict_occurrence_models(train.assign(previous_same_unit_growth=1),y,test,2023)

    def test_preprocessing_uses_train_only(self):
        train,test=frames();y=np.array([0,1]*30)
        _,models=predict_occurrence_models(train,y,test,2023)
        prep=models['Logistic'].named_steps['preprocessing']
        numeric=prep.transformers_[0][2];i=numeric.index('sqft')
        self.assertAlmostEqual(prep.named_transformers_['num'].named_steps['imputation'].statistics_[i],train.sqft.median())
        scaler=prep.named_transformers_['num'].named_steps['standardisation']
        self.assertAlmostEqual(scaler.mean_[i],train.sqft.fillna(train.sqft.median()).mean())
        cat=prep.named_transformers_['cat'].named_steps['encodage']
        self.assertNotIn('INCONNU',cat.categories_[1])

    def test_2026_and_future_train_rejected(self):
        train,test=frames();y=np.array([0,1]*30)
        with self.assertRaises(ValueError):predict_occurrence_models(train,y,test,2026)
        with self.assertRaises(ValueError):predict_occurrence_models(train.assign(forecast_year=2023),y,test,2023)
        with self.assertRaises(ValueError):temporal_training(pd.DataFrame(),2026)

    def test_small_negative_support_has_baseline_only(self):
        train,test=frames();y=np.ones(60);y[:2]=0
        predictions,models=predict_occurrence_models(train,y,test,2023)
        self.assertEqual(set(predictions),{'historical_global'})
        self.assertFalse(models)
        self.assertAlmostEqual(predictions['historical_global'][0],58/60)

    def test_metrics_and_p1_synthetic(self):
        m=occurrence_metrics([0,1],[.2,.8])
        self.assertAlmostEqual(m['Brier'],.04)
        labels=pd.DataFrame({'transition_occurred':[0,1,1], 'conditional_growth':[np.nan,.02,.08]})
        m=p1_diagnostic(np.array([.5,.5,.5]),np.array([.04,.04,.04]),labels)
        self.assertAlmostEqual(m['predicted_P1_pct'],2)
        self.assertAlmostEqual(m['realized_P1_pct'],2)
        self.assertAlmostEqual(m['error_pp'],0)

    def test_temporal_history_future_injection_and_maturity(self):
        # Baux synthétiques successifs ; transition 2022 non mûre à D 2022.
        rows=[]
        for year in range(2018,2024):
            rows.append(dict(sPropCode='P',sUnitCode='U',sLeaseFrom=f'{year}-07-01',
                sSignDate=f'{year}-06-01',sLeaseTo=f'{year+1}-06-30',
                sRent=1000+year,sRentEffective=900+year,sBuilding='B',sState='Quebec',
                sBeds=2,sBaths=1,sSqft=1000,sFloor=1,sUnitSubtype='A',sTermMonths=12))
        raw=pd.DataFrame(rows)
        a,g,p=temporal_training(raw,2023)
        future=raw.iloc[-1:].copy();future['sLeaseFrom']='2024-07-01';future['sSignDate']='2024-06-01'
        future['sLeaseTo']='2025-06-30';future['sRentEffective']=999999
        b,h,q=temporal_training(pd.concat([raw,future],ignore_index=True),2023)
        pd.testing.assert_frame_equal(a,b);pd.testing.assert_frame_equal(g,h)
        self.assertTrue(a.forecast_year.lt(2023).all())
        self.assertFalse(g.forecast_year.eq(2022).any())
        self.assertGreater(p,0);self.assertEqual(p,q)


if __name__=='__main__':unittest.main()
