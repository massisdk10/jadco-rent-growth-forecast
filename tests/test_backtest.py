"""Tests synthétiques du protocole walk-forward et des baselines."""
import unittest
import numpy as np
import pandas as pd
from src.features.model_dataset import build_model_dataset, MAIN_TARGET, prediction_origin
from src.evaluation.backtest import (split_year, baseline_previous_year,
    baseline_recent_history, evaluate_predictions, run_backtest)
from test_model_dataset import leases_fixture


class BacktestTests(unittest.TestCase):
    def setUp(self):
        self.leases=leases_fixture()
        self.data,_=build_model_dataset(self.leases)

    def test_train_past_test_year_deterministic(self):
        for year in [2023,2024,2025]:
            train,test=split_year(self.data,year)
            self.assertTrue(train.year.lt(year).all())
            self.assertTrue(train.label_available_date.le(prediction_origin(year)).all())
            self.assertTrue(test.year.eq(year).all())
            self.assertFalse(set(train.index)&set(test.index))
            again=split_year(self.data,year)
            pd.testing.assert_frame_equal(train,again[0])
            pd.testing.assert_frame_equal(test,again[1])

    def test_delayed_or_missing_label_excluded(self):
        data=self.data.copy();data.loc[data.year.eq(2024),'label_available_date']=pd.Timestamp('2026-01-01')
        train,_=split_year(data,2025)
        self.assertFalse(train.year.eq(2024).any())
        data['label_available_date']=pd.NaT
        self.assertTrue(split_year(data,2025)[0].empty)

    def test_old_effective_rent_must_also_be_mature(self):
        leases=self.leases.copy()
        leases.loc[leases.sLeaseFrom.eq('2023-01-01'),'sLeaseTo']='2026-12-31'
        data,_=build_model_dataset(leases)
        train,_=split_year(data,2025)
        self.assertFalse(train.year.eq(2024).any())

    def test_baselines_insensitive_to_future(self):
        train,_=split_year(self.data,2025)
        a=(baseline_previous_year(train,2025),baseline_recent_history(train,2025))
        modified=self.data.copy();modified.loc[modified.year.ge(2025),MAIN_TARGET]=999
        again,_=split_year(modified,2025)
        self.assertEqual(a,(baseline_previous_year(again,2025),baseline_recent_history(again,2025)))
        self.assertTrue(np.isnan(baseline_previous_year(train,2020)))

    def test_metrics_distinguish_individual_and_portfolio(self):
        test=pd.DataFrame({MAIN_TARGET:[0,.02,.04]})
        metrics=evaluate_predictions(test,[.02,.02,.02])
        self.assertAlmostEqual(metrics['MAE_pp'],4/3)
        self.assertAlmostEqual(metrics['RMSE_pp'],np.sqrt(8/3))
        self.assertEqual(metrics['portfolio_error_pp'],0)
        self.assertEqual(metrics['bias_pp'],0)

    def test_generic_interface_never_receives_new_bail(self):
        self.leases.loc[self.leases.sLeaseFrom.eq('2024-01-01'),'sLeaseTo']='2025-12-31'
        self.data,_=build_model_dataset(self.leases)
        def model(train,x,year,context):
            self.assertEqual(set(train.columns),{'forecast_year',MAIN_TARGET})
            self.assertEqual(set(x.columns),{'forecast_year'})
            self.assertTrue(train.forecast_year.lt(year).all())
            self.assertIsNone(context)
            return np.full(len(x),train[MAIN_TARGET].median())
        result=run_backtest(model,self.data,self.leases,2025,features=['forecast_year'])
        self.assertEqual(result['test_transitions'],1)
        with self.assertRaises(ValueError):
            run_backtest(model,self.data,self.leases,2025,features=['new_sRent'])

    def test_prediction_call_has_no_dependency_on_realized_test(self):
        self.leases.loc[self.leases.sLeaseFrom.eq('2024-01-01'),'sLeaseTo']='2025-12-31'
        data,_=build_model_dataset(self.leases)
        calls=[]
        def model(train,x,year,context):
            calls.append((train.copy(),x.copy()))
            self.assertNotIn(MAIN_TARGET,x)
            self.assertNotIn('sRenewal',x)
            return np.full(len(x),.03)
        first=run_backtest(model,data,self.leases,2025,features=['prior_contractual_rent'])
        past=self.leases.loc[self.leases.sLeaseFrom.lt('2025-01-01')]
        historical_data,_=build_model_dataset(past)
        second=run_backtest(model,historical_data,past,2025,features=['prior_contractual_rent'])
        pd.testing.assert_frame_equal(calls[0][0],calls[1][0])
        pd.testing.assert_frame_equal(calls[0][1],calls[1][1])
        self.assertEqual(first['candidate_units'],second['candidate_units'])
        self.assertEqual(second['realized_candidate_units'],0)
        self.assertEqual(second['test_transitions'],0)

    def test_occurrence_prediction_rejects_actual_occurrence(self):
        from src.evaluation.backtest import predict_occurrence,aggregate_expected_contribution
        history=pd.DataFrame({'province':['Quebec']*4,'occurrence_target':[1,1,1,0]})
        x=pd.DataFrame({'province':['Quebec','Ontario']})
        probabilities=predict_occurrence(history,x)
        np.testing.assert_allclose(probabilities,[.75,.75])
        np.testing.assert_allclose(predict_occurrence(history,x,'historical_province'),probabilities)
        with self.assertRaises(ValueError):predict_occurrence(history,x.assign(occurrence_target=[0,1]))
        with self.assertRaises(ValueError):predict_occurrence(history,x.assign(new_sRenewal=[1,0]))
        a=aggregate_expected_contribution(probabilities,[.04,.08])
        self.assertAlmostEqual(a,.045)
        self.assertEqual(a,aggregate_expected_contribution(probabilities,[.04,.08]))

    def test_portfolio_predictions_invariant_to_future_outcomes(self):
        from src.evaluation.backtest import run_portfolio_baselines
        raw=self.leases.copy()
        for y in range(2020,2026):
            raw.loc[raw.sLeaseFrom.eq(f'{y}-01-01'),'sLeaseTo']=f'{y+1}-01-01'
        a=run_portfolio_baselines(raw,2025)
        changed=raw.loc[raw.sLeaseFrom.lt('2025-01-01')].copy()
        b=run_portfolio_baselines(changed,2025)
        pd.testing.assert_series_equal(a.predicted_P1_pct,b.predicted_P1_pct)
        pd.testing.assert_series_equal(a.predicted_rate,b.predicted_rate)
        self.assertTrue(a.realized_units.gt(b.realized_units).all())


if __name__=='__main__':unittest.main()
