"""Assemblage, récence et refus des informations futures."""
import json
import unittest
import numpy as np
import pandas as pd
from src.models.model_a import (recency_weights, model_a_config, assemble_p1,
                               predict_model_a, predict_from_fold)


class ModelATests(unittest.TestCase):
    def test_equal_and_half_lives(self):
        years=[2019,2020,2021,2022];origin=pd.Timestamp('2022-12-31')
        np.testing.assert_array_equal(recency_weights(years,origin,'equal'),np.ones(4))
        age=(origin-pd.to_datetime([f'{y}-12-31' for y in years])).days/365.25
        for variant,half_life in [('mild',3),('moderate',1.5)]:
            expected=np.exp(-np.log(2)*age/half_life)
            np.testing.assert_allclose(recency_weights(years,origin,variant),expected)
            self.assertEqual(recency_weights(years,origin,variant)[-1],1)

    def test_future_year_rejected_even_equal(self):
        for variant in ('equal','mild','moderate'):
            with self.assertRaises(ValueError):recency_weights([2023],'2022-12-31',variant)
        with self.assertRaises(ValueError):recency_weights([2021.5],'2022-12-31')
        with self.assertRaises(ValueError):recency_weights([2021],'2022-12-31','inconnue')

    def test_p1_contributions_and_median(self):
        p=np.array([.5,.8,1]);g=np.array([.02,.03,-.01])
        result=assemble_p1(p,g)
        np.testing.assert_allclose(result['contributions'],[.01,.024,-.01])
        self.assertAlmostEqual(result['p1_pct'],1)
        with self.assertRaises(ValueError):assemble_p1([1.1],[.02])

    def test_configuration_json_core_only(self):
        config=model_a_config(recency_variant={'occurrence':'moderate','growth':'mild'})
        self.assertEqual(config,json.loads(json.dumps(config)))
        with self.assertRaises(ValueError):model_a_config(feature_set='enriched')
        with self.assertRaises(ValueError):model_a_config(growth_method='HGB',recency_variant='mild')
        with self.assertRaises(ValueError):model_a_config(occurrence_method='Logistic',recency_variant='mild')
        with self.assertRaises(ValueError):predict_model_a(pd.DataFrame(),2026)

    def test_equal_assembly_reproduces_original_baseline(self):
        y=np.array([0,1,1]);g=np.array([.01,.03])
        p=np.average(y,weights=recency_weights([2020,2021,2022],'2022-12-31'))
        mean=np.average(g,weights=recency_weights([2020,2021],'2022-12-31'))
        fold={'occurrence':{('historical_global','equal'):np.full(5,p)},
              'growth':{('historical_mean','equal'):np.full(5,mean)}}
        self.assertAlmostEqual(predict_from_fold(fold,model_a_config())['p1_pct'],100*y.mean()*g.mean())

    def test_future_injection_leaves_recency_predictions_unchanged(self):
        from test_model_a_features import fixture
        rows=[]
        for unit in range(6):
            for year in range(2018,2023):
                row=fixture().iloc[0].to_dict()
                row.update(sUnitCode=str(unit),sLeaseFrom=f'{year}-07-01',sSignDate=f'{year}-06-15',
                    sLeaseTo=f'{year+1}-06-30',sRentEffective=1000*(1.02**(year-2018)))
                rows.append(row)
        raw=pd.DataFrame(rows)
        for variant in ('equal','mild','moderate'):
            a=predict_model_a(raw,2023,recency_variant=variant)
            future=raw.iloc[:1].copy();future['sLeaseFrom']='2026-01-01';future['sSignDate']='2025-12-01'
            future['sLeaseTo']='2026-12-31';future['sRentEffective']=999999
            b=predict_model_a(pd.concat([raw,future],ignore_index=True),2023,recency_variant=variant)
            np.testing.assert_array_equal(a['contributions'],b['contributions'])
            self.assertEqual(a['p1_pct'],b['p1_pct'])


if __name__=='__main__':unittest.main()
