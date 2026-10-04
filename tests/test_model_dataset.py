"""Tests synthétiques des cibles, du snapshot et des agrégations."""
import unittest
import numpy as np
import pandas as pd
from src.features.model_dataset import (build_model_dataset, build_features,
    validate_features, aggregate_growth, known_building_weights, available_external_context, build_prediction_cohort)


def leases_fixture():
    return pd.DataFrame([
        {'sPropCode':'p1','sUnitCode':'001','sLeaseFrom':f'{y}-01-01',
         'sLeaseTo':f'{y}-12-31','sSignDate':f'{y-1}-12-15',
         'sRent':1000*(1.1**(y-2020)), 'sRentEffective':900*(1.05**(y-2020)),
         'sConcession':1,'sTermMonths':12,'sTermSeq':y-2019,'sRenewal':1,
         'sBuilding':'B1','sState':'Quebec','sBeds':2,'sBaths':1,'sSqft':800,
         'sFloor':2,'sUnitSubtype':'A'} for y in range(2020,2026)])


class ModelDatasetTests(unittest.TestCase):
    def test_targets_reuse_exact_ratios(self):
        raw=leases_fixture();before=raw.copy(deep=True)
        data,_=build_model_dataset(raw)
        np.testing.assert_allclose(data.target_contractual_raw,.1)
        np.testing.assert_allclose(data.target_effective_raw,.05)
        expected=(1.05**(365.25/data.days_between)-1).to_numpy()
        np.testing.assert_allclose(data.target_effective_annualized,expected)
        pd.testing.assert_frame_equal(raw,before)

    def test_snapshot_ignores_future_and_actual_previous_test_lease(self):
        raw=leases_fixture();data,_=build_model_dataset(raw)
        test=data.loc[data.year.eq(2025)]
        a=build_features(raw,test,2025)
        raw.loc[raw.sLeaseFrom.ge('2025'),'sRent']=999999
        raw.loc[raw.sLeaseFrom.ge('2025'),'sSqft']=999999
        b=build_features(raw,test,2025)
        pd.testing.assert_frame_equal(a,b)
        self.assertAlmostEqual(a.prior_contractual_rent.iloc[0],1000*1.1**4)

    def test_signature_after_origin_and_unknown_unit_not_filled(self):
        raw=leases_fixture();raw.loc[raw.sLeaseFrom.eq('2024-01-01'),'sSignDate']='2025-01-02'
        observations=pd.DataFrame({'sPropCode':['p1','inconnu'],'sUnitCode':['001','001']})
        x=build_features(raw,observations,2025)
        self.assertAlmostEqual(x.prior_contractual_rent.iloc[0],1000*1.1**3)
        self.assertTrue(pd.isna(x.prior_contractual_rent.iloc[1]))
        self.assertTrue(pd.isna(x.property_code.iloc[1]))

    def test_forbidden_and_unknown_features_rejected(self):
        for col in ['new_sRent','sRenewal','old_sRentEffective','target_effective_raw','surprise']:
            with self.assertRaises(ValueError):validate_features([col])
        validate_features(['prior_contractual_rent','province'])

    def test_aggregation_and_fixed_weights(self):
        d=pd.DataFrame({'sBuilding':['A','A','B']})
        v=np.array([.01,.03,.1])
        self.assertAlmostEqual(aggregate_growth(d,v)['growth'],.03)
        self.assertAlmostEqual(aggregate_growth(d,v,'building_equal')['growth'],.06)
        w=pd.Series({'A':.75,'B':.25})
        self.assertAlmostEqual(aggregate_growth(d,v,'known_unit_weights',w)['growth'],.04)
        missing=aggregate_growth(d.iloc[:2],v[:2],'known_unit_weights',w)
        self.assertEqual(missing['weight_coverage'],.75)
        empty=aggregate_growth(d,v,'known_unit_weights',pd.Series(dtype=float))
        self.assertTrue(np.isnan(empty['growth']))
        self.assertEqual(empty['weight_coverage'],0)
        raw=leases_fixture();a=known_building_weights(raw,2025)
        raw.loc[raw.sLeaseFrom.ge('2025'),'sBuilding']='FUTUR'
        pd.testing.assert_series_equal(a,known_building_weights(raw,2025))

    def test_external_false_missing_and_future_excluded(self):
        data=pd.DataFrame({'available_date':['2024-01-01','2025-01-01',None,'2024-01-01'],
                           'availability_verified':[True,True,True,False], 'value':[1,2,3,4]})
        result=available_external_context(data,2025)
        self.assertEqual(result.index.tolist(),[0])

    def test_cutoff_cohort_ignores_all_new_leases(self):
        raw=leases_fixture()
        raw.loc[raw.sLeaseFrom.eq('2024-01-01'),'sLeaseTo']='2025-12-31'
        expected=build_prediction_cohort(raw,2025)
        self.assertEqual(len(expected),1)
        future=raw.loc[raw.sLeaseFrom.ge('2025-01-01')].copy()
        past=raw.loc[raw.sLeaseFrom.lt('2025-01-01')].copy()
        pd.testing.assert_frame_equal(expected,build_prediction_cohort(past,2025))
        future['sUnitCode']='future';future['sRent']=999;future['sRenewal']=0
        future['sLeaseTo']='2025-01-10'
        pd.testing.assert_frame_equal(expected,build_prediction_cohort(pd.concat([past,future]),2025))
        self.assertFalse(any('target' in c or 'new_' in c for c in expected.columns))
        self.assertTrue(expected.known_sign_date.le(pd.Timestamp('2024-12-31')).all())

    def test_invalid_or_unannounced_expiry_not_imputed(self):
        raw=leases_fixture().iloc[:1].copy()
        raw['sLeaseTo']='2025-12-31'
        self.assertEqual(len(build_prediction_cohort(raw,2025)),1)
        raw['sSignDate']='2025-01-01'
        self.assertTrue(build_prediction_cohort(raw,2025).empty)
        raw['sSignDate']='2019-12-15';raw['sLeaseTo']=None
        self.assertTrue(build_prediction_cohort(raw,2025).empty)

    def test_occurrence_labels_revealed_on_fixed_cohort(self):
        from src.features.model_dataset import reveal_occurrence,build_occurrence_features
        raw=leases_fixture()
        raw.loc[raw.sLeaseFrom.eq('2024-01-01'),'sLeaseTo']='2025-12-31'
        cohort=build_prediction_cohort(raw,2025)
        data,_=build_model_dataset(raw)
        positive=reveal_occurrence(cohort,data,2025)
        self.assertEqual(positive.occurrence_target.tolist(),[1])
        empty=data.loc[data.year.lt(2025)]
        negative=reveal_occurrence(cohort,empty,2025)
        self.assertEqual(negative.occurrence_target.tolist(),[0])
        x=build_occurrence_features(raw,cohort,2025)
        self.assertNotIn('occurrence_target',x)
        self.assertNotIn('unit_growth',x)
        self.assertNotIn('sRenewal',x)
        self.assertEqual(x.expiry_month.tolist(),[12])
        self.assertTrue(reveal_occurrence(cohort,data,2025,label_cutoff='2024-12-31').occurrence_target.isna().all())

    def test_2026_cohort_ignores_post_cutoff_records(self):
        raw=leases_fixture()
        raw.loc[raw.sLeaseFrom.eq('2025-01-01'),'sLeaseTo']='2026-12-31'
        cohort=build_prediction_cohort(raw,2026)
        future=raw.iloc[-1:].copy();future['sLeaseFrom']='2026-01-01'
        future['sSignDate']='2025-12-15';future['sUnitCode']='999'
        pd.testing.assert_frame_equal(cohort,build_prediction_cohort(pd.concat([raw,future]),2026))
        self.assertEqual(len(cohort),1)
        self.assertTrue(cohort.known_lease_start.le(pd.Timestamp('2025-12-31')).all())


if __name__=='__main__':unittest.main()
