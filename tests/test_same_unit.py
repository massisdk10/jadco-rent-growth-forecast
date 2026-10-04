"""Tests ciblés avec données synthétiques ; aucun enregistrement confidentiel."""
import hashlib
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from src.same_unit import UNIT_KEY, build_same_unit_pairs


def sample():
    """Trois baux synthétiques d'une unité ; codes avec zéros initiaux."""
    return pd.DataFrame({'sPropCode':['P1']*3, 'sUnitCode':['001']*3,
                         'sSite':['SITE']*3,
                         'sLeaseFrom':['2020-01-01','2021-01-01','2022-01-01'],
                         'sRent':[1000,1100,1210], 'sRentEffective':[900,945,1008],
                         'sTermSeq':[1,2,3], 'sTermMonths':[12,12,12], 'sRenewal':[0,1,0]})


class SameUnitTests(unittest.TestCase):
    def test_two_leases_and_n_minus_one(self):
        d=sample()
        self.assertEqual(len(build_same_unit_pairs(d.iloc[:2])[0]),1)
        self.assertEqual(len(build_same_unit_pairs(d)[0]),2)

    def test_official_key_and_no_cross_unit(self):
        d=sample().iloc[:2].copy()
        d['sPropCode']=['P1','P2']
        self.assertEqual(UNIT_KEY,('sPropCode','sUnitCode'))
        self.assertEqual(len(build_same_unit_pairs(d)[0]),0)
        d['sPropCode']='P1'; d['sUnitCode']=['001','002']
        self.assertEqual(len(build_same_unit_pairs(d)[0]),0)

    def test_chronology_and_consecutive_prices(self):
        p,_=build_same_unit_pairs(sample().iloc[::-1])
        self.assertTrue((p.new_lease_date > p.old_lease_date).all())
        self.assertEqual(p.old_sRent.tolist(),[1000,1100])
        self.assertEqual(p.new_sRent.tolist(),[1100,1210])
        self.assertTrue(p.flag_same_key.all())
        self.assertEqual(p.sUnitCode.tolist(),['001','001'])

    def test_both_growths_recomputable_and_annualized(self):
        p,_=build_same_unit_pairs(sample())
        for target,col in [('contractual','sRent'),('effective','sRentEffective')]:
            ratio=p['new_'+col]/p['old_'+col]
            np.testing.assert_allclose(p[target+'_growth'],ratio-1)
            np.testing.assert_allclose(p[target+'_annualized_growth'],ratio**(12/p.months_between)-1)

    def test_invalid_price_is_not_skipped(self):
        for value in [0,-1,'invalide',None,np.inf]:
            with self.subTest(value=value):
                d=sample(); d['sRent']=d['sRent'].astype(object); d.loc[1,'sRent']=value
                p,_=build_same_unit_pairs(d)
                self.assertEqual(len(p),2)
                self.assertTrue(p.contractual_growth.isna().all())
                self.assertTrue(p.effective_growth.notna().all())
                self.assertFalse(np.isinf(p.contractual_annualized_growth).any())

    def test_invalid_date_quarantines_unit_without_bridge(self):
        d=sample(); d.loc[1,'sLeaseFrom']='date invalide'
        p,t=build_same_unit_pairs(d)
        self.assertEqual(len(p),0)
        self.assertEqual(t['unconstructed_date_transitions'],2)
        self.assertEqual(t['invalid_start_date_leases'],1)

    def test_equal_dates_are_flagged_including_adjacent_transition(self):
        d=sample(); d.loc[1,'sLeaseFrom']=d.loc[0,'sLeaseFrom']
        p,_=build_same_unit_pairs(d)
        self.assertEqual(len(p),2)
        self.assertTrue(p.flag_ambiguous_date.all())
        self.assertTrue(p.contractual_growth.isna().all())

    def test_missing_key_is_traced(self):
        d=sample(); d.loc[1,'sPropCode']=None
        p,t=build_same_unit_pairs(d)
        self.assertEqual(t['missing_key_leases'],1)
        self.assertEqual(len(p),1)

    def test_numeric_identifiers_rejected(self):
        d=sample(); d['sUnitCode']=1
        with self.assertRaises(ValueError): build_same_unit_pairs(d)

    def test_duplicate_and_extreme_not_deleted(self):
        d=sample(); d.loc[2,'sRent']=100000
        p,_=build_same_unit_pairs(d)
        self.assertEqual(len(p),2)
        self.assertGreater(p.contractual_growth.max(),50)
        d=pd.concat([sample(),sample().iloc[:1]],ignore_index=True)
        p,_=build_same_unit_pairs(d)
        self.assertEqual(len(p),3)
        self.assertTrue(p.flag_duplicate_source_row.any())

    def test_input_and_raw_files_unchanged(self):
        d=sample(); original=d.copy(deep=True)
        paths=list((Path(__file__).resolve().parents[1]/'data/raw').glob('*.csv'))
        hashes={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        build_same_unit_pairs(d)
        pd.testing.assert_frame_equal(d,original)
        self.assertEqual(hashes,{p:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


if __name__ == '__main__':
    unittest.main()
