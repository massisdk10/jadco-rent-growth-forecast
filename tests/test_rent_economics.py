"""Tests économiques ciblés sur des données synthétiques uniquement."""
import hashlib
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from src.rent_economics import calculate_gaps, attach_concessions, reconstruct_candidates, error_metrics


def fixtures():
    """Deux propriétés avec même numéro d'unité, une concession chacun."""
    l=pd.DataFrame({'sPropCode':['P1','P2'],'sUnitCode':['001','001'],
        'sLeaseFrom':['2020-01-01']*2,'sLeaseTo':['2020-12-31']*2,
        'sRent':[1200,1500],'sRentEffective':[1100,1450],'sTermMonths':[12,12],
        'sConcession':[1,1]})
    c=pd.DataFrame({'sPropCode':['P1','P1','P2'],'sUnitCode':['001']*3,
        'sDateFrom':['2020-01-01']*3,'sDateTo':['2020-01-31','2020-12-31','2020-12-31'],
        'sAmount':[-1200,-10,-50],'sMonths':[1,12,12],
        'sChargeCode':['PromoPay','freepark','freepark']})
    return l,c


class RentEconomicsTests(unittest.TestCase):
    def test_absolute_and_relative_gap(self):
        l,_=fixtures();o=calculate_gaps(l)
        np.testing.assert_allclose(o.contractual_effective_gap,[100,50])
        np.testing.assert_allclose(o.relative_gap,[100/1200,50/1500])

    def test_aggregation_no_multiplication_official_key(self):
        l,c=fixtures();o,links,r=attach_concessions(l,c)
        self.assertEqual(len(o),2)
        self.assertEqual(o.detail_count.tolist(),[2,1])
        self.assertEqual(o.signed_line_total.tolist(),[-1210,-50])
        self.assertEqual(o.signed_monthly_total.tolist(),[-1320,-600])
        self.assertEqual(o.sUnitCode.tolist(),['001','001'])
        self.assertEqual(len(links),3)
        self.assertEqual(r['ambiguous_concessions'],0)

    def test_reconstructions_and_metrics(self):
        l,c=fixtures();o,_,_=attach_concessions(l,c);o=reconstruct_candidates(o)
        self.assertAlmostEqual(o.reconstructed_line_total.iloc[0],1200-1210/12)
        self.assertAlmostEqual(o.reconstructed_monthly_all.iloc[0],1090)
        self.assertAlmostEqual(o.reconstructed_promo_once.iloc[1],1450)
        m=error_metrics(o,min_count=1).set_index('candidate')
        self.assertAlmostEqual(m.loc['monthly_all','MAE ($)'],5)
        self.assertAlmostEqual(m.loc['monthly_all','biais moyen ($)'],-5)

    def test_indicator_candidate_does_not_force_unknown(self):
        l,c=fixtures();l.loc[0,'sConcession']=0;l.loc[1,'sConcession']=2
        o,_,_=attach_concessions(l,c);o=reconstruct_candidates(o)
        self.assertEqual(o.reconstructed_indicator_monthly.iloc[0],1200)
        self.assertTrue(pd.isna(o.reconstructed_indicator_monthly.iloc[1]))

    def test_no_division_zero_and_nonnumeric(self):
        l,c=fixtures();l.loc[0,'sRent']=0;l.loc[0,'sTermMonths']=0
        o,_,_=attach_concessions(l,c);o=reconstruct_candidates(o)
        self.assertTrue(pd.isna(o.relative_gap.iloc[0]))
        self.assertTrue(pd.isna(o.reconstructed_monthly_all.iloc[0]))
        self.assertFalse(np.isinf(o.reconstructed_monthly_all).any())
        l['sRent']=l.sRent.astype(object);l.loc[0,'sRent']='invalide'
        self.assertTrue(pd.isna(calculate_gaps(l).relative_gap.iloc[0]))

    def test_crossing_ambiguous_unmatched_not_forced(self):
        l,c=fixtures();c.loc[0,'sDateTo']='2021-01-31'
        o,_,r=attach_concessions(l,c)
        self.assertFalse(o.eligible_details.iloc[0])
        self.assertEqual(r['crossing_assigned_concessions'],1)
        self.assertTrue(pd.isna(reconstruct_candidates(o).reconstructed_monthly_all.iloc[0]))
        duplicated=pd.concat([l,l.iloc[:1]],ignore_index=True)
        o,_,r=attach_concessions(duplicated,c)
        self.assertEqual(r['ambiguous_concessions'],2)
        self.assertEqual(len(o),3)
        c.loc[0,'sPropCode']='INCONNUE'
        _,_,r=attach_concessions(l,c)
        self.assertEqual(r['unmatched_valid_concessions'],1)

    def test_no_detail_is_unknown_not_zero(self):
        l,c=fixtures();o,_,_=attach_concessions(l,c.iloc[:2])
        self.assertTrue(pd.isna(o.signed_line_total.iloc[1]))
        self.assertTrue(o.flag_indicator_disagreement.iloc[1])
        self.assertTrue(pd.isna(reconstruct_candidates(o).reconstructed_promo_once.iloc[1]))

    def test_handles_invalid_dates_and_values_flagged(self):
        l,c=fixtures();l['hUnit']=['1','2'];c['hUnit']=['9','1','2']
        o,_,r=attach_concessions(l,c)
        self.assertEqual(r['handle_mismatch_links'],1)
        self.assertFalse(o.eligible_details.iloc[0])
        c.loc[1,'sMonths']=0
        o,_,_=attach_concessions(l,c)
        self.assertFalse(o.eligible_details.iloc[0])
        c.loc[0,'sDateFrom']='date invalide'
        _,_,r=attach_concessions(l,c)
        self.assertEqual(r['invalid_concession_key_or_dates'],1)

    def test_inputs_and_raw_unchanged(self):
        l,c=fixtures();old_l=l.copy(deep=True);old_c=c.copy(deep=True)
        paths=list((Path(__file__).resolve().parents[1]/'data/raw').glob('*.csv'))
        hashes={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        o,_,_=attach_concessions(l,c);reconstruct_candidates(o)
        pd.testing.assert_frame_equal(l,old_l);pd.testing.assert_frame_equal(c,old_c)
        self.assertEqual(hashes,{p:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


if __name__=='__main__':unittest.main()
