"""Validation du forecast final, cohorte officielle et invariance au futur."""
from pathlib import Path
import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from src.models.model_a import estimate_2026,backtest,champion_config,_champion_components


class FinalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw=pd.read_csv(Path(__file__).resolve().parents[1]/'data/raw/equinoxe_lease_history.csv',dtype={'sPropCode':'string','sUnitCode':'string'})
        cls.report=estimate_2026(cls.raw,None,return_details=True)

    def test_cohort_counts_and_expiry(self):
        c=self.report['cohort']
        self.assertEqual(c['n_units'],931)
        self.assertEqual(c['province'],{'Quebec':810,'Ontario':121})
        self.assertEqual(c['building'],{'Daniel-Johnson':128,'Le Carlyle':180,'Levesque':75,'Saint-Elzear':268,'The Met':121,'Westpark':159})
        self.assertEqual([c['expiry_month'][str(i)] for i in range(1,13)],[38,52,65,62,91,143,90,91,108,61,69,61])

    def test_champion_exact_and_runner_cannot_replace_it(self):
        self.assertEqual(champion_config(),{'occurrence_method':'historical_global','growth_method':'historical_median',
            'recency_variant':{'occurrence':'equal','growth':'equal'},'feature_set':'core'})
        self.assertEqual(self.report['architecture'],champion_config())
        self.assertEqual(self.report['predicted_growth'],self.report['forecast']['base'])
        self.assertNotEqual(self.report['predicted_growth'],self.report['runner_up_sensitivity']['forecast_pct'])
        copy=champion_config();copy['growth_method']='HGB'
        self.assertEqual(champion_config()['growth_method'],'historical_median')

    def test_no_2026_labels_no_enriched_and_p1(self):
        c,h,g,pending,result=_champion_components(self.raw,2026)
        self.assertIsNone(c.labels);self.assertEqual(c.feature_set,'core')
        self.assertEqual(c.X.shape,(931,13));self.assertEqual(c.X_growth.shape,(931,12))
        self.assertTrue(h.forecast_year.lt(2026).all())
        np.testing.assert_allclose(result['contributions'],result['probabilities']*result['conditional_growth'])
        self.assertAlmostEqual(100*np.median(result['contributions']),self.report['predicted_growth'])
        self.assertTrue(((result['probabilities']>=0)&(result['probabilities']<=1)).all())

    def test_future_injection_no_effect_and_raw_not_modified(self):
        before=self.raw.copy(deep=True)
        future=self.raw.iloc[:1].copy();future['sLeaseFrom']='2026-07-01';future['sSignDate']='2026-06-15'
        future['sLeaseTo']='2027-06-30';future['sRentEffective']=999999
        r=estimate_2026(pd.concat([self.raw,future],ignore_index=True),None,return_details=True)
        self.assertEqual(r,self.report)
        pd.testing.assert_frame_equal(self.raw,before)

    def test_starter_api_and_ignored_unqualified_inputs(self):
        class ForbiddenInput:
            def __getattribute__(self,name):raise AssertionError('Source non qualifiée consultée')
        self.assertEqual(estimate_2026(self.raw,ForbiddenInput(),ForbiddenInput()),self.report['predicted_growth'])
        with self.assertRaises(ValueError):backtest(self.raw,2026)

    def test_backtests_reproduce_selection(self):
        predicted=[2.8700597508977657,3.2691787873371383,2.9693278810997157]
        actual=[2.0361023490776855,1.607852406246998,3.3008784572648096]
        for i,y in enumerate([2023,2024,2025]):
            r=backtest(self.raw,y)
            self.assertAlmostEqual(r['predicted_P1_pct'],predicted[i])
            self.assertAlmostEqual(r['realized_P1_pct'],actual[i])
            self.assertEqual(r['observed_units'],[401,634,773][i])
            self.assertEqual(r['configuration'],champion_config())

    def test_scenarios_seed_and_envelope_unchanged(self):
        r=estimate_2026(self.raw,None,return_details=True)
        self.assertEqual(r,self.report)
        u=r['uncertainty'];f=r['forecast']
        self.assertLessEqual(f['low'],f['base']);self.assertLessEqual(f['base'],f['high'])
        self.assertEqual(u['bootstrap_width_pp'],0)
        self.assertEqual(u['random_state'],42);self.assertEqual(u['n_bootstrap'],1000)
        self.assertAlmostEqual(u['model_error_radius_pp'],1.6613263810901402)

    def test_cohort_mismatch_stops_and_accents_do_not_split(self):
        raw=self.raw.copy();raw['sBuilding']=raw.sBuilding.replace({'Levesque':'Lévesque','Saint-Elzear':'Saint-Elzéar'})
        self.assertEqual(estimate_2026(raw,None),self.report['predicted_growth'])
        raw=self.raw.copy();raw['sState']='Ontario'
        with self.assertRaises(AssertionError):estimate_2026(raw,None)


if __name__=='__main__':unittest.main()
