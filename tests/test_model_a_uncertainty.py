"""Bootstrap reproductible et scénarios sans couverture probabiliste inventée."""
import unittest
import numpy as np
import pandas as pd
from src.evaluation.model_a_uncertainty import bootstrap_p1, historical_error_envelope, uncertainty_scenarios


class UncertaintyTests(unittest.TestCase):
    def test_bootstrap_deterministic_no_mutation(self):
        c=np.array([.01,.02,.03,.04,.05]);before=c.copy()
        a=bootstrap_p1(c,1000,42);b=bootstrap_p1(c,1000,42)
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(c,before)
        self.assertTrue(np.isfinite(a).all())
        self.assertTrue(((a>=1)&(a<=5)).all())

    def test_constant_contributions_have_zero_bootstrap_width(self):
        np.testing.assert_allclose(bootstrap_p1(np.full(100,.03)),np.full(1000,3))

    def test_exact_scenario_formula(self):
        errors=pd.DataFrame({'year':[2023,2024,2025],'error_pp':[.8,1.7,-.2]})
        result=uncertainty_scenarios(np.full(10,.9),np.full(10,.03),errors)
        self.assertAlmostEqual(result['BASE'],2.7)
        self.assertAlmostEqual(result['LOW'],1)
        self.assertAlmostEqual(result['HIGH'],4.4)
        self.assertEqual(result['signed_errors_pp'],[.8,1.7,-.2])

    def test_future_or_duplicate_errors_rejected(self):
        for years in [[2024,2025,2026],[2023,2023,2025]]:
            with self.assertRaises(ValueError):historical_error_envelope(pd.DataFrame({'year':years,'error_pp':[1,2,3]}))
        with self.assertRaises(ValueError):bootstrap_p1([np.nan])
        with self.assertRaises(ValueError):bootstrap_p1([.01],random_state=None)


if __name__=='__main__':unittest.main()
