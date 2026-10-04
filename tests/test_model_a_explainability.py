"""Rapports JSON agrégés, sécurité des sorties et schéma comparable A/B."""
import json
from pathlib import Path
import tempfile
import unittest
import pandas as pd
from src.models.model_a import estimate_2026
from src.explainability.model_a_explainability import export_model_a_reports,validate_aggregate_payload


class ExplainabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw=pd.read_csv(Path(__file__).resolve().parents[1]/'data/raw/equinoxe_lease_history.csv',dtype={'sPropCode':'string','sUnitCode':'string'})
        cls.report=estimate_2026(raw,None,return_details=True)

    def test_json_and_comparison_schema(self):
        r=json.loads(json.dumps(self.report,ensure_ascii=False,allow_nan=False))
        self.assertEqual(r,self.report)
        self.assertTrue(set(['method_name','forecast_year','predicted_growth','target_definition','backtest_by_year',
                             'aggregate_metrics','uncertainty','drivers','risk_factors','limitations']).issubset(r))
        self.assertLessEqual(len(r['drivers']),5)
        self.assertEqual(r['components']['constant_across_units'],True)

    def test_export_contains_only_aggregates_and_no_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            paths=export_model_a_reports(self.report,directory)
            self.assertEqual(len(paths),3)
            for p in paths:
                content=Path(p).read_text()
                for forbidden in ['sPropCode','sUnitCode','UNIT_KEY','sRentEffective','contributions','probabilities']:
                    self.assertNotIn(forbidden,content)
                validate_aggregate_payload(json.loads(content))
            self.assertEqual(json.loads(Path(paths[-1]).read_text()),self.report)

    def test_record_level_or_nonfinite_data_refused(self):
        for bad in [{'UNIT_KEY':['P','U']},{'sUnitCode':'001'},{'records':[{}]},{'x':pd.DataFrame({'a':[1]})},{'x':float('nan')}]:
            with self.assertRaises(ValueError):validate_aggregate_payload(bad)
        with tempfile.TemporaryDirectory() as d:
            bad=dict(self.report);bad['raw_records']=[]
            with self.assertRaises(ValueError):export_model_a_reports(bad,d)
            self.assertEqual(list(Path(d).iterdir()),[])

    def test_drivers_and_limits_are_honest(self):
        r=self.report
        self.assertIn('Probabilité historique',r['drivers'][0]['name'])
        self.assertIn('portefeuille',r['cohort']['segment_interpretation'])
        self.assertIn('2025',' '.join(r['risk_factors']))
        self.assertIn('rent roll',' '.join(r['limitations']))
        self.assertIn('confiance',' '.join(r['limitations']))
        self.assertIn('espérance',' '.join(r['limitations']))


if __name__=='__main__':unittest.main()
