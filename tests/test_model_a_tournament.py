"""Classement déterministe, équivalence pratique et références communes."""
import unittest
import numpy as np
import pandas as pd
from src.evaluation.model_a_tournament import rank_candidates,run_tournament
from src.models.model_a import model_a_config
from unittest.mock import patch


def summary():
    rows=[]
    for name,mae,complexity in [('complexe',.99,3),('simple',1.01,0),('lointain',1.3,0)]:
        rows.append({'candidate':name,'MAE_P1':mae,'RMSE_P1':1.2,'bias_P1':.1,
            'worst_year_absolute_error':1.5,'occurrence_brier':.04,'growth_mae_pp':4.2,
            'complexity_level':complexity,'interpretability_level':complexity,
            'max_degradation_vs_baseline_pp':0.0})
    return pd.DataFrame(rows)


class TournamentTests(unittest.TestCase):
    def test_practical_equivalence_prefers_simple_and_deterministic(self):
        a=rank_candidates(summary());b=rank_candidates(summary().iloc[::-1])
        self.assertEqual(a.candidate.tolist(),b.candidate.tolist())
        self.assertEqual(a.candidate.iloc[0],'simple')
        changed=summary();changed.loc[changed.candidate.eq('complexe'),'MAE_P1']=.8
        self.assertEqual(rank_candidates(changed).candidate.iloc[0],'complexe')

    def test_worst_year_breaks_equal_complexity_tie(self):
        data=summary().iloc[:2].copy();data['complexity_level']=0;data['interpretability_level']=0
        data.loc[data.candidate.eq('simple'),'worst_year_absolute_error']=1.4
        self.assertEqual(rank_candidates(data).candidate.iloc[0],'simple')

    def test_one_bad_year_prevents_promotion(self):
        data=summary();data.loc[data.candidate.eq('complexe'),'max_degradation_vs_baseline_pp']=.6
        ranked=rank_candidates(data)
        self.assertFalse(ranked.loc[ranked.candidate.eq('complexe'),'eligible_for_promotion'].iloc[0])

    def test_tournament_uses_only_three_years_and_identical_references(self):
        calls=[]
        labels=pd.DataFrame({'transition_occurred':[0,1,1,1,1],
                             'conditional_growth':[np.nan,.01,.02,.03,.04]})
        def fold(leases,year):
            calls.append(year)
            return {'occurrence':{(m,v):np.full(5,.9) for m in ['historical_global','Logistic'] for v in ['equal','mild','moderate']},
                    'growth':{(m,v):np.full(5,.02) for m in ['historical_mean','historical_median','Ridge','HGB'] for v in ['equal','mild','moderate']}}
        with patch('src.evaluation.model_a_tournament.assert_foundation_reference',return_value=pd.DataFrame()), \
             patch('src.evaluation.model_a_tournament.fold_predictions',side_effect=fold), \
             patch('src.evaluation.model_a_tournament.build_model_a_features') as build:
            build.return_value.labels=labels
            r=run_tournament(pd.DataFrame())
        self.assertEqual(calls,[2023,2024,2025])
        self.assertEqual(set(r['annual'].year),{2023,2024,2025})
        self.assertTrue(r['annual'].groupby('year').realized_P1_pct.nunique().eq(1).all())
        self.assertEqual(r['MODEL_A_CHAMPION']['feature_set'],'core')
        for config in r['configs'].values():self.assertEqual(config,model_a_config(**config))
        import json
        json.dumps(r['explainability'],allow_nan=False)


if __name__=='__main__':unittest.main()
