"""Tests synthétiques du shrinkage fixe et de l'isolement temporel de D."""
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest

from src.models.model_d import ALPHA, BuildingShrunkGrowth
from src.evaluation.model_d_evaluation import _predict_d_fold, run_model_d_experiment
from src.models.model_a import _champion_components
from tests.test_model_ab_tournament import synthetic_leases


def test_alpha_is_fixed_and_shrinkage_matches_support_rule():
    assert ALPHA == 30
    building = pd.Series(['A']*30+['B']*40)
    growth = pd.Series([.4]*30+[0.]*40)
    model = BuildingShrunkGrowth().fit(building, growth)
    assert model.global_growth == 0
    assert model.predict(pd.Series(['A']))[0] == pytest.approx(.2)
    row = model.diagnostics(pd.Series(['A']*5)).set_index('building').loc['A']
    assert row.w_b == .5
    assert row.n_b == 30


def test_unknown_null_and_blank_buildings_fall_back_to_global():
    model = BuildingShrunkGrowth().fit(pd.Series(['A']*5), pd.Series([-.1, 0, .01, .02, .3]))
    np.testing.assert_array_equal(model.predict(pd.Series(['INCONNU', None, ' '])), [.01]*3)


def test_negative_growth_is_kept_and_inputs_are_unchanged():
    buildings, growth = pd.Series(['A']*5), pd.Series([-.3, -.2, -.1, 0, .1])
    original_building, original_growth = buildings.copy(), growth.copy()
    model = BuildingShrunkGrowth().fit(buildings, growth)
    assert model.global_growth == -.1
    pd.testing.assert_series_equal(buildings, original_building)
    pd.testing.assert_series_equal(growth, original_growth)


def test_empty_invalid_and_misaligned_training_is_rejected():
    for values in ([], [np.nan], [np.inf], ['invalide']):
        with pytest.raises(ValueError):
            BuildingShrunkGrowth().fit(pd.Series(['A']*len(values)), pd.Series(values))
    with pytest.raises(ValueError):
        BuildingShrunkGrowth().fit(pd.Series(['A'], index=[0]), pd.Series([.1], index=[1]))
    with pytest.raises(RuntimeError):
        BuildingShrunkGrowth().predict(pd.Series(['A']))


def test_diagnostics_count_cohort_and_mask_small_historical_statistics():
    model = BuildingShrunkGrowth().fit(pd.Series(['A']*2+['B']*6), pd.Series([.1]*2+[.02]*6))
    result = model.diagnostics(pd.Series(['A']*5+['B']*10+['C']*5)).set_index('building')
    assert result.candidate_units.sum() == 20
    assert result.cohort_share_pct.sum() == 100
    assert pd.isna(result.loc['A', 'g_b_pct'])
    assert pd.isna(result.loc['A', 'g_b_shrunk_pct'])
    assert result.loc['C', 'fallback_global']
    assert result.loc['C', 'g_b_shrunk_pct'] == 100*model.global_growth
    assert not {'UNIT_KEY','sUnitCode','sPropCode'}.intersection(result.columns)


def test_d_uses_exact_a_occurrence_cohort_and_global_growth():
    leases = synthetic_leases()
    fold = _predict_d_fold(leases, 2023)
    candidate, _, positive, _, a = _champion_components(leases, 2023)
    np.testing.assert_array_equal(fold['p'], a['probabilities'])
    pd.testing.assert_frame_equal(fold['candidate'].audit, candidate.audit)
    assert fold['train_positive_count'] == len(positive)
    assert fold['A_growth_global_pct'] == 100*a['conditional_growth'][0]
    assert fold['prediction'] == 100*np.median(fold['p']*fold['g'])


def test_future_records_and_immature_effective_rents_cannot_change_d():
    leases = synthetic_leases()
    before = _predict_d_fold(leases, 2023)
    changed = leases.copy()
    immature = pd.to_datetime(changed.sLeaseTo).gt('2022-12-31')
    changed.loc[immature, 'sRentEffective'] = 999999.
    future = pd.to_datetime(changed.sLeaseFrom).gt('2022-12-31')
    changed.loc[future, 'sBuilding'] = 'FUTUR'
    changed.loc[future, 'sState'] = 'FUTUR'
    after = _predict_d_fold(changed, 2023)
    np.testing.assert_array_equal(before['p'], after['p'])
    np.testing.assert_array_equal(before['g'], after['g'])
    pd.testing.assert_frame_equal(before['diagnostics'], after['diagnostics'])
    assert before['train_positive_count'] == after['train_positive_count']


def test_three_d_predictions_precede_first_comparative_scoring():
    from src.evaluation import model_d_evaluation as module
    events = []
    original = module._predict_d_fold
    def capture_prediction(*args):
        events.append('prediction')
        return original(*args)
    def capture_scoring(*args):
        events.append('scoring')
        raise RuntimeError('Scoring atteint après fixation des forecasts.')
    with patch.object(module, '_predict_d_fold', capture_prediction), patch.object(module, 'run_tournament', capture_scoring):
        with pytest.raises(RuntimeError, match='Scoring atteint'):
            run_model_d_experiment(synthetic_leases())
    assert events == ['prediction']*3+['scoring']


def test_d_refuses_2026_and_other_years():
    for year in (2022, 2026):
        with pytest.raises(ValueError):
            _predict_d_fold(synthetic_leases(), year)
