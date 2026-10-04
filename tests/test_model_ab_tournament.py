"""Contrôles synthétiques du contrat A/B et des garde-fous temporels."""
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest

from src.evaluation.model_ab_tournament import (
    A_REFERENCE, _evaluated_folds, _predict_fold, _reveal_fold,
    assert_a_reference, error_summary, fixed_ensemble,
)
from src.models.model_b.experiments import run_p1_combination_tournament


def synthetic_leases():
    """Baux fictifs : cohortes à échéance et historique Y−1 déjà mûr séparés."""
    rows = []
    for unit in range(18):
        province = 'Quebec' if unit < 10 else 'Ontario'
        for year in range(2018, 2026):
            auxiliary = unit >= 15
            rows.append({'sPropCode': 'P-Q' if province == 'Quebec' else 'P-O',
                         'sUnitCode': f'{unit:04d}', 'sBuilding': 'B-Q' if province == 'Quebec' else 'B-O',
                         'sState': province, 'sBeds': 1 if unit % 2 else 2, 'sBaths': 1,
                         'sSqft': 700+20*unit, 'sFloor': 2, 'sUnitSubtype': 'TYPE',
                         'sLeaseFrom': f'{year}-01-01' if auxiliary else f'{year}-03-01',
                         'sLeaseTo': f'{year}-06-30' if auxiliary else f'{year+1}-02-28',
                         'sSignDate': f'{year-1}-12-15', 'sRent': 1000*1.03**(year-2018),
                         'sRentEffective': 900*1.02**(year-2018),
                         'sTermMonths': 6 if auxiliary else 12,
                         'sConcession': 0, 'sRenewal': 1})
    return pd.DataFrame(rows)


def test_a_reference_rejects_changed_prediction_and_cohort():
    rows = [{'year': year, 'candidate_units': n, 'observed_units': o,
             'predicted_P1_pct': p, 'realized_P1_pct': r}
            for year, (n, o, p, r) in A_REFERENCE.items()]
    data = pd.DataFrame(rows)
    assert_a_reference(data)
    for column in ('predicted_P1_pct', 'candidate_units'):
        changed = data.copy()
        changed.loc[0, column] += .01 if column.endswith('pct') else 1
        with pytest.raises(AssertionError):
            assert_a_reference(changed)


def test_scalar_ensemble_requires_comparability_and_finite_alignment():
    assert fixed_ensemble([2.], [4.], comparable=True)[0] == 3.
    with pytest.raises(ValueError):
        fixed_ensemble([2.], [4.], comparable=False)
    for a, b in (([1.], [1., 2.]), ([np.nan], [2.])):
        with pytest.raises(ValueError):
            fixed_ensemble(a, b, comparable=True)


def test_scalar_ensemble_is_not_median_of_unitwise_average():
    a, b = np.array([0., 0., 9.]), np.array([9., 0., 0.])
    scalar = fixed_ensemble(np.median(a), np.median(b), comparable=True)
    assert scalar == 0
    assert np.median(.5*a+.5*b) == 4.5


def test_error_summary_uses_signed_pp_and_requires_three_distinct_years():
    frame = pd.DataFrame({'model': ['X']*3, 'year': [2023, 2024, 2025], 'error_pp': [1., -2., 0.]})
    row = error_summary(frame).iloc[0]
    assert row.MAE_pp == 1
    assert row.bias_pp == pytest.approx(-1/3)
    assert row.RMSE_pp == pytest.approx(np.sqrt(5/3))
    assert row.worst_absolute_error_pp == 2
    with pytest.raises(ValueError):
        error_summary(frame.iloc[:2])


def test_future_records_do_not_change_candidates_training_or_predictions():
    leases = synthetic_leases()
    before = _predict_fold(leases, 2023)
    changed = leases.copy()
    future = pd.to_datetime(changed.sLeaseFrom).gt('2022-12-31')
    changed.loc[future, ['sRent', 'sRentEffective']] = 999999.
    changed.loc[future, 'sBuilding'] = 'FUTUR'
    changed.loc[future, 'sState'] = 'FUTUR'
    after = _predict_fold(changed, 2023)
    pd.testing.assert_frame_equal(before['X'], after['X'])
    pd.testing.assert_frame_equal(before['train_B'], after['train_B'])
    for name in before['components']:
        for left, right in zip(before['components'][name], after['components'][name]):
            np.testing.assert_array_equal(left, right)


def test_adapter_reproduces_all_four_frozen_b_predictions():
    leases = synthetic_leases()
    original, _ = run_p1_combination_tournament(leases)
    for year in (2023, 2024, 2025):
        fold = _predict_fold(leases, year)
        for name in ('P0_G0', 'P0_G1', 'P1_G0', 'P1_G1'):
            p, g = fold['components'][name]
            expected = original.loc[original.year.eq(year) & original.model_name.eq(name), 'predicted_P1_percent'].iloc[0]
            assert 100*np.median(p*g) == pytest.approx(expected, abs=1e-12)


def test_all_predictions_are_fixed_before_first_test_reveal():
    from src.evaluation import model_ab_tournament as module
    events = []
    predict, reveal = module._predict_fold, module._reveal_fold
    def capture_predict(*args):
        events.append('prediction')
        return predict(*args)
    def capture_reveal(*args):
        events.append('reveal')
        return reveal(*args)
    with patch.object(module, '_predict_fold', capture_predict), patch.object(module, '_reveal_fold', capture_reveal):
        _evaluated_folds(synthetic_leases())
    assert events == ['prediction']*3+['reveal']*3


def test_different_full_and_year_end_labels_block_comparison():
    leases = synthetic_leases()
    fold = _predict_fold(leases, 2023)
    from src.evaluation import model_ab_tournament as module
    original = module.reveal_occurrence
    calls = []
    def changed_reveal(*args):
        result = original(*args)
        calls.append(1)
        if len(calls) == 2:
            result.loc[result.index[0], 'occurrence_target'] = 0
        return result
    with patch.object(module, 'reveal_occurrence', changed_reveal), pytest.raises(AssertionError):
        _reveal_fold(leases, fold)


def test_no_forecast_2026_is_computed_by_adapter():
    with pytest.raises(ValueError):
        _predict_fold(synthetic_leases(), 2026)
