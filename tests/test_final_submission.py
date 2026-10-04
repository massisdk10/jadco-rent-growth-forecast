"""Validation des interfaces finales et de leur adaptation au moteur gelé."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.models.final_submission import estimate_2026, backtest, production_summary, _forecast, FROZEN_FINGERPRINT
from src.models.contextual_experts import rules_fingerprint

@pytest.fixture(scope='module')
def leases():
    return pd.read_csv(Path(__file__).resolve().parents[1]/'data/raw/equinoxe_lease_history.csv',
                       dtype={'sPropCode':'string','sUnitCode':'string'})

@pytest.fixture(scope='module')
def summary(leases):
    return production_summary(leases, None)


def test_frozen_engine():
    assert rules_fingerprint() == FROZEN_FINGERPRINT


def test_2026_cohort_and_assembly(leases, summary):
    f = _forecast(leases, 2026)
    assert len(f['X']) == 931
    assert summary['province_counts'] == {'Quebec':810, 'Ontario':121}
    for letter in ('p','g'):
        w=f['weights_'+letter]
        assert np.isfinite(w).all() and (w>=0).all() and (w<=1).all()
        np.testing.assert_allclose(w.sum(axis=1),1)
    np.testing.assert_allclose(f['contribution'],f['p']*f['g'])
    assert summary['forecast_P1_pct'] == 100*np.median(f['contribution'])
    assert estimate_2026(leases, None) == summary['forecast_P1_pct']


def test_scenarios_and_safe_json(summary):
    s=summary['scenarios']
    assert s['LOW'] <= s['BASE'] <= s['HIGH']
    assert s['BASE'] == summary['forecast_P1_pct']
    assert s['model_error_radius_pp'] == max(abs(r['error_pp']) for r in summary['backtests'])
    encoded=json.dumps(summary,allow_nan=False)
    for forbidden in ('sUnitCode','sPropCode','hUnit','UNIT_KEY','contribution":','weights_p'):
        assert forbidden not in encoded

@pytest.mark.parametrize('year',[2023,2024,2025])
def test_backtest_reproduces_frozen_result(leases,summary,year):
    row=backtest(leases,year)
    reference=next(r for r in summary['backtests'] if r['year']==year)
    assert row['predicted_P1_pct'] == reference['predicted_P1_pct']
    assert np.isfinite(row['realized_P1_pct'])
    expected={2023:2.872357,2024:3.271845,2025:2.972696}
    assert row['predicted_P1_pct'] == pytest.approx(expected[year],abs=1e-6)


def test_future_leases_do_not_change_forecast(leases):
    future=leases.iloc[[0]].copy()
    future['sLeaseFrom']='2026-06-01'
    future['sSignDate']='2026-05-01'
    future['sLeaseTo']='2027-06-01'
    future['sRentEffective']=999999.
    future['sRent']=999999.
    combined=pd.concat([leases,future],ignore_index=True)
    assert estimate_2026(combined,None) == estimate_2026(leases,None)


def test_optional_sources_cannot_change_frozen_prediction(leases):
    assert estimate_2026(leases,pd.DataFrame({'future_value':[999]}),pd.DataFrame({'value':[999]})) == estimate_2026(leases,None)


def test_invalid_backtest_year(leases):
    with pytest.raises(ValueError):
        backtest(leases,2026)
