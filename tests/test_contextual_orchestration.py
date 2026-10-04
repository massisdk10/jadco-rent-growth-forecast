"""Tests synthétiques du contexte fermé, du trust et de la chaîne point-in-time."""
from dataclasses import FrozenInstanceError
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest

from src.models.contextual_experts import (
    RULES, mix_component, rules_fingerprint, _prepare_experts,
    _orchestrate_prepared, _predict_contextual,
)
from src.evaluation.contextual_orchestration import run_contextual_orchestration
from src.features.model_dataset import UNIT_KEY, prediction_origin
from tests.test_model_ab_tournament import synthetic_leases


def context(n=4, **changes):
    result=pd.DataFrame({'b_support':np.full(n,100.),'b_fallback':False,'b_global':.02,
                        'long_scale':.05,'long_spread':.05,'c_support':100.,
                        'c_coverage':1.,'c_spread':.02})
    for key,value in changes.items(): result[key]=value
    return result


@pytest.mark.parametrize('component',['occurrence','growth'])
def test_weights_are_bounded_normalized_finite_and_separate(component):
    pred,weights=mix_component(np.full(4,.03),np.full(4,.07),.01,context(),component=component)
    assert ((weights>=0)&(weights<=1)).all()
    np.testing.assert_allclose(weights.sum(axis=1),1)
    assert np.isfinite(pred).all()
    assert (weights[:,0]>=1/3).all()


def test_determinism_and_no_input_mutation():
    a,b=np.full(4,.03),np.full(4,.07);ctx=context();original=ctx.copy(deep=True)
    first=mix_component(a,b,.01,ctx,component='growth')
    second=mix_component(a,b,.01,ctx,component='growth')
    for x,y in zip(first,second): np.testing.assert_array_equal(x,y)
    pd.testing.assert_frame_equal(ctx,original)
    np.testing.assert_array_equal(a,np.full(4,.03))
    np.testing.assert_array_equal(b,np.full(4,.07))


@pytest.mark.parametrize('field',['realized_transition','target_effective_annualized','new_sRentEffective'])
def test_realized_or_future_fields_are_rejected_by_closed_context(field):
    ctx=context();ctx[field]=1
    with pytest.raises(ValueError): mix_component(np.ones(4)*.03,np.ones(4)*.04,np.nan,ctx,component='growth')


@pytest.mark.parametrize('support,coverage',[(0,1),(29,1),(100,.79)])
def test_c_has_zero_trust_without_support_or_coverage(support,coverage):
    _,weights=mix_component(np.full(4,.03),np.full(4,.07),.01,
                           context(c_support=support,c_coverage=coverage),component='growth')
    assert (weights[:,2]==0).all()


def test_full_b_fallback_has_zero_trust_and_returns_a_when_c_absent():
    pred,w=mix_component(np.full(4,.03),np.full(4,.07),np.nan,
                         context(b_fallback=True,c_support=0),component='growth')
    np.testing.assert_array_equal(pred,np.full(4,.03))
    np.testing.assert_array_equal(w,np.tile([1.,0.,0.],(4,1)))


def test_missing_specialist_values_cannot_break_a():
    pred,w=mix_component(np.full(4,.03),np.full(4,np.nan),np.nan,
                         context(b_support=np.nan,c_support=np.nan,c_spread=np.nan),component='growth')
    np.testing.assert_array_equal(pred,np.full(4,.03))
    assert (w[:,0]==1).all()


def test_equal_experts_preserve_prediction():
    pred,_=mix_component(np.full(4,.03),np.full(4,.03),.03,context(),component='growth')
    np.testing.assert_array_equal(pred,np.full(4,.03))


def test_b_support_monotonicity_and_global_deviation_requirement():
    _,low=mix_component(np.full(4,.03),np.full(4,.07),np.nan,context(b_support=10),component='growth')
    _,high=mix_component(np.full(4,.03),np.full(4,.07),np.nan,context(b_support=100),component='growth')
    assert (high[:,1]>low[:,1]).all()
    _,global_only=mix_component(np.full(4,.03),np.full(4,.07),np.nan,context(b_global=.07),component='growth')
    assert (global_only[:,1]==0).all()


def test_c_can_activate_but_cannot_dominate_weak_recent_support():
    _,high=mix_component(np.full(4,.03),np.full(4,.07),.01,context(c_support=300),component='growth')
    _,low=mix_component(np.full(4,.03),np.full(4,.07),.01,context(c_support=10),component='growth')
    assert (high[:,2]>0).all()
    assert (low[:,2]==0).all()
    assert (high[:,2]<=high[:,0]).all()


def test_exact_cutoffs_and_frozen_constants():
    for y,date in {2023:'2022-12-31',2024:'2023-12-31',2025:'2024-12-31',2026:'2025-12-31'}.items():
        assert prediction_origin(y)==pd.Timestamp(date)
    with pytest.raises(FrozenInstanceError): RULES.support_scale=1
    assert rules_fingerprint()==rules_fingerprint()


def test_future_rows_and_post_cutoff_features_cannot_change_predictions():
    leases=synthetic_leases();original=leases.copy(deep=True)
    before=_predict_contextual(leases,2023)
    changed=leases.copy()
    future=pd.to_datetime(changed.sLeaseFrom).gt('2022-12-31')
    immature=pd.to_datetime(changed.sLeaseTo).gt('2022-12-31')
    changed.loc[future,'sBuilding']='FUTUR'
    changed.loc[future,'sState']='FUTUR'
    changed.loc[immature,'sRentEffective']=999999.
    extra=changed.loc[future].copy()
    extra['sLeaseFrom']='2027-01-01';extra['sSignDate']='2026-12-01';extra['sLeaseTo']='2028-12-31'
    after=_predict_contextual(pd.concat([changed,extra],ignore_index=True),2023)
    for key in ('p','g','weights_p','weights_g','contribution'):
        np.testing.assert_array_equal(before[key],after[key])
    pd.testing.assert_frame_equal(leases,original)


def test_cohort_and_occurrence_anchor_are_identical_to_a():
    from src.models.model_a import _champion_components
    leases=synthetic_leases();result=_predict_contextual(leases,2023)
    a,_,_,_,assembly=_champion_components(leases,2023)
    pd.testing.assert_frame_equal(a.audit,result['candidate'].audit)
    np.testing.assert_array_equal(assembly['probabilities'],result['A_p'])
    assert result['candidate'].labels is None


def test_prediction_never_reveals_target_year_or_2026_labels():
    from src.features import model_dataset as foundation
    original=foundation.reveal_occurrence;calls=[]
    def capture(cohort,transitions,year,label_cutoff=None):
        assert label_cutoff is not None
        assert year<2023
        calls.append(year)
        return original(cohort,transitions,year,label_cutoff)
    with patch.object(foundation,'reveal_occurrence',capture):
        _predict_contextual(synthetic_leases(),2023)
    assert calls


def test_official_property_unit_key_keeps_site_collisions_separate():
    leases=synthetic_leases();leases['sSite']='SITE_PARTAGÉ'
    leases.loc[leases.sUnitCode.eq('0010'),'sUnitCode']='0000'
    result=_predict_contextual(leases,2023)
    keys=result['candidate'].audit[list(UNIT_KEY)]
    assert len(keys)==15
    assert not keys.duplicated().any()
    assert keys.sUnitCode.eq('0000').sum()==2


def test_contributions_and_median_p1_are_correct():
    result=_predict_contextual(synthetic_leases(),2023)
    np.testing.assert_array_equal(result['contribution'],result['p']*result['g'])
    assert result['P1_pct']==100*np.median(result['contribution'])
    assert np.isfinite(result['P1_pct'])


def test_occurrence_and_growth_weights_are_not_forced_equal():
    e=_prepare_experts(synthetic_leases(),2023)
    e['p_context']=context(len(e['X']),b_fallback=True,c_support=0)
    e['g_context']=context(len(e['X']),b_fallback=False,c_support=0,b_global=.01)
    e['B_g']=e['A_g']+.03
    result=_orchestrate_prepared(e)
    assert (result['weights_p'][:,0]==1).all()
    assert (result['weights_g'][:,1]>0).all()


def test_scoring_requires_frozen_fingerprint():
    with pytest.raises(ValueError): run_contextual_orchestration(synthetic_leases())
    with pytest.raises(ValueError): run_contextual_orchestration(synthetic_leases(),expected_fingerprint='invalide')


def test_all_weights_are_fixed_before_reference_scoring():
    from src.evaluation import contextual_orchestration as module
    events=[];original=module._orchestrate_prepared
    def capture_mix(*args):
        events.append('weights')
        return original(*args)
    def capture_reference(*args):
        events.append('scoring')
        raise RuntimeError('Scoring seulement après freeze des poids.')
    with patch.object(module,'_orchestrate_prepared',capture_mix),patch.object(module,'run_model_d_experiment',capture_reference):
        with pytest.raises(RuntimeError,match='Scoring seulement'):
            run_contextual_orchestration(synthetic_leases(),expected_fingerprint=rules_fingerprint())
    assert events==['weights']*3+['scoring']
