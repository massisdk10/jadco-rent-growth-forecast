"""Tests des marginales, de la confidentialité et des classifications temporelles."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock

from src.evaluation.error_forensics import (
    _median, cutoff_composition, external_qualification, mature_rent_economics,
    realized_segments, signal_availability, total_variation,
)
from src.evaluation.model_ab_tournament import _evaluated_folds
from tests.test_model_ab_tournament import synthetic_leases


def test_total_variation_checks_disjoint_and_identical_composition():
    assert total_variation(pd.Series(['A', 'B']), pd.Series(['B', 'A'])) == 0
    assert total_variation(pd.Series(['A']), pd.Series(['B'])) == 1
    assert np.isnan(total_variation(pd.Series(dtype=str), pd.Series(['B'])))


def test_small_groups_do_not_publish_numeric_record_values():
    assert np.isnan(_median(pd.Series([1234.])))
    assert _median(pd.Series([1., 2., 3., 4., 5.])) == 3


def test_cutoff_composition_ignores_future_price_and_characteristics():
    leases = synthetic_leases()
    before = cutoff_composition(leases, 2023)
    changed = leases.copy()
    future = pd.to_datetime(changed.sLeaseFrom).gt('2022-12-31')
    changed.loc[future, 'sRent'] = 999999.
    changed.loc[future, 'sSqft'] = 999999.
    changed.loc[future, 'sBuilding'] = 'FUTUR'
    after = cutoff_composition(changed, 2023)
    for key in before:
        pd.testing.assert_frame_equal(before[key], after[key])


def test_composition_counts_exhaust_cohort_and_have_no_identifiers():
    result = cutoff_composition(synthetic_leases(), 2023)
    grouped = result['categories'].groupby('feature').agg(units=('units','sum'), share=('share_pct','sum'))
    assert grouped.units.eq(15).all()
    assert grouped.share.eq(100).all()
    for table in result.values():
        assert not {'sUnitCode', 'UNIT_KEY', 'sPropCode', 'hTenant'}.intersection(table.columns)


def test_mature_rent_economics_never_uses_running_or_future_leases():
    leases = synthetic_leases()
    before = mature_rent_economics(leases, 2023)
    changed = leases.copy()
    immature = pd.to_datetime(changed.sLeaseTo).gt('2022-12-31')
    changed.loc[immature, 'sRentEffective'] = 1.
    changed.loc[immature, 'sConcession'] = 1
    after = mature_rent_economics(changed, 2023)
    pd.testing.assert_frame_equal(before, after)
    assert set(before.availability_class) == {'UNKNOWN'}


def test_retrospective_segments_have_non_additive_median_sensitivity():
    fold = _evaluated_folds(synthetic_leases())[0]
    result = realized_segments(fold)
    buildings = result.loc[result.dimension.eq('building')]
    assert buildings.candidate_units.sum() == len(fold['X'])
    assert buildings.observed_units.sum() == fold['observed'].occurrence_target.sum()
    assert set(result.availability_class) == {'RETROSPECTIVE_ONLY'}
    assert 'leave_out_sensitivity_pp' in result
    assert 'contribution_to_global_P1' not in result
    assert not {'sUnitCode', 'UNIT_KEY', 'sRentEffective'}.intersection(result.columns)


def test_availability_does_not_promote_realized_growth_or_gap_to_features():
    signals = signal_availability(_evaluated_folds(synthetic_leases()))
    assert signals.loc[signals.signal.eq('Taux de transition de Y'), 'availability_class'].eq('RETROSPECTIVE_ONLY').all()
    assert signals.loc[signals.signal.eq('Gap contractuel/effectif des baux mûrs'), 'availability_class'].eq('UNKNOWN').all()
    assert set(signals.availability_class) == {'PREDICTABLE_AT_CUTOFF', 'RETROSPECTIVE_ONLY', 'UNKNOWN'}


def test_external_qualification_does_not_backdate_reference_year():
    data = pd.DataFrame({'signal': ['test', 'test', 'nonqualifie'],
                         'source_name': ['Source']*3, 'value': [1., 2., 3.],
                         'available_date': ['2025-01-21', '2026-01-19', '2020-01-01'],
                         'availability_verified': [True, True, False]})
    result = external_qualification(data).set_index('signal')
    assert result.loc['test', ['eligible_2023','eligible_2024','eligible_2025']].eq(0).all()
    assert result.loc['test', 'eligible_2026'] == 1
    assert result.loc['test', 'classification'] == 'SAFE_BUT_WEAK'
    assert result.loc['nonqualifie', 'classification'] == 'NOT_HISTORICALLY_QUALIFIED'


def test_forensics_stops_before_analysis_when_a_reference_diverges(monkeypatch):
    from src.evaluation import error_forensics as module
    analyze = Mock()
    monkeypatch.setattr(module, '_evaluated_folds', analyze)
    with pytest.raises(AssertionError):
        module.run_forensics(synthetic_leases())
    analyze.assert_not_called()
