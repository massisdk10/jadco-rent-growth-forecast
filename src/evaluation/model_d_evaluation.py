"""Comparer l'expérience D fixe à A/B/50-50 ; aucune prévision 2026."""
import numpy as np
import pandas as pd

from src.evaluation.model_ab_tournament import YEARS, B_LABEL, error_summary, run_tournament
from src.features.model_a_features import build_model_a_features
from src.features.model_dataset import UNIT_KEY, known_leases, prediction_origin
from src.models.model_a import _champion_components, assemble_p1
from src.models.model_d import ALPHA, BuildingShrunkGrowth


def _predict_d_fold(leases, year):
    """Prédiction privée à cutoff ; aucune cible de Y consultée."""
    if year not in YEARS:
        raise ValueError('Expérience limitée aux backtests 2023/2024/2025.')
    past = known_leases(leases, prediction_origin(year))
    candidate, history, positive, pending, a = _champion_components(past, year)
    if not positive.forecast_year.lt(year).all() or not positive.occurrence_target.eq(1).all():
        raise AssertionError('Historique de croissance incompatible avec A.')
    estimator = BuildingShrunkGrowth().fit(positive.building, positive.unit_growth)
    if not np.isclose(estimator.global_growth, a['conditional_growth'][0], rtol=0, atol=1e-15):
        raise AssertionError('Divergence de la croissance globale A ; arrêt.')
    p = a['probabilities'].copy()
    g = estimator.predict(candidate.X.building)
    assembly = assemble_p1(p, g)
    if not np.array_equal(assembly['probabilities'], a['probabilities']):
        raise AssertionError('Occurrence A modifiée ; arrêt.')
    diagnostics = estimator.diagnostics(candidate.X.building)
    diagnostics.insert(0, 'year', year)
    return {'year': year, 'candidate': candidate, 'p': p, 'g': g,
            'A_probability': a['probabilities'], 'prediction': assembly['p1_pct'],
            'diagnostics': diagnostics, 'train_positive_count': len(positive),
            'qualified_occurrence': len(history), 'pending_labels': pending,
            'A_growth_global_pct': 100*estimator.global_growth}


def run_model_d_experiment(leases):
    """Sorties agrégées exclusivement ; protocole unique, pas de sélection alpha."""
    # Fixer les trois forecasts D avant tout scoring ou comparaison de résultats.
    folds = [_predict_d_fold(leases, year) for year in YEARS]
    reference = run_tournament(leases)
    rows, diagnostics, checks = [], [], []
    for fold in folds:
        year = fold['year']
        realized = build_model_a_features(leases, year, include_targets=True)
        if not fold['candidate'].audit[list(UNIT_KEY)].equals(realized.audit[list(UNIT_KEY)]):
            raise AssertionError('Cohorte D différente de A ; arrêt.')
        labels = realized.labels
        o = labels.transition_occurred.to_numpy(dtype=float)
        growth = labels.conditional_growth.to_numpy(dtype=float)
        contributions = np.where(o == 0, 0., growth)
        if not np.isin(o, [0, 1]).all() or not np.isfinite(contributions).all():
            raise AssertionError('Labels réalisés incomplets ; aucune imputation.')
        actual = 100*float(np.median(contributions))
        common = reference['common'].loc[reference['common'].year.eq(year)].iloc[0]
        if not np.isclose(actual, common.realized_P1, rtol=0, atol=1e-10):
            raise AssertionError('Cible réalisée D différente de A ; arrêt.')
        for name, value in [('A', common.A_prediction), (B_LABEL, common.B_prediction),
                            ('A/B 50-50', common.ensemble_prediction), ('D', fold['prediction'])]:
            rows.append({'year': year, 'model': name, 'prediction_pct': value,
                         'realized_pct': actual, 'error_pp': value-actual,
                         'absolute_error_pp': abs(value-actual)})
        diagnostics.append(fold['diagnostics'])
        checks.append({'year': year, 'prediction_origin': str(prediction_origin(year).date()),
                       'candidate_units': len(fold['candidate'].X),
                       'observed_units': int(o.sum()), 'train_positive_unit_years': fold['train_positive_count'],
                       'alpha': ALPHA, 'identical_A_occurrence': True,
                       'identical_A_cohort_and_target': True,
                       'g_global_pct': fold['A_growth_global_pct'],
                       'temporal_status': 'SAFE_WITH_CONDITIONS'})
    results = pd.DataFrame(rows)
    return {'results': results, 'metrics': error_summary(results),
            'building_diagnostics': pd.concat(diagnostics, ignore_index=True),
            'checks': pd.DataFrame(checks)}
