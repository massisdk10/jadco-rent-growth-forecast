"""Évaluation commune des modèles figés ; tables publiques agrégées uniquement.

Les détails unitaires sont privés, conservés en mémoire pour les diagnostics.
P1_G1 désigne l'architecture B entièrement hiérarchique, pas un champion choisi.
La sécurité temporelle reste conditionnelle à la fidélité de l'extrait CRM.
"""
import numpy as np
import pandas as pd

from src.evaluation.backtest import (
    aggregate_expected_contribution, baseline_previous_year, occurrence_history,
    predict_occurrence, split_year,
)
from src.features.model_dataset import (
    MAIN_TARGET, OCCURRENCE_FEATURES, UNIT_KEY, build_features,
    build_model_dataset, build_occurrence_features, build_prediction_cohort,
    known_leases, prediction_origin, reveal_occurrence,
)
from src.models.model_a import _champion_components, backtest
from src.features.model_a_features import assert_foundation_reference
from src.models.model_b.experiments import (
    P1_COMBINATIONS, P1_GROWTH_CONFIG, P1_OCCURRENCE_CONFIG,
    _p1_growth_training, run_p1_combination_tournament,
)
from src.models.model_b.growth import HierarchicalGrowthEstimator
from src.models.model_b.occurrence import HierarchicalOccurrenceEstimator

YEARS = (2023, 2024, 2025)
B_REFERENCE = 'P1_G1'
B_LABEL = 'Model B — full hierarchical'
A_REFERENCE = {
    2023: (405, 401, 2.8700597508977657, 2.0361023490776855),
    2024: (645, 634, 3.2691787873371383, 1.607852406246998),
    2025: (856, 773, 2.9693278810997157, 3.3008784572648096),
}


def assert_a_reference(results):
    """Arrêter avant la comparaison si le champion A diverge de la référence."""
    if results.year.tolist() != list(YEARS):
        raise AssertionError('Les trois années de référence sont obligatoires.')
    for row in results.itertuples():
        n, observed, predicted, realized = A_REFERENCE[row.year]
        if (row.candidate_units, row.observed_units) != (n, observed):
            raise AssertionError('Divergence de cohorte A ; comparaison arrêtée.')
        if not np.allclose([row.predicted_P1_pct, row.realized_P1_pct],
                           [predicted, realized], rtol=0, atol=1e-8):
            raise AssertionError('Divergence de forecast A ; comparaison arrêtée.')


def error_summary(results):
    """Résumer les erreurs signées en pp, sans classement ni choix automatique."""
    rows = []
    for model, group in results.groupby('model', sort=False):
        errors = group.error_pp.to_numpy(dtype=float)
        if len(errors) != 3 or set(group.year) != set(YEARS) or not np.isfinite(errors).all():
            raise ValueError('Trois erreurs finies et trois années distinctes sont nécessaires.')
        rows.append({'model': model, 'MAE_pp': float(np.mean(np.abs(errors))),
                     'RMSE_pp': float(np.sqrt(np.mean(errors**2))),
                     'bias_pp': float(np.mean(errors)),
                     'worst_absolute_error_pp': float(np.max(np.abs(errors)))})
    return pd.DataFrame(rows)


def fixed_ensemble(a, b, *, comparable):
    """Moyenne des deux P1 scalaires ; aucun mélange de médianes unitaires."""
    if not comparable:
        raise ValueError('Ensemble interdit sans comparabilité vérifiée.')
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Les forecasts doivent être finis et alignés.')
    return .5 * a + .5 * b


def _predict_fold(leases, year):
    """Rejouer les composantes fixes existantes, sans consulter les résultats Y.

    Retour privé : identifiants et vecteurs restent exclusivement en mémoire.
    Les helpers/configurations de B sont réutilisés, sans recherche de paramètres.
    """
    if year not in YEARS:
        raise ValueError('Cette évaluation ne calcule aucun forecast 2026.')
    origin = prediction_origin(year)
    past = known_leases(leases, origin)
    candidate, qualified, positive, pending, a = _champion_components(past, year)
    cohort = build_prediction_cohort(past, year)
    if not candidate.audit[list(UNIT_KEY)].reset_index(drop=True).equals(
            cohort[list(UNIT_KEY)].reset_index(drop=True)):
        raise AssertionError('Les clés candidates A/B ne sont pas identiques.')
    history = occurrence_history(past, year)
    train_occ = history.loc[history.occurrence_target.notna()].copy()
    pairs, _ = build_model_dataset(past)
    train, _ = split_year(pairs, year, MAIN_TARGET)
    if train.empty or train_occ.empty:
        raise ValueError('Historique qualifié insuffisant ; aucun repli inventé.')
    if not (pd.to_datetime(train.label_available_date).le(origin).all()
            and train.year.lt(year).all() and train_occ.forecast_year.lt(year).all()):
        raise AssertionError('Label futur dans le train.')
    x = build_occurrence_features(past, cohort, year).reset_index(drop=True)
    p1 = HierarchicalOccurrenceEstimator(P1_OCCURRENCE_CONFIG).fit(
        train_occ[list(OCCURRENCE_FEATURES)], train_occ.occurrence_target.astype(float)
    ).predict_proba(x)
    p0 = predict_occurrence(history, x, method='historical_global')
    growth_train = _p1_growth_training(past, train, year).reset_index(drop=True)
    g1 = HierarchicalGrowthEstimator(P1_GROWTH_CONFIG).fit(
        growth_train[['province']], growth_train[MAIN_TARGET].rename(MAIN_TARGET)
    ).predict(build_features(past, cohort, year, features=('province',)))
    g0_value = baseline_previous_year(train, year, MAIN_TARGET)
    if not np.isfinite(g0_value):
        raise ValueError('Médiane Y−1 indisponible ; aucun remplissage.')
    g0 = np.full(len(cohort), g0_value)
    return {'year': year, 'cohort': cohort, 'X': x, 'history': history,
            'positive_A': positive, 'train_B': growth_train, 'pairs_train': train,
            'pending': pending,
            'components': {'A': (a['probabilities'], a['conditional_growth']),
                           'P0_G0': (p0, g0), 'P0_G1': (p0, g1),
                           'P1_G0': (p1, g0), 'P1_G1': (p1, g1)}}


def _reveal_fold(leases, fold):
    """Vérifier l'identité des labels A complets et B connus à fin d'année."""
    year, cohort = fold['year'], fold['cohort']
    full, _ = build_model_dataset(leases)
    year_end, _ = build_model_dataset(known_leases(leases, pd.Timestamp(year, 12, 31)))
    a = reveal_occurrence(cohort, full, year)
    b = reveal_occurrence(cohort, year_end, year)
    for name in ('occurrence_target', 'unit_growth'):
        if not np.allclose(a[name].to_numpy(dtype=float), b[name].to_numpy(dtype=float),
                           equal_nan=True, rtol=0, atol=1e-12):
            raise AssertionError('Labels A/B différents ; comparaison et ensemble arrêtés.')
    o = a.occurrence_target.to_numpy(dtype=float)
    g = a.unit_growth.to_numpy(dtype=float)
    c = np.where(o == 0, 0., g)
    if not np.isin(o, [0., 1.]).all() or not np.isfinite(c).all():
        raise AssertionError('Labels de test incomplets ; aucune imputation.')
    fold['observed'] = a.reset_index(drop=True)
    fold['realized_P1'] = 100 * float(np.median(c))
    return fold


def _evaluated_folds(leases):
    """Fixer toutes les prédictions des trois années avant toute révélation test."""
    folds = [_predict_fold(leases, year) for year in YEARS]
    return [_reveal_fold(leases, fold) for fold in folds]


def component_diagnostics(folds, b_reference=B_REFERENCE):
    """Comparer p et g séparément sur une population de test commune."""
    rows = []
    for fold in folds:
        o = fold['observed'].occurrence_target.to_numpy(dtype=float)
        g_real = fold['observed'].unit_growth.to_numpy(dtype=float)
        positive = o == 1
        for name in ('A', b_reference):
            p, g = fold['components'][name]
            errors = 100 * (g[positive] - g_real[positive])
            rows.append({'year': fold['year'], 'model': name,
                         'historical_occurrence_pct': 100 * fold['history'].occurrence_target.mean(),
                         'predicted_occurrence_pct': 100 * p.mean(),
                         'realized_occurrence_pct': 100 * o.mean(),
                         'occurrence_error_pp': 100 * (p.mean() - o.mean()),
                         'Brier': float(np.mean((p-o)**2)),
                         'predicted_growth_pct': 100 * float(np.median(g[positive])),
                         'realized_growth_pct': 100 * float(np.median(g_real[positive])),
                         'growth_median_error_pp': 100 * float(np.median(g[positive])-np.median(g_real[positive])),
                         'growth_MAE_pp': float(np.mean(np.abs(errors))),
                         'growth_bias_pp': float(errors.mean())})
    return pd.DataFrame(rows)


def run_tournament(leases, b_reference=B_REFERENCE):
    """Retourner uniquement des tableaux agrégés ; arrêter en cas de divergence.

    B_REFERENCE est déclaré avant scoring. Les quatre variantes sont publiées,
    aucune n'est promue champion par ce module. La sécurité est celle du contrat
    temporel figé, avec réserve explicite sur les versions historiques CRM.
    """
    if b_reference not in P1_COMBINATIONS:
        raise ValueError('Combinaison B hors de la shortlist figée.')
    a = pd.DataFrame([backtest(leases, y) for y in YEARS])
    assert_a_reference(a)
    foundation = assert_foundation_reference(leases)
    folds = _evaluated_folds(leases)
    original_b, original_summary = run_p1_combination_tournament(leases)
    rows, checks = [], []
    for fold in folds:
        year, real = fold['year'], fold['realized_P1']
        ref_a = a.loc[a.year.eq(year)].iloc[0]
        if not np.isclose(real, ref_a.realized_P1_pct, rtol=0, atol=1e-10):
            raise AssertionError('Référence réalisée A différente ; arrêt.')
        for name, (p, g) in fold['components'].items():
            pred = 100 * aggregate_expected_contribution(p, g)
            expected = ref_a.predicted_P1_pct if name == 'A' else original_b.loc[
                original_b.year.eq(year) & original_b.model_name.eq(name), 'predicted_P1_percent'].iloc[0]
            if not np.isclose(pred, expected, rtol=0, atol=1e-10):
                raise AssertionError('Adaptateur différent du modèle figé ; arrêt.')
            rows.append({'year': year, 'model': name, 'realized_P1': real,
                         'prediction': pred, 'error_pp': pred-real})
        checks.append({'year': year, 'prediction_origin': str(prediction_origin(year).date()),
                       'candidate_units': len(fold['X']),
                       'observed_units': int(fold['observed'].occurrence_target.sum()),
                       'identical_keys': True, 'identical_labels': True,
                       'qualified_occurrence': int(fold['history'].occurrence_target.notna().sum()),
                       'pending_labels': fold['pending'], 'growth_A_unit_years': len(fold['positive_A']),
                       'growth_B_transitions': len(fold['train_B']),
                       'temporal_status': 'SAFE_WITH_CONDITIONS'})
    predictions = pd.DataFrame(rows)
    wide = predictions.pivot(index='year', columns='model', values='prediction')
    realized = predictions.groupby('year').realized_P1.first()
    common = pd.DataFrame({'year': wide.index, 'realized_P1': realized.to_numpy(),
                           'A_prediction': wide.A.to_numpy(),
                           'B_prediction': wide[b_reference].to_numpy()})
    common['A_error'] = common.A_prediction-common.realized_P1
    common['B_error'] = common.B_prediction-common.realized_P1
    common['ensemble_prediction'] = fixed_ensemble(common.A_prediction, common.B_prediction, comparable=True)
    common['ensemble_error'] = common.ensemble_prediction-common.realized_P1
    for row in common.itertuples():
        rows.append({'year': row.year, 'model': 'A/B 50-50', 'realized_P1': row.realized_P1,
                     'prediction': row.ensemble_prediction, 'error_pp': row.ensemble_error})
    scores = pd.DataFrame(rows)
    main = scores.loc[scores.model.isin(['A', b_reference, 'A/B 50-50'])]
    b_label = B_LABEL if b_reference == B_REFERENCE else f'Model B — {b_reference}'
    metrics = error_summary(main)
    metrics['model'] = metrics.model.replace({b_reference: b_label})
    components = component_diagnostics(folds, b_reference)
    components['model'] = components.model.replace({b_reference: b_label})
    return {'common': common, 'metrics': metrics, 'foundation': foundation,
            'all_B_results': original_b, 'all_B_metrics': original_summary,
            'comparability': pd.DataFrame(checks),
            'components': components}
