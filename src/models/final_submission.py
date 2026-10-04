"""Interfaces de soumission : adaptateur agrégé du système définitivement gelé."""
import numpy as np
import pandas as pd
from src.models.contextual_experts import rules_fingerprint, _predict_contextual
from src.evaluation.contextual_orchestration import _diagnostics
from src.features.model_a_features import build_model_a_features
from src.features.model_dataset import UNIT_KEY
from src.evaluation.model_a_uncertainty import uncertainty_scenarios

FROZEN_FINGERPRINT = 'f8ce4bd1355a1ab3ee238bdbbb8754e38a70d332b72f5548c1187952dc51fa07'


def _forecast(leases, year):
    """Calcul privé ; vecteurs confidentiels conservés uniquement en mémoire."""
    if rules_fingerprint() != FROZEN_FINGERPRINT:
        raise ValueError('Le moteur diffère du protocole gelé avant scoring.')
    f = _predict_contextual(leases, year)
    for letter in ('p', 'g'):
        w = f['weights_'+letter]
        if not (np.isfinite(w).all() and (w >= 0).all() and (w <= 1).all()
                and np.allclose(w.sum(axis=1), 1)):
            raise AssertionError('Poids invalides.')
    if not np.isfinite(f['contribution']).all() or not np.allclose(f['contribution'], f['p']*f['g']):
        raise AssertionError('Contributions invalides.')
    if not np.isclose(f['P1_pct'], 100*np.median(f['p']*f['g'])):
        raise AssertionError('Agrégation P1 invalide.')
    if year == 2026 and len(f['X']) != 931:
        raise AssertionError('La cohorte de soumission attendue comporte 931 unités.')
    if rules_fingerprint() != FROZEN_FINGERPRINT:
        raise AssertionError('Protocole modifié pendant le calcul.')
    return f


def estimate_2026(leases, asking, external=None):
    """P1 2026 en pourcentage ; asking/external conservés pour compatibilité.

    Ces sources restent exclues des prédicteurs gelés ; aucune requalification.
    """
    return _forecast(leases, 2026)['P1_pct']


def backtest(leases, target_year):
    """Prédire à D avant révélation des labels ; résultats uniquement agrégés."""
    if target_year not in (2023, 2024, 2025):
        raise ValueError('Backtests autorisés : 2023, 2024 et 2025.')
    f = _forecast(leases, target_year)
    observed = build_model_a_features(leases, target_year, include_targets=True)
    if not f['candidate'].audit[list(UNIT_KEY)].equals(observed.audit[list(UNIT_KEY)]):
        raise AssertionError('Cohortes de prédiction et scoring différentes.')
    row, _ = _diagnostics(f, observed.labels)
    return row


def _summary(f, mask=None):
    """Résumé d'au moins cinq candidates ; aucune contribution individuelle."""
    mask = np.ones(len(f['X']), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    n = int(mask.sum())
    if n < 5:
        return {'candidate_units': n, 'statistics_masked': True}
    result = {'candidate_units': n, 'occurrence_pct': float(100*f['p'][mask].mean()),
              'conditional_growth_median_pct': float(100*np.median(f['g'][mask])),
              'median_contribution_pct': float(100*np.median(f['contribution'][mask]))}
    for letter, name in (('p', 'occurrence'), ('g', 'growth')):
        context = f[letter+'_context']
        item = {'B_global_fallback_count': int(context.b_fallback.to_numpy()[mask].sum()),
                'B_support_median': float(np.median(context.b_support.to_numpy()[mask])),
                'B_support_min': float(context.b_support.to_numpy()[mask].min()),
                'B_support_max': float(context.b_support.to_numpy()[mask].max())}
        for j, expert in enumerate(('A','B','C')):
            value = f[expert+'_'+letter]
            item[expert+'_estimate_pct'] = (float(100*(np.mean(value[mask]) if letter=='p' else np.median(value[mask])))
                                           if expert != 'C' else float(100*value) if np.isfinite(value) else None)
            item[expert+'_mean_weight'] = float(f['weights_'+letter][mask,j].mean())
            item[expert+'_min_weight'] = float(f['weights_'+letter][mask,j].min())
            item[expert+'_max_weight'] = float(f['weights_'+letter][mask,j].max())
            item[expert+'_activation_count'] = int((f['weights_'+letter][mask,j]>0).sum())
        if letter == 'p':
            child = f['child_supported'][mask]
            fallback = context.b_fallback.to_numpy()[mask]
            item['B_child_support_count'] = int(child.sum())
            item['B_parent_fallback_count'] = int((~child & ~fallback).sum())
        result[name] = item
    return result


def production_summary(leases, asking, external=None):
    """Prévision, diagnostics et scénarios exportables sans lignes CRM."""
    f = _forecast(leases, 2026)
    historical = pd.DataFrame([backtest(leases, y) for y in (2023,2024,2025)])
    scenarios = uncertainty_scenarios(f['p'], f['g'], historical, n_bootstrap=1000, random_state=42)
    x = f['X']
    return {'forecast_year':2026, 'cutoff':'2025-12-31', 'rules_fingerprint':FROZEN_FINGERPRINT,
            'target_definition':'P1 = 100 × médiane des contributions unitaires p × g dans la cohorte fixe',
            'forecast_P1_pct':f['P1_pct'], 'portfolio':_summary(f),
            'province_counts':{str(k):int(v) for k,v in x.province.value_counts(dropna=False).items()},
            'building_counts':{str(k):int(v) for k,v in x.building.value_counts(dropna=False).items()},
            'expiry_month_counts':{str(k):int(v) for k,v in x.expiry_month.value_counts(dropna=False).sort_index().items()},
            'province_diagnostics':{str(k):_summary(f,x.province.eq(k)) for k in x.province.dropna().unique()},
            'building_diagnostics':{str(k):_summary(f,x.building.eq(k)) for k in x.building.dropna().unique()},
            'C_qualification':{'occurrence_qualified':bool(f['C_p_valid']), 'growth_qualified':bool(f['C_g_valid']),
                               'known_recent_labels':len(f['last_known']), 'positive_recent_labels':len(f['last_positive']),
                               'recent_candidate_units':len(f['last']), 'label_coverage':f['coverage'],
                               'rule':'Support ≥30 par composante et couverture ≥80 %, avec cohérence et divergence gelées.'},
            'scenarios':scenarios, 'backtests':historical.astype(object).where(pd.notna(historical), None).to_dict(orient='records'),
            'limitations':['Seulement trois années de backtest ; erreurs A/B corrélées.',
              'Aucun signal fiable à cutoff pour le ralentissement 2024 ou la rupture Saint-Elzear 2025.',
              'C peut rester inactif ; les scénarios ne sont pas des intervalles de confiance calibrés.',
              'Extrait CRM rétrospectif sans journal de versions ; fidélité historique conditionnelle.',
              'P1 ne mesure ni le revenu total ni le taux d’occupation du portefeuille réel.']}
