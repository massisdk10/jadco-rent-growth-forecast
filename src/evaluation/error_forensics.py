"""Diagnostics agrégés : séparer informations à coupure et résultats rétrospectifs.

Aucune caractéristique nouvelle n'est livrée à un modèle. Les champs effectifs
et concessions demeurent descriptifs et hors liste des prédicteurs admissibles.
"""
import numpy as np
import pandas as pd

from src.evaluation.model_ab_tournament import (
    B_REFERENCE, YEARS, _evaluated_folds, assert_a_reference,
)
from src.models.model_a import backtest
from src.features.model_a_features import assert_foundation_reference
from src.external_data import get_available_external_data
from src.features.model_dataset import (
    MAIN_TARGET, UNIT_KEY, build_model_dataset, build_occurrence_features, build_prediction_cohort,
    known_leases, prediction_origin,
)

MIN_DISPLAY_SUPPORT = 5
CATEGORICAL = ('building', 'province', 'bedrooms', 'floor', 'unit_subtype', 'expiry_month')
NUMERIC = ('sqft', 'floor', 'prior_contractual_rent', 'prior_term_months', 'months_since_known_start')
SOURCE_RESERVATION = 'Contrat daté respecté ; fidélité des versions CRM non démontrée.'


def total_variation(current, previous):
    """Distance entre parts catégorielles ; zéro = mêmes parts, un = disjointes."""
    a = current.astype('string').fillna('MANQUANT').value_counts(normalize=True)
    b = previous.astype('string').fillna('MANQUANT').value_counts(normalize=True)
    keys = a.index.union(b.index)
    if current.empty or previous.empty:
        return np.nan
    return float(.5 * (a.reindex(keys, fill_value=0)-b.reindex(keys, fill_value=0)).abs().sum())


def _median(values, scale=1):
    values = pd.to_numeric(values, errors='coerce')
    values = values[np.isfinite(values)]
    return float(values.median()*scale) if len(values) >= MIN_DISPLAY_SUPPORT else np.nan


def _safe_categories(values):
    """Regrouper les modalités rares pour ne pas décrire de dossier individuel."""
    values = values.astype('string').fillna('MANQUANT')
    counts = values.value_counts()
    return values.where(values.map(counts).ge(MIN_DISPLAY_SUPPORT), 'MODALITÉS RARES')


def cutoff_composition(leases, year, *, fold=None):
    """Composition observée à D et changement depuis la cohorte précédente.

    Comparaison entre deux cohortes reconstruites à leurs propres origines.
    Pas d'ouverture réelle d'immeuble déduite de sa première présence dans CSV.
    """
    if year not in YEARS:
        raise ValueError('Diagnostics annuels limités à 2023–2025.')
    past = known_leases(leases, prediction_origin(year))
    x = fold['X'].copy() if fold is not None else build_occurrence_features(
        past, build_prediction_cohort(past, year), year).reset_index(drop=True)
    previous = build_occurrence_features(past, build_prediction_cohort(past, year-1), year-1)
    categories, numeric, shifts = [], [], []
    for feature in CATEGORICAL:
        for value, count in _safe_categories(x[feature]).value_counts().items():
            categories.append({'year': year, 'feature': feature, 'category': value,
                               'units': int(count), 'share_pct': 100*count/len(x)})
        shifts.append({'year': year, 'feature': feature,
                       'total_variation': total_variation(x[feature], previous[feature]),
                       'current_units': len(x), 'previous_units': len(previous)})
    for feature in NUMERIC:
        values = pd.to_numeric(x[feature], errors='coerce').replace([np.inf, -np.inf], np.nan)
        prev = pd.to_numeric(previous[feature], errors='coerce')
        row = {'year': year, 'feature': feature, 'available': int(values.notna().sum()),
               'missing': int(values.isna().sum()), 'median': _median(values),
               'previous_median': _median(prev)}
        row.update({f'q{q}': float(values.quantile(q/100)) if values.notna().sum() >= 5 else np.nan
                    for q in (25, 75)})
        numeric.append(row)
    return {'categories': pd.DataFrame(categories), 'numeric': pd.DataFrame(numeric),
            'mix_changes': pd.DataFrame(shifts)}


def mature_rent_economics(leases, year):
    """Économie des seuls baux signés, commencés ET terminés à D.

    Gap source = 100*(sRent-sRentEffective)/sRent. Aucun étalement de PromoPay,
    aucune jointure de concessions et aucune prétention de disponibilité absolue.
    """
    origin = prediction_origin(year)
    past = known_leases(leases, origin)
    end = pd.to_datetime(past.sLeaseTo, errors='coerce', format='mixed')
    mature = past.loc[end.le(origin) & end.ge(past['_start'])].copy()
    rent = pd.to_numeric(mature.sRent, errors='coerce')
    effective = pd.to_numeric(mature.sRentEffective, errors='coerce')
    valid = np.isfinite(rent) & np.isfinite(effective) & rent.gt(0) & effective.gt(0)
    mature['_gap_pct'] = (100*(rent-effective)/rent).where(valid)
    flag = pd.to_numeric(mature.sConcession, errors='coerce')
    mature['_concession'] = flag.where(flag.isin([0, 1]))
    mature['_start_year'] = mature['_start'].dt.year
    rows = []
    # Marginales séparées : pas de cellule bâtiment/année à faible effectif.
    for dimension in ('sBuilding', '_start_year'):
        for segment, group in mature.groupby(dimension, dropna=False, sort=True):
            if len(group) < MIN_DISPLAY_SUPPORT:
                continue
            rows.append({'year': year, 'dimension': dimension, 'segment': str(segment),
                         'mature_leases': len(group),
                         'valid_gap': int(group['_gap_pct'].notna().sum()),
                         'contractual_rent_median': _median(group.sRent),
                         'effective_rent_median': _median(group.sRentEffective),
                         'gap_median_pct': _median(group['_gap_pct']),
                         'concession_flag_known': int(group['_concession'].notna().sum()),
                         'concession_flag_pct': 100*float(group['_concession'].mean())
                         if group['_concession'].notna().sum() >= 5 else np.nan,
                         'availability_class': 'UNKNOWN'})
    return pd.DataFrame(rows)


def historical_segments(fold):
    """Supports, retard de labels et croissance mûre, disponibles selon contrat."""
    history, train = fold['history'], fold['train_B']
    rows = []
    for dimension in ('building', 'province', 'forecast_year'):
        for segment, group in history.groupby(dimension, dropna=False, sort=True):
            known = group.loc[group.occurrence_target.notna()]
            positive = known.loc[known.occurrence_target.eq(1)]
            rows.append({'year': fold['year'], 'dimension': dimension, 'segment': str(segment),
                         'qualified_occurrence': len(known),
                         'pending_labels': int(group.occurrence_target.isna().sum()),
                         'historical_occurrence_pct': 100*float(known.occurrence_target.mean())
                         if len(known) >= 5 else np.nan,
                         'qualified_positive_unit_years': len(positive),
                         'historical_unit_growth_median_pct': _median(positive.unit_growth, 100),
                         'historical_unit_growth_q25_pct': float(positive.unit_growth.quantile(.25)*100)
                         if len(positive) >= 5 else np.nan,
                         'historical_unit_growth_q75_pct': float(positive.unit_growth.quantile(.75)*100)
                         if len(positive) >= 5 else np.nan})
    for province, group in train.groupby('province', dropna=False, sort=True):
        rows.append({'year': fold['year'], 'dimension': 'B_growth_province', 'segment': str(province),
                     'qualified_growth_transitions': len(group),
                     'historical_transition_growth_median_pct': _median(group[MAIN_TARGET], 100)})
    # Fenêtres descriptives fixées, sans régler ni entraîner un modèle de récence.
    pairs = fold['pairs_train']
    for label, subset in (
        ('Toutes transitions mûres', pairs),
        ('Début en Y−1 et déjà mûr', pairs.loc[pairs.year.eq(fold['year']-1)]),
        ('Début dans Y−3 à Y−1 et déjà mûr', pairs.loc[pairs.year.between(fold['year']-3, fold['year']-1)]),
    ):
        rows.append({'year': fold['year'], 'dimension': 'mature_growth_window', 'segment': label,
                     'qualified_growth_transitions': len(subset),
                     'historical_transition_growth_median_pct': _median(subset[MAIN_TARGET], 100),
                     'growth_q25_pct': float(subset[MAIN_TARGET].quantile(.25)*100) if len(subset) >= 5 else np.nan,
                     'growth_q75_pct': float(subset[MAIN_TARGET].quantile(.75)*100) if len(subset) >= 5 else np.nan})
    return pd.DataFrame(rows)


def realized_segments(fold):
    """Marginales ex post : aucune attribution additive d'une médiane globale."""
    x, observed = fold['X'], fold['observed']
    o = observed.occurrence_target.to_numpy(dtype=float)
    g = observed.unit_growth.to_numpy(dtype=float)
    contributions = np.where(o == 0, 0., g)
    rows = []
    for dimension in CATEGORICAL:
        categories = _safe_categories(x[dimension])
        for segment in categories.unique():
            mask = categories.eq(segment).to_numpy(dtype=bool)
            positive = mask & (o == 1)
            n, positives = int(mask.sum()), int(positive.sum())
            row = {'year': fold['year'], 'dimension': dimension, 'segment': str(segment),
                   'candidate_units': n, 'share_pct': 100*n/len(x),
                   'observed_units': positives, 'non_transitions': n-positives,
                   'realized_occurrence_pct': 100*float(o[mask].mean()) if n >= 5 else np.nan,
                   'realized_conditional_growth_pct': _median(pd.Series(g[positive]), 100),
                   'realized_segment_P1_pct': _median(pd.Series(contributions[mask]), 100),
                   'realized_growth_q25_pct': float(np.quantile(g[positive], .25)*100) if positives >= 5 else np.nan,
                   'realized_growth_q75_pct': float(np.quantile(g[positive], .75)*100) if positives >= 5 else np.nan,
                   'availability_class': 'RETROSPECTIVE_ONLY'}
            for name in ('A', B_REFERENCE):
                p_pred, g_pred = fold['components'][name]
                row[f'{name}_occurrence_pct'] = 100*float(p_pred[mask].mean()) if n >= 5 else np.nan
                row[f'{name}_growth_pct'] = _median(pd.Series(g_pred[positive]), 100)
                row[f'{name}_segment_P1_pct'] = _median(pd.Series(p_pred[mask]*g_pred[mask]), 100)
                # Le nombre attendu d'occurrences est additif, contrairement à P1.
                row[f'{name}_occurrence_error_count'] = float(p_pred[mask].sum()-positives) if n >= 5 else np.nan
            if dimension == 'building' and n >= 5 and (~mask).sum() >= 5:
                row['P1_without_building_pct'] = 100*float(np.median(contributions[~mask]))
                row['leave_out_sensitivity_pp'] = row['P1_without_building_pct']-fold['realized_P1']
            rows.append(row)
    return pd.DataFrame(rows)


def retrospective_components(fold):
    """Scénarios explicatifs avec O réalisé ; jamais des prédictions utilisables."""
    o = fold['observed'].occurrence_target.to_numpy(dtype=float)
    rows = []
    for name in ('A', B_REFERENCE):
        p, g = fold['components'][name]
        predicted = 100*float(np.median(p*g))
        occurrence_replaced = 100*float(np.median(o*g))
        rows.append({'year': fold['year'], 'model': name, 'predicted_P1_pct': predicted,
                     'P1_with_realized_occurrence_pct': occurrence_replaced,
                     'realized_P1_pct': fold['realized_P1'],
                     'occurrence_replacement_change_pp': occurrence_replaced-predicted,
                     'remaining_growth_and_distribution_gap_pp': occurrence_replaced-fold['realized_P1'],
                     'availability_class': 'RETROSPECTIVE_ONLY'})
    return pd.DataFrame(rows)


def building_presence(leases, fold):
    """Première présence dans l'extrait et variation de poids à D, pas ouverture."""
    year = fold['year']
    past = known_leases(leases, prediction_origin(year))
    previous = build_occurrence_features(past, build_prediction_cohort(past, year-1), year-1)
    x = fold['X']
    rows = []
    for building, group in past.groupby('sBuilding', sort=True):
        current_n = int(x.building.eq(building).sum())
        previous_n = int(previous.building.eq(building).sum())
        rows.append({'year': year, 'building': building,
                     'first_start_in_known_extract': str(group['_start'].min().date()),
                     'known_lease_count': len(group), 'candidate_units': current_n,
                     'share_pct': 100*current_n/len(x), 'previous_candidates': previous_n,
                     'previous_share_pct': 100*previous_n/len(previous) if len(previous) else np.nan,
                     'new_to_candidate_cohort': bool(current_n and not previous_n)})
    return pd.DataFrame(rows)


def label_coverage(leases, folds):
    """Couverture des résultats après prédiction ; absence dans l'extrait ≠ vacance."""
    pairs, _ = build_model_dataset(leases)
    rows = []
    for fold in folds:
        year = fold['year']
        annual = pairs.loc[pairs.year.eq(year)].merge(
            fold['cohort'][list(UNIT_KEY)], on=list(UNIT_KEY), how='inner', validate='many_to_one')
        eligible = annual.loc[annual.eligible_main]
        observed_units = int(fold['observed'].occurrence_target.sum())
        rows.append({'year': year, 'candidate_units': len(fold['X']),
                     'annual_pairs_in_cohort': len(annual), 'eligible_pairs_in_cohort': len(eligible),
                     'ineligible_pairs_in_cohort': len(annual)-len(eligible),
                     'observed_units': observed_units,
                     'non_transitions_in_extract': len(fold['X'])-observed_units,
                     'extra_eligible_transitions_beyond_one_per_unit': len(eligible)-observed_units,
                     'availability_class': 'RETROSPECTIVE_ONLY'})
    return pd.DataFrame(rows)


def external_qualification(external):
    """Auditer les séries déjà présentes, sans téléchargement ni imputation."""
    rows = []
    for signal, group in external.groupby('signal', sort=True):
        eligible = {y: len(get_available_external_data(group, prediction_origin(y))) for y in (*YEARS, 2026)}
        safe = sum(eligible[y] for y in YEARS)
        verified = group.availability_verified.map(
            lambda value: isinstance(value, (bool, np.bool_)) and bool(value))
        status = 'SAFE_BUT_WEAK' if safe or eligible[2026] else (
            'NOT_RELEVANT' if verified.any() else 'NOT_HISTORICALLY_QUALIFIED')
        # Aucun signal actuellement admissible aux trois backtests : les statuts
        # ci-dessus ne dispensent jamais d'une revue d'applicabilité économique.
        rows.append({'signal': signal, 'source': ' / '.join(sorted(group.source_name.dropna().unique())),
                     'rows': len(group), 'values_present': int(group.value.notna().sum()),
                     'eligible_2023': eligible[2023], 'eligible_2024': eligible[2024],
                     'eligible_2025': eligible[2025], 'eligible_2026': eligible[2026],
                     'classification': status,
                     'limitation': 'Applicabilité et utilité prédictive à confirmer ; aucune utilisation dans A/B.'
                     if status == 'SAFE_BUT_WEAK' else
                     'Publication après les origines étudiées ; exclue, même si date vérifiée.'
                     if status == 'NOT_RELEVANT' else
                     'Version/date historique non qualifiée ; exclusion des prédicteurs.'})
    return pd.DataFrame(rows)


def signal_availability(folds):
    """Ne pas confondre reconstruction contractuelle et preuve d'archive CRM."""
    rows = []
    for fold in folds:
        year, x = fold['year'], fold['X']
        p, g = fold['components']['A']
        o = fold['observed'].occurrence_target
        items = [
            ('Calendrier de coupure et année prédite', str(prediction_origin(year).date()),
             'PREDICTABLE_AT_CUTOFF', 'Calendrier déterministe, indépendant des données futures.', 'Calendrier autorisé.'),
            ('Composition et échéances reconstruites', f'{len(x)} candidates', 'UNKNOWN',
             SOURCE_RESERVATION, 'Admissible selon contrat ; confirmer la fidélité historique avant qualification stricte.'),
            ('Historique qualifié d’occurrence et de croissance', f'p A={100*p.mean():.3f} %, g A={100*np.median(g):.3f} %',
             'UNKNOWN', 'Dates de maturité <= D ; aucun journal des versions CRM.',
             'Statistiques déjà utilisées sous réserve contractuelle ; aucun ajout de feature.'),
            ('Gap contractuel/effectif des baux mûrs', 'Voir économie locative agrégée à D.', 'UNKNOWN',
             'Baux terminés <= D ; ancien effectif/gap non qualifié comme feature par la fondation.',
             'Diagnostic seulement ; ne pas réintroduire le gap dans X.'),
            ('Taux de transition de Y', f'{100*o.mean():.3f} %', 'RETROSPECTIVE_ONLY',
             'Label révélé après les prédictions de Y.', 'Explication ex post uniquement.'),
            ('Croissance conditionnelle de Y', f'{100*fold["observed"].unit_growth.median():.3f} %',
             'RETROSPECTIVE_ONLY', 'Loyers des nouveaux baux et intervalle réalisés après D.', 'Explication ex post uniquement.'),
            ('Mix réalisé renouvellement/relocation', 'Non utilisé par A/B ; statut futur interdit.',
             'RETROSPECTIVE_ONLY', 'sRenewal du nouveau bail appartient aux résultats futurs.', 'Scénario ou diagnostic, jamais feature.'),
        ]
        if year == 2025:
            items.append(('Concentration des absences à Saint-Elzear', 'À vérifier dans la marginale bâtiment.',
                          'RETROSPECTIVE_ONLY', 'Absences observées en 2025, inconnues au 2024-12-31.',
                          'Ne pas encoder les 65 absences comme signal disponible à D.'))
        for signal, effect, availability, evidence, use in items:
            rows.append({'signal': signal, 'year': year, 'observed_effect': effect,
                         'availability_class': availability, 'evidence': evidence,
                         'potential_model_use': use})
    return pd.DataFrame(rows)


def run_forensics(leases, external=None):
    """Retourner uniquement des marginales agrégées, sans clé/ligne CRM."""
    assert_a_reference(pd.DataFrame([backtest(leases, year) for year in YEARS]))
    assert_foundation_reference(leases)
    folds = _evaluated_folds(leases)
    compositions = [cutoff_composition(leases, f['year'], fold=f) for f in folds]
    result = {name: pd.concat([part[name] for part in compositions], ignore_index=True)
              for name in ('categories', 'numeric', 'mix_changes')}
    for name, helper in [('historical_segments', historical_segments),
                         ('realized_segments', realized_segments),
                         ('retrospective_components', retrospective_components)]:
        result[name] = pd.concat([helper(f) for f in folds], ignore_index=True)
    result['rent_economics'] = pd.concat([mature_rent_economics(leases, y) for y in YEARS], ignore_index=True)
    result['building_presence'] = pd.concat([building_presence(leases, f) for f in folds], ignore_index=True)
    result['label_coverage'] = label_coverage(leases, folds)
    result['signals'] = signal_availability(folds)
    # Cohorte 2026 connue seulement à D=2025-12-31 : cohérence de périmètre,
    # pas de forecast ni de cible 2026 et aucun usage pour sélectionner un modèle.
    past = known_leases(leases, prediction_origin(2026))
    c = build_prediction_cohort(past, 2026)
    x = build_occurrence_features(past, c, 2026)
    rows = []
    for feature in ('building', 'province', 'expiry_month'):
        for value, count in _safe_categories(x[feature]).value_counts().items():
            rows.append({'year': 2026, 'feature': feature, 'category': value,
                         'units': int(count), 'share_pct': 100*count/len(x)})
    result['cohort_2026'] = pd.DataFrame(rows)
    if external is not None:
        result['external'] = external_qualification(external)
    return result
