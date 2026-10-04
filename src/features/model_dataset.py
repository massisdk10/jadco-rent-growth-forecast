"""Contrat des transitions et prédicteurs à origine annuelle, en mémoire."""
import numpy as np
import pandas as pd
from src.same_unit import UNIT_KEY, build_same_unit_pairs
from src.rent_economics import calculate_gaps
from src.external_data import get_available_external_data

TARGETS = {
    'target_contractual_raw': 'contractual_growth',
    'target_effective_raw': 'effective_growth',
    'target_contractual_annualized': 'contractual_annualized_growth',
    'target_effective_annualized': 'effective_annualized_growth',
}
MAIN_TARGET = 'target_effective_annualized'
# Disponibilité conditionnelle à un bail déjà signé et commencé à l'origine.
SOURCE_FEATURES = {
    'property_code': 'sPropCode', 'building': 'sBuilding', 'province': 'sState',
    'bedrooms': 'sBeds', 'bathrooms': 'sBaths', 'sqft': 'sSqft', 'floor': 'sFloor',
    'unit_subtype': 'sUnitSubtype', 'prior_contractual_rent': 'sRent',
    'prior_term_months': 'sTermMonths',
}
ADMISSIBLE_FEATURES = tuple(SOURCE_FEATURES) + ('forecast_year', 'months_since_known_start')
OCCURRENCE_FEATURES = ADMISSIBLE_FEATURES + ('expiry_month',)
FORBIDDEN_FEATURES = (
    'new_sRent', 'new_sRentEffective', 'new_sConcession', 'new_sTermMonths',
    'new_sSignDate', 'new_sLeaseFrom', 'new_sLeaseTo', 'new_sRenewal',
    'sRenewal', 'days_between', 'months_between', 'old_sRentEffective',
    'old_concession_gap', 'previous_effective_growth', 'future_segment_weight',
) + tuple(TARGETS)


def validate_features(features):
    """Liste fermée : tout champ inconnu, cible ou issu du nouveau bail échoue."""
    rejected = set(features) - set(ADMISSIBLE_FEATURES)
    if rejected:
        raise ValueError(f'Features interdites ou non qualifiées : {sorted(rejected)}')
    if len(features) != len(set(features)):
        raise ValueError('Features dupliquées.')


def prediction_origin(year):
    """Fin de Y−1 inclusive, comme dans le starter (résolution journalière)."""
    return pd.Timestamp(year=int(year)-1, month=12, day=31)


def build_model_dataset(leases):
    """Réutiliser le moteur officiel ; conserver cibles et réserves en mémoire.

    label_available_date est une borne prudente : maximum début/signature/fin
    des deux baux. La fin évite de traiter l'effectif comme un encaissement
    connu à la signature. Cette borne ne prouve pas l'absence de révisions CRM.
    """
    pairs, trace = build_same_unit_pairs(leases)
    out = pairs.copy(deep=True)
    for target, source in TARGETS.items():
        out[target] = out[source]
    dates = pd.DataFrame({
        side+'_'+col: pd.to_datetime(
            out.get(side+'_'+col, pd.Series(pd.NaT, index=out.index)),
            errors='coerce', format='mixed')
        for side in ('old', 'new') for col in ('sLeaseFrom', 'sSignDate', 'sLeaseTo')
    })
    valid_dates = dates.notna().all(axis=1)
    for side in ('old', 'new'):
        valid_dates &= dates[side+'_sLeaseTo'].ge(dates[side+'_sLeaseFrom'])
    out['label_available_date'] = dates.max(axis=1).where(valid_dates)
    out['eligible_main'] = np.isfinite(out[MAIN_TARGET]) & out.valid_temporal
    return out, trace


def known_leases(leases, origin):
    """Snapshot reconstruit des baux commencés ET signés avant la coupure.

    Pas de listings actuels, ni de bail futur déjà présent dans l'extrait.
    Les dates invalides échouent par exclusion ; les prix invalides ne sont
    pas sautés pour chercher un autre bail. Une égalité de débuts est ambiguë.
    """
    d = leases.copy(deep=True)
    d['_start'] = pd.to_datetime(d.sLeaseFrom, errors='coerce', format='mixed')
    d['_sign'] = pd.to_datetime(d.sSignDate, errors='coerce', format='mixed')
    mask = d['_start'].le(origin) & d['_sign'].le(origin)
    mask &= d[list(UNIT_KEY)].notna().all(axis=1)
    for col in UNIT_KEY:
        mask &= d[col].astype('string').str.strip().ne('').fillna(False)
    return d.loc[mask].copy()


def build_features(leases, observations, year, features=ADMISSIBLE_FEATURES):
    """Prédicteurs figés à l'origine : jamais le précédent réel futur du test.

    observations ne fournit que la clé officielle, pas les caractéristiques
    nouvelles. Les unités absentes du snapshot restent manquantes. Les unités
    test ne doivent pas servir à constituer le portefeuille futur à prédire.
    """
    validate_features(features)
    origin = prediction_origin(year)
    past = known_leases(leases, origin)
    ambiguous = past.duplicated(list(UNIT_KEY)+['_start'], keep=False)
    past['_ambiguous'] = ambiguous
    snapshot = past.sort_values(list(UNIT_KEY)+['_start'], kind='stable').drop_duplicates(list(UNIT_KEY), keep='last')
    snapshot = snapshot.loc[~snapshot['_ambiguous']]
    cols = [c for c in SOURCE_FEATURES.values() if c in snapshot]
    merged = observations[list(UNIT_KEY)].reset_index(drop=True).merge(
        snapshot[list(dict.fromkeys(list(UNIT_KEY)+cols+['_start']))],
        on=list(UNIT_KEY), how='left', validate='many_to_one', sort=False)
    result = pd.DataFrame(index=observations.index)
    for feature, source in SOURCE_FEATURES.items():
        result[feature] = merged[source].where(merged['_start'].notna()).to_numpy() if source in merged else np.nan
    result['forecast_year'] = int(year)
    result['months_since_known_start'] = ((origin-merged['_start']).dt.total_seconds()/86400/(365.25/12)).to_numpy()
    return result.loc[:, list(features)].copy()


def available_external_context(external, year):
    """Bloc optionnel à la coupure ; aucune jointure provinciale implicite."""
    return get_available_external_data(external, prediction_origin(year))


def known_building_weights(leases, year):
    """Parts d'unités distinctes par bâtiment dans l'extrait connu à la coupure.

    Ce sont des poids de l'extrait connu, pas les volumes réels du portefeuille.
    Chaque unité est affectée à son dernier bâtiment connu, sans double compte.
    """
    d = known_leases(leases, prediction_origin(year))
    d = d.sort_values('_start', kind='stable').drop_duplicates(list(UNIT_KEY), keep='last')
    counts = d.groupby('sBuilding', observed=True).size()
    return counts/counts.sum() if counts.sum() else counts.astype(float)


def aggregate_growth(data, values, method='transition_median', weights=None):
    """Agrégat en fraction et couverture ; aucun poids futur n'est construit.

    Les variantes par bâtiment combinent leurs médianes. Un bâtiment absent
    n'est jamais imputé ; la masse de poids perdue est rendue explicite.
    """
    v = pd.Series(np.asarray(values, dtype=float), index=data.index)
    valid = np.isfinite(v)
    if method == 'transition_median':
        return {'growth': float(v[valid].median()), 'coverage': float(valid.mean()), 'weight_coverage': 1.0 if valid.any() else 0.0}
    if method == 'transition_mean':
        return {'growth': float(v[valid].mean()), 'coverage': float(valid.mean()), 'weight_coverage': 1.0 if valid.any() else 0.0}
    groups = v[valid].groupby(data.loc[valid, 'sBuilding']).median()
    if method == 'building_equal':
        return {'growth': float(groups.mean()), 'coverage': float(valid.mean()), 'weight_coverage': np.nan}
    if method != 'known_unit_weights' or weights is None:
        raise ValueError('Méthode/poids d’agrégation manquants ou inconnus.')
    if weights.empty:
        return {'growth':np.nan, 'coverage':float(valid.mean()), 'weight_coverage':0.0}
    if not weights.index.is_unique or not np.isfinite(weights).all() or (weights<0).any() or not np.isclose(weights.sum(),1):
        raise ValueError('Poids invalides : parts finies, positives, uniques, somme 1.')
    aligned = weights.reindex(groups.index).dropna()
    mass = float(aligned.sum())
    growth = float((groups.loc[aligned.index]*aligned).sum()/mass) if mass else np.nan
    return {'growth': growth, 'coverage': float(valid.mean()), 'weight_coverage': mass}


def target_summary(data, groups=()):
    """Diagnostics agrégés, masqués sous cinq valeurs ; jamais de lignes CRM."""
    rows = []
    grouped = data.groupby(list(groups), dropna=False, observed=True) if groups else [((),data)]
    for key, part in grouped:
        key = key if isinstance(key, tuple) else (key,)
        for target in TARGETS:
            v = part[target].dropna()
            row = dict(zip(groups,key))
            row.update(target=target, transitions=len(part), valides=len(v), couverture=len(v)/len(part))
            for name, value in {'moyenne':v.mean(), 'mediane':v.median(), 'ecart_type':v.std(),
                                'q05':v.quantile(.05), 'q95':v.quantile(.95)}.items():
                row[name] = value if len(v)>=5 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def build_prediction_cohort(leases, year):
    """Unités candidates à échéance en Y, sélectionnées sans nouveau bail.

    Dernier bail commencé et signé à la coupure, début non ambigu et fin
    contractuelle valide dans Y. Une échéance indique une possibilité de
    transition, pas une transition certaine. Aucun terme arrondi pour imputer
    une fin manquante. Sortie en mémoire uniquement, sans cible ni loyer futur.
    La fidélité historique des dates contractuelles reste à confirmer.
    """
    past = known_leases(leases, prediction_origin(year))
    past['_ambiguous'] = past.duplicated(list(UNIT_KEY)+['_start'], keep=False)
    latest = past.sort_values(list(UNIT_KEY)+['_start'], kind='stable').drop_duplicates(list(UNIT_KEY), keep='last')
    end = pd.to_datetime(latest.sLeaseTo, errors='coerce', format='mixed')
    eligible = ~latest['_ambiguous'] & end.ge(latest['_start']) & end.dt.year.eq(year)
    out = latest.loc[eligible, list(UNIT_KEY)].copy()
    out['known_lease_start'] = latest.loc[eligible, '_start']
    out['known_sign_date'] = latest.loc[eligible, '_sign']
    out['known_lease_end'] = end.loc[eligible]
    return out.sort_values(list(UNIT_KEY), kind='stable').reset_index(drop=True)


def build_occurrence_features(leases, cohort, year):
    """Features à coupure : liste admise + mois d'échéance connu, sans labels."""
    x = build_features(leases, cohort, year)
    x['expiry_month'] = pd.to_datetime(cohort.known_lease_end).dt.month.to_numpy()
    return x


def reveal_occurrence(cohort, transitions, year, label_cutoff=None):
    """Révéler des labels sur une cohorte déjà fixée ; aucun prédicteur ici.

    0 = aucune transition admissible observée, pas absence d'évolution du loyer.
    unit_growth est la moyenne des croissances annualisées admissibles d'une
    unité (plusieurs transitions possibles). À date, un label positif est mûr
    seulement après label_available_date ; un négatif avec paire immature
    reste inconnu. Une année non terminée ne fournit aucun label.
    """
    out = cohort[list(UNIT_KEY)].copy()
    out['occurrence_target'] = np.nan
    out['unit_growth'] = np.nan
    if label_cutoff is not None and pd.Timestamp(label_cutoff) < pd.Timestamp(int(year),12,31):
        return out
    annual = transitions.loc[transitions.year.eq(year)].copy()
    valid = annual.eligible_main & np.isfinite(annual[MAIN_TARGET])
    pending = pd.Series(False, index=annual.index)
    if label_cutoff is not None:
        dates = pd.to_datetime(annual.label_available_date)
        pending = dates.isna() | dates.gt(pd.Timestamp(label_cutoff))
        valid &= ~pending
    growth = annual.loc[valid].groupby(list(UNIT_KEY))[MAIN_TARGET].mean().rename('unit_growth')
    out = out.drop(columns='unit_growth').merge(growth.reset_index(),on=list(UNIT_KEY),how='left',validate='one_to_one')
    out['occurrence_target'] = out.unit_growth.notna().astype(float)
    if pending.any():
        unknown = annual.loc[pending,list(UNIT_KEY)].drop_duplicates().assign(_pending=True)
        out = out.merge(unknown,on=list(UNIT_KEY),how='left',validate='one_to_one')
        out.loc[out._pending.eq(True) & out.occurrence_target.eq(0),'occurrence_target'] = np.nan
        out = out.drop(columns='_pending')
    return out
