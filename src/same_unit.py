"""Moteur historique à unité constante, sans export ni modification des entrées.

L'ordre repose sur sLeaseFrom. Les loyers invalides restent dans la succession
et ne sont jamais sautés pour chercher un précédent « valide ». Une unité avec
une date de début invalide est mise à part : son ordre ne peut être établi.
Les égalités de dates sont conservées, mais les transitions touchant une date
ambiguë ne reçoivent pas de croissance valide. Aucun filtre métier définitif
ni filtre d'outliers n'est appliqué.
"""
import numpy as np
import pandas as pd

UNIT_KEY = ('sPropCode', 'sUnitCode')
PRICES = {'contractual': 'sRent', 'effective': 'sRentEffective'}
CHARACTERISTICS = ('hBuilding', 'sBuilding', 'sCity', 'sState', 'sBeds',
                   'sBaths', 'sSqft', 'sFloor', 'sUnitType', 'sUnitSubtype')


def build_same_unit_pairs(leases):
    """Retourner les paires en mémoire et un bilan agrégé de traçabilité.

    Les identifiants doivent être fournis comme chaînes pour conserver leurs
    zéros initiaux. Les champs old_/new_ gardent les valeurs sources ; les
    conversions numériques et de dates sont temporaires. Les flags décrivent
    les exclusions mathématiques et les réserves de comparabilité séparément.
    Les croissances sont des fractions (0.05 = 5 %).

    months_between = days_between / (365.25 / 12) : durée en mois équivalents,
    pas nombre de mois calendaires ni durée contractuelle sTermMonths.
    """
    required = list(UNIT_KEY) + ['sLeaseFrom', 'sRent', 'sRentEffective']
    missing = [c for c in required if c not in leases]
    if missing:
        raise ValueError(f'Colonnes obligatoires absentes : {missing}')
    d = leases.copy(deep=True)
    for col in UNIT_KEY:
        if not all(isinstance(x, str) for x in d[col].dropna()):
            raise ValueError(f'{col} doit être chargé comme chaîne (dtype="string").')
        d[col] = d[col].astype('string')
    d['_source_row'] = np.arange(len(d))
    d['_date'] = pd.to_datetime(d['sLeaseFrom'], errors='coerce', format='mixed')
    missing_key = d[list(UNIT_KEY)].isna().any(axis=1)
    missing_key |= d[list(UNIT_KEY)].apply(lambda s: s.str.strip().eq('')).any(axis=1)
    identified = d.loc[~missing_key].copy()
    sizes = identified.groupby(list(UNIT_KEY), sort=False).size()
    theoretical = int((sizes - 1).sum())
    invalid_unit = identified.groupby(list(UNIT_KEY))['_date'].transform(lambda x: x.isna().any())
    ordered = identified.loc[~invalid_unit].copy()
    ordered['_duplicate_row'] = leases.duplicated(keep=False).to_numpy()[ordered['_source_row']]
    ordered['_ambiguous_date'] = ordered.duplicated(list(UNIT_KEY) + ['_date'], keep=False)
    ordered = ordered.sort_values(list(UNIT_KEY) + ['_date', '_source_row'], kind='stable')
    groups = ordered.groupby(list(UNIT_KEY), sort=False)
    previous = groups.shift(1)
    has_previous = groups.cumcount().gt(0)
    current = ordered.loc[has_previous].reset_index(drop=True)
    previous = previous.loc[has_previous].reset_index(drop=True)
    pairs = current[list(UNIT_KEY)].copy()
    # Identité ancienne/nouvelle explicitement traçable, sans appariement inter-unités.
    old_keys = groups[list(UNIT_KEY)].shift(1).loc[has_previous].reset_index(drop=True)
    for col in UNIT_KEY:
        pairs['old_' + col] = old_keys[col]
        pairs['new_' + col] = current[col]
    fields = [c for c in ['sLeaseFrom', 'sLeaseTo', 'sSignDate', 'sRent', 'sRentEffective',
                          'sConcession', 'sTermMonths', 'sTermSeq', 'sRenewal', *CHARACTERISTICS]
              if c in ordered]
    for col in fields:
        pairs['old_' + col] = previous[col]
        pairs['new_' + col] = current[col]
    for col in ['sRenewal', *CHARACTERISTICS]:
        if col in current:
            pairs[col] = current[col]
    pairs['old_lease_date'] = previous['_date']
    pairs['new_lease_date'] = current['_date']
    pairs['year'] = pairs['new_lease_date'].dt.year
    pairs['days_between'] = (pairs['new_lease_date'] - pairs['old_lease_date']).dt.total_seconds() / 86400
    pairs['months_between'] = pairs['days_between'] / (365.25 / 12)
    pairs['flag_same_key'] = pd.concat([
        pairs['old_'+c].eq(pairs['new_'+c]) for c in UNIT_KEY], axis=1).all(axis=1)
    pairs['flag_nonpositive_interval'] = pairs['days_between'].le(0)
    pairs['flag_ambiguous_date'] = previous['_ambiguous_date'] | current['_ambiguous_date']
    pairs['flag_duplicate_source_row'] = previous['_duplicate_row'] | current['_duplicate_row']
    pairs['valid_temporal'] = pairs['flag_same_key'] & ~pairs['flag_nonpositive_interval'] & ~pairs['flag_ambiguous_date']
    # Fenêtre du starter : diagnostic candidat uniquement, pas exclusion automatique.
    pairs['flag_outside_starter_interval'] = ~pairs['days_between'].div(365.25).between(.5, 2.5, inclusive='neither')
    if 'sLeaseTo' in current:
        old_end = pd.to_datetime(previous['sLeaseTo'], errors='coerce', format='mixed')
        new_end = pd.to_datetime(current['sLeaseTo'], errors='coerce', format='mixed')
        pairs['flag_invalid_lease_end'] = old_end.isna() | new_end.isna()
        pairs['flag_reversed_lease_dates'] = old_end.lt(previous['_date']) | new_end.lt(current['_date'])
        pairs['flag_overlap'] = old_end.gt(current['_date'])
    if 'sSignDate' in current:
        old_sign = pd.to_datetime(previous['sSignDate'], errors='coerce', format='mixed')
        new_sign = pd.to_datetime(current['sSignDate'], errors='coerce', format='mixed')
        pairs['flag_invalid_sign_date'] = old_sign.isna() | new_sign.isna()
        pairs['flag_sign_after_start'] = old_sign.gt(previous['_date']) | new_sign.gt(current['_date'])
    if 'sTermSeq' in current:
        a = pd.to_numeric(previous['sTermSeq'], errors='coerce')
        b = pd.to_numeric(current['sTermSeq'], errors='coerce')
        pairs['term_seq_delta'] = b-a
        pairs['flag_invalid_term_sequence'] = a.isna() | b.isna()
        pairs['flag_term_sequence_not_next'] = (b-a).ne(1)
    if 'sTermMonths' in current:
        a = pd.to_numeric(previous['sTermMonths'], errors='coerce')
        b = pd.to_numeric(current['sTermMonths'], errors='coerce')
        pairs['flag_invalid_term_months'] = a.isna() | b.isna() | a.le(0) | b.le(0)
        pairs['flag_atypical_term'] = a.isin([5.9, 17.9, 23.9]) | b.isin([5.9, 17.9, 23.9])
        pairs['interval_minus_old_term_months'] = pairs['months_between'] - a
    if 'sRenewal' in current:
        pairs['flag_unknown_renewal'] = ~pd.to_numeric(current['sRenewal'], errors='coerce').isin([0,1])
    changes = []
    for col in CHARACTERISTICS:
        if col in current:
            a, b = previous[col], current[col]
            changed = ~(a.eq(b) | (a.isna() & b.isna())).fillna(False)
            pairs['flag_changed_' + col] = changed
            changes.append(changed)
    pairs['flag_characteristics_changed'] = pd.concat(changes, axis=1).any(axis=1) if changes else False
    pairs = pairs.copy()
    for target, col in PRICES.items():
        a = pd.to_numeric(previous[col], errors='coerce').astype(float)
        b = pd.to_numeric(current[col], errors='coerce').astype(float)
        pairs[f'flag_{target}_missing_rent'] = previous[col].isna() | current[col].isna()
        pairs[f'flag_{target}_nonnumeric_rent'] = (previous[col].notna() & a.isna()) | (current[col].notna() & b.isna())
        pairs[f'flag_{target}_zero_rent'] = a.eq(0) | b.eq(0)
        pairs[f'flag_{target}_negative_rent'] = a.lt(0) | b.lt(0)
        pairs[f'flag_{target}_nonfinite_rent'] = ~(np.isfinite(a) & np.isfinite(b))
        valid = pairs['valid_temporal'] & np.isfinite(a) & np.isfinite(b) & a.gt(0) & b.gt(0)
        # np.divide ne calcule que les ratios valides ; aucun zéro n'est divisé.
        ratio = np.full(len(pairs), np.nan)
        np.divide(b.to_numpy(dtype=float), a.to_numpy(dtype=float), out=ratio, where=valid.to_numpy())
        growth = ratio - 1
        valid = valid & np.isfinite(growth)
        pairs[f'valid_{target}'] = valid
        pairs[f'{target}_growth'] = pd.Series(growth).where(valid)
        with np.errstate(over='ignore', invalid='ignore'):
            annual = np.expm1(np.log(ratio) * 12 / pairs['months_between'].where(valid))
        annual_valid = valid & np.isfinite(annual)
        pairs[f'valid_{target}_annualized'] = annual_valid
        pairs[f'{target}_annualized_growth'] = pd.Series(annual).where(annual_valid)
        # Diagnostic robuste 3 × IQR : signal statistique, jamais filtre d'exclusion.
        q1, q3 = pairs[f'{target}_growth'].quantile([.25,.75])
        pairs[f'flag_{target}_extreme_iqr'] = (pairs[f'{target}_growth'].lt(q1-3*(q3-q1)) |
                                              pairs[f'{target}_growth'].gt(q3+3*(q3-q1)))
    affected_sizes = identified.loc[invalid_unit].groupby(list(UNIT_KEY)).size()
    trace = {'total_leases': len(d), 'identified_units': len(sizes),
             'missing_key_leases': int(missing_key.sum()),
             'invalid_start_date_leases': int(d['_date'].isna().sum()),
             'theoretical_transitions': theoretical,
             'quarantined_date_units': len(affected_sizes),
             'quarantined_date_leases': int(invalid_unit.sum()),
             'unconstructed_date_transitions': int((affected_sizes-1).sum()),
             'constructed_pairs': len(pairs), 'paired_units': len(pairs[list(UNIT_KEY)].drop_duplicates()),
             'valid_temporal_pairs': int(pairs['valid_temporal'].sum())}
    for target in PRICES:
        trace[f'valid_{target}_pairs'] = int(pairs[f'valid_{target}'].sum())
        trace[f'valid_{target}_annualized_pairs'] = int(pairs[f'valid_{target}_annualized'].sum())
    assert trace['constructed_pairs'] + trace['unconstructed_date_transitions'] == theoretical
    return pairs, trace
