"""Diagnostics économiques en mémoire, sans conversion finale ni export.

Les concessions sont reliées aux baux par UNIT_KEY et date de début comprise
entre début et fin du bail (bornes inclusives). Une association n'est retenue
que si un seul bail est candidat. La reconstruction prudente exclut du calcul
les baux à détails dépassant leur fin, dupliqués ou incohérents ; ils restent
présents dans les sorties et sont comptés. L'absence de détail ne prouve pas
l'absence de concession.
"""
import numpy as np
import pandas as pd
from .same_unit import UNIT_KEY

CANDIDATES = ('baseline', 'line_total', 'monthly_all', 'promo_once', 'indicator_monthly')


def numeric(values):
    """Conversion temporaire ; valeurs non finies remplacées par NaN."""
    result = pd.to_numeric(values, errors='coerce').astype(float)
    return result.where(np.isfinite(result))


def calculate_gaps(leases):
    """Retourner une copie avec écarts absolus ($) et relatifs (fractions)."""
    out = leases.copy(deep=True)
    contracted, effective = numeric(out.sRent), numeric(out.sRentEffective)
    out['contractual_effective_gap'] = contracted - effective
    out['relative_gap'] = out.contractual_effective_gap.div(contracted.where(contracted.gt(0)))
    return out


def attach_concessions(leases, concessions):
    """Retourner baux enrichis, associations uniques et bilan agrégé.

    Les identifiants _lease_id et _concession_id sont des positions locales,
    pas des identifiants métier. Les entrées et leur index ne sont pas modifiés.
    hUnit/hProperty servent de contrôle secondaire si disponibles ; la clé
    officielle reste sPropCode/sUnitCode. Aucun total partiel n'est interprété
    comme un total exhaustif de concessions pour le bail.
    """
    l, c = leases.copy(deep=True).reset_index(drop=True), concessions.copy(deep=True).reset_index(drop=True)
    for df in [l,c]:
        for col in UNIT_KEY:
            if not all(isinstance(x,str) for x in df[col].dropna()):
                raise ValueError(f'{col} doit être chargé comme chaîne.')
            df[col] = df[col].astype('string')
    l['_lease_id'] = np.arange(len(l)); c['_concession_id'] = np.arange(len(c))
    l['_lease_start'] = pd.to_datetime(l.sLeaseFrom, errors='coerce', format='mixed')
    l['_lease_end'] = pd.to_datetime(l.sLeaseTo, errors='coerce', format='mixed')
    c['_con_start'] = pd.to_datetime(c.sDateFrom, errors='coerce', format='mixed')
    c['_con_end'] = pd.to_datetime(c.sDateTo, errors='coerce', format='mixed')
    c['_duplicate_detail'] = concessions.duplicated(keep=False).to_numpy()
    c['_signed_amount'] = numeric(c.sAmount)
    c['_detail_months'] = numeric(c.sMonths)
    c['_monthly_amount'] = c._signed_amount * c._detail_months
    # Hypothèse mixte seulement : PromoPay ponctuel, autres lignes mensuelles.
    c['_mixed_amount'] = c._signed_amount * c._detail_months.where(c.sChargeCode.ne('PromoPay'),1)
    valid_l = l[list(UNIT_KEY)].notna().all(axis=1) & l._lease_start.notna() & l._lease_end.ge(l._lease_start)
    valid_c = c[list(UNIT_KEY)].notna().all(axis=1) & c._con_start.notna() & c._con_end.ge(c._con_start)
    for col in UNIT_KEY:
        valid_l &= l[col].str.strip().ne('').fillna(False)
        valid_c &= c[col].str.strip().ne('').fillna(False)
    lease_cols = list(UNIT_KEY)+['_lease_id','_lease_start','_lease_end']
    handles = [x for x in ['hUnit','hProperty'] if x in l and x in c]
    candidates = c.loc[valid_c].merge(l.loc[valid_l,lease_cols+handles],on=list(UNIT_KEY),
                                    suffixes=('_con','_lease'),how='inner',validate='many_to_many')
    contained = candidates._con_start.ge(candidates._lease_start) & candidates._con_start.le(candidates._lease_end)
    candidates = candidates.loc[contained].copy()
    multiplicity = candidates.groupby('_concession_id').size()
    unique_ids = multiplicity[multiplicity.eq(1)].index
    links = candidates.loc[candidates._concession_id.isin(unique_ids)].copy()
    links['flag_crosses_lease_end'] = links._con_end.gt(links._lease_end)
    links['flag_handle_mismatch'] = False
    for handle in handles:
        links['flag_handle_mismatch'] |= links[handle+'_con'].astype('string').ne(
            links[handle+'_lease'].astype('string')).fillna(True)
    links['flag_bad_value'] = links._signed_amount.isna() | links._detail_months.isna() | links._detail_months.le(0)
    links['flag_unsafe_detail'] = links.flag_crosses_lease_end | links.flag_handle_mismatch | links.flag_bad_value | links._duplicate_detail
    agg = links.groupby('_lease_id').agg(detail_count=('_concession_id','size'),
        signed_line_total=('_signed_amount',lambda x:x.sum(min_count=1)),
        signed_monthly_total=('_monthly_amount',lambda x:x.sum(min_count=1)),
        signed_mixed_total=('_mixed_amount',lambda x:x.sum(min_count=1)),
        unsafe_detail_count=('flag_unsafe_detail','sum'), crosses_end_count=('flag_crosses_lease_end','sum'),
        concession_types=('sChargeCode',lambda x:' + '.join(sorted(set(x.dropna())))))
    # Baux potentiellement concernés par une concession ambiguë : aucune attribution forcée.
    ambiguous_lease_ids = candidates.loc[~candidates._concession_id.isin(unique_ids),'_lease_id'].unique()
    out = calculate_gaps(l).merge(agg,on='_lease_id',how='left',validate='one_to_one')
    for col in ['detail_count','unsafe_detail_count','crosses_end_count']:
        out[col] = out[col].fillna(0).astype(int)
    out['concession_types'] = out.concession_types.fillna('Aucun détail associé')
    out['flag_ambiguous_concession'] = out._lease_id.isin(ambiguous_lease_ids)
    out['flag_indicator_disagreement'] = (numeric(out.sConcession).eq(1) & out.detail_count.eq(0)) | (numeric(out.sConcession).eq(0) & out.detail_count.gt(0))
    out['eligible_details'] = out.detail_count.gt(0) & out.unsafe_detail_count.eq(0) & ~out.flag_ambiguous_concession
    report = {'concessions':len(c),'leases':len(l),'invalid_concession_key_or_dates':int((~valid_c).sum()),
              'invalid_lease_key_or_dates':int((~valid_l).sum()), 'candidate_links':len(candidates),
              'uniquely_assigned_concessions':len(links),'ambiguous_concessions':int(multiplicity.gt(1).sum()),
              'unmatched_valid_concessions':int(valid_c.sum()-len(multiplicity)),
              'fully_contained_assigned_concessions':int((~links.flag_crosses_lease_end).sum()),
              'crossing_assigned_concessions':int(links.flag_crosses_lease_end.sum()),
              'handle_mismatch_links':int(links.flag_handle_mismatch.sum()),
              'leases_with_details':int(out.detail_count.gt(0).sum()),
              'eligible_detail_leases':int(out.eligible_details.sum()),
              'indicator_disagreement_leases':int(out.flag_indicator_disagreement.sum())}
    assert len(out)==len(leases) and out._lease_id.is_unique
    return out, links, report


def reconstruct_candidates(enriched, prudent=True):
    """Calculer cinq hypothèses sans estimer de coefficients.

    baseline : sRent ; line_total : sRent + sum(sAmount)/sTermMonths ;
    monthly_all : sRent + sum(sAmount*sMonths)/sTermMonths ;
    promo_once : sRent + sum(sAmount*(1 si PromoPay sinon sMonths))/sTermMonths.
    indicator_monthly : monthly_all si sConcession=1, sRent si sConcession=0.
    Les autres codes restent indéterminés. Les montants gardent leur signe.
    Le périmètre prudent exige des détails
    intégralement contenus et fiables. prudent=False est une sensibilité sur
    les mêmes associations uniques, pas une validation de leur période.
    """
    out=enriched.copy(deep=True)
    rent,term=numeric(out.sRent),numeric(out.sTermMonths)
    eligible = out.eligible_details if prudent else (out.detail_count.gt(0) & ~out.flag_ambiguous_concession)
    valid = eligible & term.gt(0) & rent.gt(0)
    out['reconstructed_baseline']=rent.where(rent.gt(0))
    for candidate,col in [('line_total','signed_line_total'),('monthly_all','signed_monthly_total'),('promo_once','signed_mixed_total')]:
        reconstructed=rent+numeric(out[col]).div(term.where(valid))
        out['reconstructed_'+candidate]=reconstructed.where(valid & np.isfinite(reconstructed))
    indicator = numeric(out.sConcession)
    out['reconstructed_indicator_monthly'] = out.reconstructed_monthly_all.where(indicator.eq(1))
    out.loc[indicator.eq(0), 'reconstructed_indicator_monthly'] = out.loc[indicator.eq(0), 'reconstructed_baseline']
    for candidate in CANDIDATES:
        out['error_'+candidate]=out['reconstructed_'+candidate]-numeric(out.sRentEffective)
        out['flag_negative_reconstruction_'+candidate]=out['reconstructed_'+candidate].lt(0)
    return out


def error_metrics(df, groups=None, min_count=5):
    """Agrégats des erreurs signées reconstruction − observé, sans lignes brutes."""
    rows=[]
    grouped=[((),df)] if not groups else df.groupby(groups,observed=True,dropna=False)
    for key,part in grouped:
        key=key if isinstance(key,tuple) else (key,)
        for candidate in CANDIDATES:
            errors=part['error_'+candidate].dropna()
            row=dict(zip(groups or [],key));row.update(candidate=candidate,effectif=len(errors))
            metrics={'MAE ($)':errors.abs().mean(),'erreur absolue médiane ($)':errors.abs().median(),
                     'biais moyen ($)':errors.mean(),'Q05 erreur ($)':errors.quantile(.05),
                     'Q95 erreur ($)':errors.quantile(.95),'part erreur ≤ 1 $ (%)':errors.abs().le(1).mean()*100}
            row.update({k:v if len(errors)>=min_count else np.nan for k,v in metrics.items()})
            rows.append(row)
    return pd.DataFrame(rows)
