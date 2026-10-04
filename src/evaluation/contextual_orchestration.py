"""Audit à cutoff, orchestration figée et scoring agrégé après prédiction."""
import numpy as np
import pandas as pd

from src.models.contextual_experts import RULES, rules_fingerprint, _prepare_experts, _orchestrate_prepared
from src.evaluation.model_ab_tournament import YEARS, B_LABEL, error_summary
from src.evaluation.model_d_evaluation import run_model_d_experiment
from src.features.model_a_features import build_model_a_features
from src.features.model_dataset import UNIT_KEY, prediction_origin
from src.external_data import get_available_external_data


def _median(values):
    v = pd.to_numeric(values,errors='coerce').dropna()
    return float(100*v.median()) if len(v)>=5 else np.nan


def _audit_prepared(f, external=None):
    """Statistiques historiques seulement ; aucune cible réalisée de Y."""
    year = f['year']
    last, known, positive = f['last'], f['last_known'], f['last_positive']
    rolling = f['history'].loc[f['history'].forecast_year.between(year-3,year-1)]
    rolling_positive = rolling.loc[rolling.occurrence_target.eq(1)]
    pairs, mature = f['pairs_recent'], f['pairs_recent_mature']
    rows = []
    def add(component, signal, definition, support, coverage, estimate, status, reason):
        rows.append({'year':year,'component':component,'signal':signal,'definition':definition,
                     'support':support,'coverage':coverage,'estimate_pct':estimate,
                     'status':status,'reason':reason,'availability':'SAFE_WITH_CONDITIONS'
                     if status not in ('REJECT_UNQUALIFIED','REJECT_EXTERNAL_FOR_C') else 'UNKNOWN'})
    add('growth','C_growth : dernière cohorte historique', 'Médiane G des unités positives de Y−1 mûres à D.',
        len(positive),f['coverage'],_median(positive.unit_growth),
        'SELECTED' if f['C_g_valid'] else 'NONE',
        'Support >=30 et couverture labels >=80% requis ; aucune imputation.')
    add('growth','Croissance récente par transition','Médiane effective annualisée des transitions de Y−1 mûres à D.',
        len(mature),len(mature)/len(pairs) if len(pairs) else 0.,_median(mature.target_effective_annualized),
        'AUDIT_ONLY','Maturité sélective ; population différente des unités/années A. Pas de second C testé.')
    add('growth','Médiane glissante trois cohortes','Médiane G positifs des cohortes Y−3 à Y−1 qualifiés à D.',
        len(rolling_positive),float(rolling.occurrence_target.notna().mean()) if len(rolling) else 0.,
        _median(rolling_positive.unit_growth),'AUDIT_ONLY','Retard de maturité ; signal plus ancien et proche de A. Pas de recherche de fenêtre.')
    add('growth','Divergence récent/long terme','C_growth moins médiane historique de A, si C qualifié.',
        len(positive),f['coverage'],100*(f['C_g']-f['A_g'][0]) if f['C_g_valid'] else np.nan,
        'CONTEXT_ONLY','Confiance de C, jamais expert indépendant supplémentaire.')
    for value, name in ((1,'Momentum renouvellement'),(0,'Momentum relocation')):
        sub = mature.loc[pd.to_numeric(mature.get('new_sRenewal'),errors='coerce').eq(value)]
        add('growth',name,'Croissance des transitions historiques mûres de Y−1, statut passé seulement.',
            len(sub),len(sub)/len(pairs) if len(pairs) else 0.,_median(sub.target_effective_annualized),
            'AUDIT_ONLY','Faible historique récent ; type futur inconnu, aucune affectation aux candidates.')
    for signal in ('Momentum niveau effectif','Momentum gap contractuel/effectif','Régime concessions','Momentum asking'):
        add('growth',signal,'Qualification source existante conservée, aucun nouvel indicateur construit.',
            np.nan,np.nan,np.nan,'REJECT_UNQUALIFIED','Hors prédicteurs qualifiés par la fondation ; versions ou disponibilité non démontrées.')
    add('occurrence','C_occurrence : dernière cohorte historique','Moyenne O connus de Y−1 à D ; pending reste inconnu.',
        len(known),f['coverage'],float(100*known.occurrence_target.mean()) if len(known)>=5 else np.nan,
        'SELECTED' if f['C_p_valid'] else 'NONE','Support >=30 et couverture >=80% requis ; sous-échantillon immature non représentatif.')
    for signal in ('Taux récent par bâtiment','Momentum occurrence portefeuille','Dynamique occurrence bâtiment','Maturité/support récent'):
        add('occurrence',signal,'Taux/support de la dernière cohorte, historique par bâtiment ou portefeuille uniquement.',
            len(known),f['coverage'],np.nan,'AUDIT_ONLY','Même censure ; ne pas fabriquer un expert de rupture à partir des absences futures.')
    for signal in ('Composition des termes','Concentration termes 12 mois','Concentration échéances','Changement de mix candidat','HHI bâtiment'):
        add('occurrence',signal,'Caractéristiques du dernier bail connu et composition candidate à D.',
            len(f['X']),1.,np.nan,'CONTEXT_ONLY','Contexte reconstructible, mais aucune relation causale/probabilité d’occurrence validée.')
    eligible = len(get_available_external_data(external,prediction_origin(year))) if external is not None else 0
    for component in ('growth','occurrence'):
        add(component,'Régime externe/réglementaire','Filtre public existant : date vérifiée <=D et valeur présente.',
            eligible,np.nan,np.nan,'REJECT_EXTERNAL_FOR_C','Aucun signal historiquement qualifié et utile dans ces folds ; pas de collecte ni réaudit économique.')
    return pd.DataFrame(rows)


def audit_contextual_signals(leases, external=None):
    """Phase A/B indépendante du scoring ; uniquement tableaux agrégés."""
    return pd.concat([_audit_prepared(_prepare_experts(leases,y),external) for y in YEARS],ignore_index=True)


def _diagnostics(f, labels):
    o = labels.transition_occurred.to_numpy(dtype=float)
    realized_growth = labels.conditional_growth.to_numpy(dtype=float)
    real = 100*float(np.median(np.where(o==0,0.,realized_growth)))
    row = {'year':f['year'],'cohort_size':len(o),'realized_transition_pct':100*float(o.mean()),
           'realized_conditional_growth_pct':float(100*np.median(realized_growth[o==1])),
           'predicted_P1_pct':f['P1_pct'],'realized_P1_pct':real,
           'error_pp':f['P1_pct']-real,'absolute_error_pp':abs(f['P1_pct']-real),
           'occurrence_error_pp':100*float(f['p'].mean()-o.mean()),
           'Brier':float(np.mean((f['p']-o)**2)),
           'growth_median_error_pp':100*float(np.median(f['g'][o==1])-np.median(realized_growth[o==1])),
           'growth_MAE_pp':100*float(np.abs(f['g'][o==1]-realized_growth[o==1]).mean()),
           'recent_known_labels':len(f['last_known']),'recent_positive_labels':len(f['last_positive']),
           'recent_label_coverage':f['coverage'],
           'building_HHI':float((f['X'].building.value_counts(normalize=True)**2).sum()),
           'expiry_HHI':float((f['X'].expiry_month.value_counts(normalize=True)**2).sum()),
           'term12_share':float(pd.to_numeric(f['X'].prior_term_months,errors='coerce').eq(12).mean())}
    buildings=[]
    for component,letter in (('occurrence','p'),('growth','g')):
        weights=f['weights_'+letter];context=f[letter+'_context']
        row['final_'+component+'_pct'] = 100*float(f[letter].mean() if letter=='p' else np.median(f[letter][o==1]))
        for j, expert in enumerate(('A','B','C')):
            val=f[expert+'_'+letter]
            row[expert+'_'+component+'_pct'] = float(100*np.mean(val)) if np.isfinite(val).all() else np.nan
            if letter=='g' and expert!='C': row[expert+'_'+component+'_pct']=float(100*np.median(val[o==1]))
            row['weight_'+expert+'_'+letter]=float(weights[:,j].mean())
            row['activation_'+expert+'_'+letter]=float((weights[:,j]>0).mean())
        row['A_dominance_'+letter]=float((np.argmax(weights,axis=1)==0).mean())
        row['B_global_fallback_share_'+letter]=float(context.b_fallback.mean())
        row['B_support_share_'+letter]=float((~context.b_fallback).mean())
        row['B_median_support_'+letter]=float(context.b_support.median())
        row['B_deviation_from_global_pp_'+letter]=100*float(np.abs(f['B_'+letter]-context.b_global).mean())
        for building in f['X'].building.dropna().unique():
            mask=f['X'].building.eq(building).to_numpy(dtype=bool)
            item={'year':f['year'],'building':building,'component':component,
                  'candidate_units':int(mask.sum()),'cohort_share_pct':100*float(mask.mean()),
                  'B_support_mean':float(context.loc[mask,'b_support'].mean()),
                  'B_global_fallback_share':float(context.loc[mask,'b_fallback'].mean()),
                  'C_selected':f['C_'+letter+'_valid'],
                  'recent_support':int(context.c_support.iloc[0]),'recent_coverage':f['coverage']}
            for j, expert in enumerate(('A','B','C')): item['weight_'+expert]=float(weights[mask,j].mean())
            item['final_component_pct']=100*float(f[letter][mask].mean() if letter=='p' else np.median(f[letter][mask]))
            buildings.append(item)
    row['hierarchical_child_support_share_p']=float(f['child_supported'].mean())
    row['B_parent_fallback_share_p']=float((~f['child_supported'] & ~f['p_context'].b_fallback.to_numpy()).mean())
    return row,buildings


def run_contextual_orchestration(leases, external=None, *, expected_fingerprint=None):
    """Scoring uniquement 2023–2025 après freeze ; aucune règle optimisée.

    La chaîne de prédiction est réutilisable à un autre cutoff. Cette fonction
    ne prépare, ne prédit et ne score jamais l'année 2026 pour l'orchestration.
    """
    fingerprint=rules_fingerprint()
    if expected_fingerprint is None or expected_fingerprint != fingerprint:
        raise ValueError('Empreinte du protocole figé obligatoire avant scoring.')
    prepared=[_prepare_experts(leases,y) for y in YEARS]
    folds=[_orchestrate_prepared(f) for f in prepared]
    # Seulement après fixation de tous les experts et poids : comparateurs,
    # référence de A et résultats réalisés. D reste inchangé et rejeté.
    reference=run_model_d_experiment(leases)
    checks=reference['checks']
    annual,building_rows=[],[]
    for fold in folds:
        year=fold['year'];observed=build_model_a_features(leases,year,include_targets=True)
        if not fold['candidate'].audit[list(UNIT_KEY)].equals(observed.audit[list(UNIT_KEY)]):
            raise AssertionError('Cohorte orchestrée différente de la fondation.')
        row,buildings=_diagnostics(fold,observed.labels)
        ref=reference['results'].loc[reference['results'].year.eq(year)]
        if not np.isclose(row['realized_P1_pct'],ref.realized_pct.iloc[0],rtol=0,atol=1e-10):
            raise AssertionError('Cible réalisée différente de A.')
        for name,letter in (('A','A'),(B_LABEL,'B')):
            forecast=100*float(np.median(fold[letter+'_p']*fold[letter+'_g']))
            if not np.isclose(forecast,ref.loc[ref.model.eq(name),'prediction_pct'].iloc[0],rtol=0,atol=1e-10):
                raise AssertionError('Expert A/B différent de sa référence figée.')
        annual.append(row);building_rows.extend(buildings)
    annual=pd.DataFrame(annual)
    results=pd.concat([reference['results'],pd.DataFrame({
        'year':annual.year,'model':'ORCHESTRATED','prediction_pct':annual.predicted_P1_pct,
        'realized_pct':annual.realized_P1_pct,'error_pp':annual.error_pp,
        'absolute_error_pp':annual.absolute_error_pp})],ignore_index=True)
    if rules_fingerprint()!=fingerprint:
        raise AssertionError('Protocole modifié après scoring.')
    return {'annual':annual,'building_weights':pd.DataFrame(building_rows),'results':results,
            'metrics':error_summary(results),'signal_audit':pd.concat([_audit_prepared(f,external) for f in prepared],ignore_index=True),
            'checks':checks,'rules_fingerprint':fingerprint}
