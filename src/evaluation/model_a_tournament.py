"""Tournoi empirique limité : trois années, sélection déclarée et non imbriquée."""
import numpy as np
import pandas as pd
from src.models.model_a import fold_predictions, predict_from_fold, model_a_config
from src.features.model_a_features import build_model_a_features, assert_foundation_reference
from src.evaluation.model_a_evaluation import p1_diagnostic, growth_metrics
from src.evaluation.backtest import occurrence_metrics

YEARS=(2023,2024,2025)
EQUIVALENCE_PP=.05
MAX_ANNUAL_DEGRADATION_PP=.5


def rank_candidates(summary):
    """Équivalence fixée à 0,05 pp : simplicité puis tie-breaks déclarés.

    Un challenger dégradant un fold de >0,5 pp vs baseline est non promu.
    Seuils opérationnels fixes, pas preuve de significativité statistique.
    """
    s=summary.copy()
    if s.empty or not s.candidate.is_unique:
        raise ValueError('Candidats uniques requis.')
    required=['MAE_P1','RMSE_P1','bias_P1','worst_year_absolute_error','occurrence_brier','growth_mae_pp','complexity_level','interpretability_level','max_degradation_vs_baseline_pp']
    if s[required].isna().any().any() or not np.isfinite(s[required].to_numpy()).all():
        raise ValueError('Métriques finies requises.')
    remaining=s.loc[s.max_degradation_vs_baseline_pp.le(MAX_ANNUAL_DEGRADATION_PP)].copy()
    rejected=s.loc[~s.index.isin(remaining.index)].copy()
    ordered=[]
    while len(remaining):
        equivalent=remaining.loc[remaining.MAE_P1.le(remaining.MAE_P1.min()+EQUIVALENCE_PP)].copy()
        equivalent['absolute_bias']=equivalent.bias_P1.abs()
        equivalent=equivalent.sort_values(['complexity_level','interpretability_level','worst_year_absolute_error',
            'RMSE_P1','absolute_bias','occurrence_brier','growth_mae_pp','candidate'],kind='stable')
        winner=equivalent.iloc[[0]].drop(columns='absolute_bias')
        ordered.append(winner)
        remaining=remaining.drop(winner.index)
    if not ordered:
        raise ValueError('Aucun candidat admissible à la promotion.')
    ranked=pd.concat(ordered+[rejected.sort_values(['MAE_P1','candidate'],kind='stable')],ignore_index=True)
    ranked['eligible_for_promotion']=ranked.max_degradation_vs_baseline_pp.le(MAX_ANNUAL_DEGRADATION_PP)
    ranked['rank']=np.arange(1,len(ranked)+1)
    return ranked


def run_tournament(leases):
    """Prévisions honnêtes par fold ; choix ex post limité aux trois backtests.

    Les meilleures variantes pondérées sont choisies parmi mild/moderate,
    séparément par Brier moyen et MAE growth moyen. Leur combinaison est une
    sélection rétrospective non imbriquée, explicitement optimiste ; ce n'est
    pas une quatrième validation indépendante. Aucun ensemble ni réglage ML.
    """
    reference=assert_foundation_reference(leases)
    folds={};labels={};recency=[]
    for year in YEARS:
        folds[year]=fold_predictions(leases,year)
        labels[year]=build_model_a_features(leases,year,include_targets=True,feature_set='core').labels
        lab=labels[year];mask=lab.transition_occurred.eq(1).to_numpy()
        for variant in ('equal','mild','moderate'):
            p=folds[year]['occurrence']['historical_global',variant]
            g=folds[year]['growth']['historical_mean',variant]
            m=occurrence_metrics(lab.transition_occurred,p)
            recency.append({'year':year,'variant':variant,'occurrence_brier':m['Brier'],
                'predicted_transition_rate_pct':100*m['predicted_rate'],'rate_error_pp':100*m['calibration_gap'],
                **growth_metrics(lab.conditional_growth.to_numpy()[mask],g[mask])})
    recency=pd.DataFrame(recency)
    weighted=recency.loc[recency.variant.ne('equal')].groupby('variant',sort=False).agg(
        brier=('occurrence_brier','mean'),growth_mae=('mae_pp','mean'))
    best_o=weighted.sort_values(['brier','variant'],kind='stable').index[0]
    best_g=weighted.sort_values(['growth_mae','variant'],kind='stable').index[0]
    configs={
        'global_mean':model_a_config(),
        'recency_best':model_a_config(recency_variant={'occurrence':best_o,'growth':best_g}),
        'global_median':model_a_config(growth_method='historical_median'),
        'logistic_mean':model_a_config(occurrence_method='Logistic'),
        'global_ridge':model_a_config(growth_method='Ridge'),
        'global_hgb':model_a_config(growth_method='HGB'),
        'logistic_hgb':model_a_config(occurrence_method='Logistic',growth_method='HGB'),
    }
    complexity={'global_mean':0,'global_median':0,'recency_best':1,'logistic_mean':2,'global_ridge':2,'global_hgb':3,'logistic_hgb':4}
    rows=[];component_metrics=[]
    for name,config in configs.items():
        for year in YEARS:
            op=(config['occurrence_method'],config['recency_variant']['occurrence'])
            gp=(config['growth_method'],config['recency_variant']['growth'])
            if op not in folds[year]['occurrence'] or gp not in folds[year]['growth']:
                continue
            result=predict_from_fold(folds[year],config)
            lab=labels[year];mask=lab.transition_occurred.eq(1).to_numpy()
            score=p1_diagnostic(result['probabilities'],result['conditional_growth'],lab)
            rows.append({'year':year,'candidate':name,**score,'absolute_error_pp':abs(score['error_pp'])})
            om=occurrence_metrics(lab.transition_occurred,result['probabilities'])
            gm=growth_metrics(lab.conditional_growth.to_numpy()[mask],result['conditional_growth'][mask])
            component_metrics.append({'candidate':name,'year':year,'occurrence_brier':om['Brier'],
                                      'growth_mae_pp':gm['mae_pp']})
    annual=pd.DataFrame(rows)
    baseline=annual.loc[annual.candidate.eq('global_mean')].set_index('year').absolute_error_pp
    summaries=[]
    for name,g in annual.groupby('candidate',sort=False):
        if tuple(sorted(g.year))!=YEARS:continue
        metrics=pd.DataFrame(component_metrics).loc[lambda x:x.candidate.eq(name)]
        errors=g.error_pp.to_numpy()
        summaries.append({'candidate':name,'MAE_P1':g.absolute_error_pp.mean(),
            'RMSE_P1':np.sqrt(np.mean(errors**2)),'bias_P1':errors.mean(),
            'worst_year_absolute_error':g.absolute_error_pp.max(),'max_to_min_error_spread':errors.max()-errors.min(),
            'occurrence_brier':metrics.occurrence_brier.mean(),'growth_mae_pp':metrics.growth_mae_pp.mean(),
            'complexity_level':complexity[name],'interpretability_level':complexity[name],
            'max_degradation_vs_baseline_pp':float((g.set_index('year').absolute_error_pp-baseline).max())})
    ranked=rank_candidates(pd.DataFrame(summaries))
    eligible=ranked.loc[ranked.eligible_for_promotion]
    champion=str(eligible.iloc[0].candidate)
    runner=str(eligible.iloc[1].candidate) if len(eligible)>1 else None
    report={'model_name':'MODEL_A_CHAMPION','candidate':champion,**configs[champion],
        'backtest':annual.loc[annual.candidate.eq(champion)].to_dict('records'),
        'aggregate_metrics':ranked.loc[ranked.candidate.eq(champion)].iloc[0].to_dict(),
        'runner_up':None if runner is None else {'candidate':runner,**configs[runner]},
        'risk_factors':['Dérive du taux de transition observée en 2025','Maturité et censure des labels','Fidélité du snapshot CRM non prouvée'],
        'limitations':['Trois années seulement','Sélection rétrospective non imbriquée ; scores optimistes après sélection',
                       'P1 est un indice événementiel, pas une croissance du rent roll','Aucun forecast 2026 calculé']}
    return {'reference':reference,'recency':recency,'annual':annual,'ranking':ranked,'configs':configs,
            'MODEL_A_CHAMPION':configs[champion],'MODEL_A_RUNNER_UP':None if runner is None else configs[runner],
            'explainability':report,'_folds':folds,'_labels':labels}
