"""Assemblage P1 CORE, expériences historiques et API finale du champion figé."""
import numpy as np
import pandas as pd
from src.features.model_dataset import prediction_origin, OCCURRENCE_FEATURES, ADMISSIBLE_FEATURES
from src.features.model_a_features import build_model_a_features
from src.evaluation.model_a_evaluation import temporal_training
from src.models.model_a_occurrence import predict_occurrence_models
from src.models.model_a_growth import predict_growth_models
from src.evaluation.backtest import aggregate_expected_contribution

HALF_LIVES = {'equal': None, 'mild': 3.0, 'moderate': 1.5}


def recency_weights(observation_years, origin, variant='equal'):
    """Âge en années de 365,25 jours depuis la fin de l'année observée.

    Proxy annuel déterministe, pas date de maturité du label. Le caller doit
    fournir seulement les labels qualifiés par temporal_training à l'origine.
    Refuser une année inachevée/future, même lorsque lambda vaut zéro.
    """
    if variant not in HALF_LIVES:
        raise ValueError('Variante de récence inconnue.')
    years = pd.Series(observation_years).reset_index(drop=True)
    numeric = pd.to_numeric(years, errors='coerce')
    if numeric.isna().any() or not np.isfinite(numeric).all() or not numeric.eq(numeric.astype(int)).all():
        raise ValueError('Années historiques entières requises.')
    dates = pd.to_datetime(numeric.astype(int).astype(str)+'-12-31',errors='coerce')
    origin = pd.Timestamp(origin)
    if pd.isna(origin) or dates.isna().any() or dates.gt(origin).any():
        raise ValueError('Observation future ou date invalide.')
    age = (origin-dates).dt.total_seconds().to_numpy()/86400/365.25
    half_life = HALF_LIVES[variant]
    rate = 0.0 if half_life is None else np.log(2)/half_life
    return np.exp(-rate*age)


def model_a_config(occurrence_method='historical_global', growth_method='historical_mean',
                   recency_variant='equal', feature_set='core'):
    """Configuration JSON explicite ; médiane et ML restent non pondérés."""
    if feature_set!='core':
        raise ValueError('CORE uniquement.')
    if occurrence_method not in ('historical_global','Logistic') or growth_method not in ('historical_mean','historical_median','Ridge','ElasticNet','HGB'):
        raise ValueError('Méthode Model A inconnue.')
    recency = {'occurrence':recency_variant,'growth':recency_variant} if isinstance(recency_variant,str) else dict(recency_variant)
    if set(recency)!= {'occurrence','growth'} or any(v not in HALF_LIVES for v in recency.values()):
        raise ValueError('Récence explicite occurrence/growth requise.')
    if occurrence_method!='historical_global' and recency['occurrence']!='equal':
        raise ValueError('Récence Logistic non expérimentée.')
    if growth_method!='historical_mean' and recency['growth']!='equal':
        raise ValueError('Récence réservée à la moyenne historique.')
    return {'occurrence_method':occurrence_method,'growth_method':growth_method,
            'recency_variant':recency,'feature_set':'core'}


def assemble_p1(probabilities, conditional_growth):
    """Fractions en entrée ; contributions uniquement en mémoire, jamais exportées."""
    p=np.asarray(probabilities,dtype=float);g=np.asarray(conditional_growth,dtype=float)
    value=aggregate_expected_contribution(p,g)
    return {'probabilities':p.copy(),'conditional_growth':g.copy(),'contributions':p*g,
            'p1_fraction':value,'p1_pct':100*value}


def fold_predictions(leases, year):
    """Fixer toutes les prédictions à D avant toute révélation des labels test."""
    if year not in (2023,2024,2025):
        raise ValueError('Aucun forecast 2026 autorisé à cette phase.')
    train,positive,pending=temporal_training(leases,year)
    candidate=build_model_a_features(leases,year,feature_set='core')
    occurrence,_=predict_occurrence_models(train[list(OCCURRENCE_FEATURES)],train.occurrence_target,candidate.X,year)
    growth,_=predict_growth_models(positive[list(ADMISSIBLE_FEATURES)],positive.unit_growth,
                                  positive.occurrence_target,candidate.X_growth,year)
    op={(method,'equal'):p for method,p in occurrence.items()}
    gp={(method,'equal'):g for method,g in growth.items()}
    for variant in ('mild','moderate'):
        w=recency_weights(train.forecast_year,prediction_origin(year),variant)
        wg=recency_weights(positive.forecast_year,prediction_origin(year),variant)
        op['historical_global',variant]=np.full(len(candidate.X),np.average(train.occurrence_target,weights=w))
        gp['historical_mean',variant]=np.full(len(candidate.X),np.average(positive.unit_growth,weights=wg))
    return {'candidate':candidate,'occurrence':op,'growth':gp,
            'n_train_occurrence':len(train),'n_train_growth':len(positive),'pending':pending}


def predict_from_fold(fold, config):
    """Assembler une configuration validée depuis des prédictions déjà figées."""
    config=model_a_config(**config)
    op=(config['occurrence_method'],config['recency_variant']['occurrence'])
    gp=(config['growth_method'],config['recency_variant']['growth'])
    if op not in fold['occurrence'] or gp not in fold['growth']:
        raise ValueError('Historique insuffisant pour cette configuration ; aucun fallback caché.')
    return assemble_p1(fold['occurrence'][op],fold['growth'][gp])


def predict_model_a(leases, year, occurrence_method='historical_global', growth_method='historical_mean',
                    recency_variant='equal', feature_set='core'):
    """Interface configurable ; 2026 reste explicitement interdit."""
    config=model_a_config(occurrence_method,growth_method,recency_variant,feature_set)
    return predict_from_fold(fold_predictions(leases,year),config)


# Configuration approuvée avant tout calcul 2026 ; aucun sélecteur dynamique.
def champion_config():
    """Retourner une copie de la configuration humaine figée."""
    return model_a_config('historical_global','historical_median','equal','core')


def _champion_components(leases, year):
    """Même historique qualifié que A3–A10 ; aucun entraînement ML requis."""
    from src.evaluation.backtest import occurrence_history
    from src.features.model_dataset import known_leases
    if year not in (2023,2024,2025,2026):
        raise ValueError('Année hors périmètre Model A.')
    past=known_leases(leases,prediction_origin(year))
    candidate=build_model_a_features(past,year,feature_set='core')
    history=occurrence_history(past,year)
    qualified=history.dropna(subset=['occurrence_target']).copy()
    growth=pd.to_numeric(qualified.unit_growth,errors='coerce')
    positive=qualified.loc[qualified.occurrence_target.eq(1) & np.isfinite(growth)].copy()
    if qualified.empty or positive.empty or candidate.X.empty:
        raise ValueError('Historique admissible ou cohorte insuffisant ; aucun repli.')
    if not qualified.forecast_year.lt(year).all():
        raise AssertionError('Label futur dans historique ; arrêt.')
    p=float(qualified.occurrence_target.mean());g=float(positive.unit_growth.median())
    result=assemble_p1(np.full(len(candidate.X),p),np.full(len(candidate.X),g))
    return candidate,qualified,positive,int(history.occurrence_target.isna().sum()),result


def backtest(leases, target_year):
    """Interface starter : champion figé, prédire puis révéler les labels de Y."""
    from src.evaluation.model_a_evaluation import p1_diagnostic
    from src.evaluation.backtest import occurrence_metrics
    if target_year not in (2023,2024,2025):
        raise ValueError('Backtests exclusivement 2023/2024/2025 ; aucun label 2026.')
    candidate,history,positive,pending,result=_champion_components(leases,target_year)
    labels=build_model_a_features(leases,target_year,include_targets=True,feature_set='core').labels
    score=p1_diagnostic(result['probabilities'],result['conditional_growth'],labels)
    metrics=occurrence_metrics(labels.transition_occurred,result['probabilities'])
    return {'year':int(target_year),'prediction_origin':str(prediction_origin(target_year).date()),
        **score,'absolute_error_pp':abs(score['error_pp']),'candidate_units':len(candidate.X),
        'observed_units':int(labels.transition_occurred.sum()),'n_train_occurrence':len(history),
        'n_train_growth':len(positive),'pending_labels':pending,'occurrence_brier':metrics['Brier'],
        'configuration':champion_config()}


def _cohort_summary(candidate):
    """Agrégats uniquement ; normalisation d'étiquettes sans modifier la cohorte."""
    import unicodedata
    def label(value):
        return ''.join(c for c in unicodedata.normalize('NFD',str(value)) if unicodedata.category(c)!='Mn')
    x=candidate.X
    buildings=x.building.map(label).value_counts().to_dict()
    province=x.province.map(label).value_counts().to_dict()
    expiry={str(m):int(x.expiry_month.eq(m).sum()) for m in range(1,13)}
    expected_buildings={'Daniel-Johnson':128,'Le Carlyle':180,'Levesque':75,'Saint-Elzear':268,'The Met':121,'Westpark':159}
    expected_expiry=[38,52,65,62,91,143,90,91,108,61,69,61]
    if len(x)!=931 or province!={'Quebec':810,'Ontario':121} or buildings!=expected_buildings or list(expiry.values())!=expected_expiry:
        raise AssertionError('Écart de cohorte 2026 : arrêt ; aucune correction silencieuse.')
    if candidate.labels is not None:
        raise AssertionError('Label 2026 interdit.')
    bedrooms=x.bedrooms.value_counts(dropna=False)
    return {'n_units':len(x),'province':{k:int(v) for k,v in province.items()},
        'building':{k:int(v) for k,v in buildings.items()},'expiry_month':expiry,
        'bedrooms':{str(k):int(v) for k,v in bedrooms.items() if v>=5},
        'bedrooms_small_groups_units':int(bedrooms.loc[bedrooms.lt(5)].sum()),
        'segment_interpretation':'Modèle au niveau portefeuille : les segments décrivent la composition et ne modifient pas la prédiction.'}


def estimate_2026(leases, asking, external=None, *, return_details=False):
    """API starter : forecast en pourcentage, ou rapport agrégé sur demande.

    asking/external conservés pour compatibilité ; aucune utilisation prédictive
    faute de qualification. Champion et méthode d'incertitude figés avant 2026.
    """
    from src.evaluation.model_a_uncertainty import uncertainty_scenarios
    from src.explainability.model_a_explainability import build_explanation
    candidate,history,positive,pending,result=_champion_components(leases,2026)
    cohort=_cohort_summary(candidate)
    # Champion calculé avant la sensibilité ; celle-ci ne le remplace jamais.
    p=float(result['probabilities'][0]);g=float(result['conditional_growth'][0])
    runner=assemble_p1(np.full(len(candidate.X),p),np.full(len(candidate.X),float(positive.unit_growth.mean())))
    annual=[backtest(leases,y) for y in (2023,2024,2025)]
    errors=pd.DataFrame(annual)
    scenarios=uncertainty_scenarios(result['probabilities'],result['conditional_growth'],errors,1000,42)
    if not scenarios['LOW']<=scenarios['BASE']<=scenarios['HIGH'] or g<=-1:
        raise AssertionError('Sanity check numérique invalide ; aucun retuning.')
    signed=errors.error_pp.to_numpy(dtype=float)
    payload={'method_name':'Model A — Hybrid Transition Intelligence','forecast_year':2026,
        'prediction_origin':'2025-12-31','target_definition':'P1_median_transition_contribution_effective_annualized',
        'target_name':'Indice médian de contribution attendue des transitions à la croissance annualisée du loyer effectif des unités à échéance.',
        'predicted_growth':result['p1_pct'],'forecast':{'low':scenarios['LOW'],'base':scenarios['BASE'],'high':scenarios['HIGH'],'unit':'pourcentage'},
        'cohort':cohort,'architecture':champion_config(),'features_used':[],
        'components':{'historical_transition_probability':p,'historical_conditional_growth':g,
            'contribution_fraction':p*g,'unit':'fractions','constant_across_units':True,
            'n_history_occurrence':len(history),'n_transitions':int(history.occurrence_target.eq(1).sum()),
            'n_non_transitions':int(history.occurrence_target.eq(0).sum()),'pending_labels':pending,
            'n_history_growth':len(positive),'growth_q05_pct':100*float(positive.unit_growth.quantile(.05)),
            'growth_q95_pct':100*float(positive.unit_growth.quantile(.95)),
            'history_year_min':int(history.forecast_year.min()),'history_year_max':int(history.forecast_year.max()),
            'qualification':'Baux commencés et signés à D ; labels mûrs selon début/signature/fin ; années historiques antérieures à 2026.'},
        'backtest_by_year':{str(row['year']):row for row in annual},
        'aggregate_metrics':{'mae_pp':float(np.abs(signed).mean()),'rmse_pp':float(np.sqrt(np.mean(signed**2))),
            'bias_pp':float(signed.mean()),'worst_year_pp':float(np.abs(signed).max())},
        'runner_up_sensitivity':{'forecast_pct':runner['p1_pct'],'difference_pp':runner['p1_pct']-result['p1_pct'],
            'configuration':model_a_config('historical_global','historical_mean','equal','core'),
            'interpretation':'Sensibilité moyenne versus médiane ; ne remplace pas le champion.'},
        'uncertainty':{**scenarios,'bootstrap_width_pp':scenarios['bootstrap_q90_pct']-scenarios['bootstrap_q10_pct'],
            'method':'LOW=q10−E ; BASE=P1 ; HIGH=q90+E ; E=max des erreurs absolues 2023/2024/2025',
            'not_captured':['Dérive future','Changements réglementaires','Dépendance entre unités','Erreur structurelle nouvelle','Révisions CRM'],
            'bootstrap_explanation':"Le bootstrap transversal n'ajoute pas de largeur car le champion applique une contribution identique à chaque unité ; l'incertitude présentée provient donc de l'erreur historique de backtest."}}
    report=build_explanation(payload)
    return report if return_details else report['predicted_growth']
