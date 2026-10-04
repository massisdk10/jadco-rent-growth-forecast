"""Walk-forward annuel et baselines transparentes, sans modèle particulier."""
import numpy as np
import pandas as pd
from src.features.model_dataset import (MAIN_TARGET, prediction_origin, build_features,
    validate_features, aggregate_growth, available_external_context, build_prediction_cohort, build_model_dataset, known_leases, OCCURRENCE_FEATURES)

BACKTEST_YEARS = (2023, 2024, 2025)


def split_year(data, year, target=MAIN_TARGET):
    """Train : labels antérieurs et mûrs à la coupure ; test : débuts en Y.

    Test ex post conditionnel : sa sélection utilise les transitions révélées.
    Ne jamais utiliser ces lignes pour constituer X à prédire au cutoff.
    Une fin/signature manquante exclut le label de tout entraînement.
    """
    origin = prediction_origin(year)
    available = pd.to_datetime(data.label_available_date, errors='coerce', format='mixed')
    valid = data.eligible_main & np.isfinite(data[target])
    train = data.loc[valid & data.year.lt(year) & data.new_lease_date.le(origin) & available.le(origin)].copy()
    test = data.loc[valid & data.year.eq(year)].copy()
    return train, test


def baseline_previous_year(train, year, target=MAIN_TARGET):
    """Médiane des labels admissibles de Y−1 ; pas de repli silencieux."""
    return float(train.loc[train.year.eq(year-1),target].median())


def baseline_recent_history(train, year, target=MAIN_TARGET):
    """Médiane des labels admissibles de Y−3 à Y−1 inclus."""
    return float(train.loc[train.year.between(year-3,year-1),target].median())


def evaluate_predictions(test, predictions, target=MAIN_TARGET):
    """Erreurs en points de pourcentage ; biais = prédit − observé."""
    p = np.asarray(predictions, dtype=float)
    if p.shape != (len(test),):
        raise ValueError('Une prédiction par transition test est nécessaire.')
    observed = test[target].to_numpy(dtype=float)
    if not np.isfinite(p).all() or not np.isfinite(observed).all() or len(test)==0:
        raise ValueError('Prédictions/observations non finies ou test vide.')
    errors = 100*(p-observed)
    pred_agg = aggregate_growth(test,p)['growth']
    actual_agg = aggregate_growth(test,observed)['growth']
    return {'test_transitions':len(test), 'MAE_pp':float(np.mean(np.abs(errors))),
            'RMSE_pp':float(np.sqrt(np.mean(errors**2))), 'bias_pp':float(np.mean(errors)),
            'predicted_growth_pct':100*pred_agg, 'observed_growth_pct':100*actual_agg,
            'portfolio_error_pp':100*(pred_agg-actual_agg),
            'portfolio_absolute_error_pp':100*abs(pred_agg-actual_agg)}


def run_backtest(model, data, leases, year, features=(), external=None, target=MAIN_TARGET):
    """Interface callable simple, sans livrer au modèle les lignes test privées.

    model(train, X_candidates, year, external_context) -> une fraction par unité candidate.
    La cohorte est fixée au cutoff ; les transitions sont révélées après appel.
    Les candidats sans transition ne reçoivent pas un label nul. Plusieurs
    transitions observées d’une unité reçoivent la même prédiction conditionnelle.
    train contient exclusivement features demandées et target. X_train est
    construit à chaque origine historique de son année, jamais à l'origine Y.
    Une implémentation future doit ajuster ses transformations sur train seul.
    """
    validate_features(features)
    historical, _ = build_model_dataset(known_leases(leases,prediction_origin(year)))
    train, _ = split_year(historical,year,target)
    parts = []
    for historical_year, part in train.groupby('year', sort=True):
        x = build_features(leases,part,int(historical_year),features)
        x[target] = part[target].to_numpy()
        parts.append(x)
    model_train = pd.concat(parts) if parts else pd.DataFrame(columns=[*features,target])
    cohort = build_prediction_cohort(leases,year)
    x_test = build_features(leases,cohort,year,features)
    context = available_external_context(external,year) if external is not None else None
    predictions = model(model_train.copy(),x_test.copy(),int(year),context)
    predictions = np.asarray(predictions, dtype=float)
    if predictions.shape != (len(cohort),) or not np.isfinite(predictions).all():
        raise ValueError('Une prédiction finie par unité candidate est nécessaire.')
    predicted = cohort[['sPropCode','sUnitCode']].copy()
    predicted['_prediction'] = predictions
    # Le futur n'intervient qu'après la prédiction, pour révéler les labels.
    _, test = split_year(data,year,target)
    observed = test.merge(predicted, on=['sPropCode','sUnitCode'], how='inner', validate='many_to_one')
    result = evaluate_predictions(observed,observed['_prediction'],target) if len(observed) else {'test_transitions':0}
    result.update(candidate_units=len(cohort),
                  realized_candidate_units=len(observed[['sPropCode','sUnitCode']].drop_duplicates()),
                  all_realized_units=len(test[['sPropCode','sUnitCode']].drop_duplicates()),
                  evaluation_scope='conditional_growth_on_cutoff_cohort')
    return result


def run_baselines(data, years=BACKTEST_YEARS, target=MAIN_TARGET):
    """Évaluer seulement les deux références autorisées, sans tuning."""
    rows=[]
    for year in years:
        train,test=split_year(data,year,target)
        for name,fn in [('mediane_annee_precedente',baseline_previous_year),
                        ('mediane_trois_ans',baseline_recent_history)]:
            value=fn(train,year,target)
            row={'year':year,'baseline':name,'train_transitions':len(train),
                 'prediction_origin':prediction_origin(year).date().isoformat()}
            if np.isfinite(value) and len(test):
                row.update(evaluate_predictions(test,np.full(len(test),value),target))
                row['statut']='calculable'
            else:
                row['statut']='historique admissible insuffisant ; aucune imputation'
            rows.append(row)
    return pd.DataFrame(rows)


def occurrence_history(leases, year):
    """Historique étiqueté strictement avant Y, avec labels mûrs seulement."""
    from src.features.model_dataset import build_occurrence_features, reveal_occurrence
    origin = prediction_origin(year)
    past = known_leases(leases,origin)
    pairs,_ = build_model_dataset(past)
    rows=[]
    for historical_year in range(int(past._start.dt.year.min()) if len(past) else year,year):
        cohort=build_prediction_cohort(past,historical_year)
        if cohort.empty:continue
        labels=reveal_occurrence(cohort,pairs,historical_year,label_cutoff=origin)
        x=build_occurrence_features(past,cohort,historical_year)
        x['occurrence_target']=labels.occurrence_target.to_numpy()
        x['unit_growth']=labels.unit_growth.to_numpy()
        rows.append(x)
    return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame(columns=['province','occurrence_target','unit_growth'])


def predict_occurrence(history, features, method='historical_global', min_segment=30):
    """Trois baselines, sans accès aux labels de l'année à prévoir.

    Seuil de segment fixé à 30 avant évaluation ; repli global explicite.
    Pas d'historique : NaN, sans fabrication de probabilité.
    """
    allowed=set(OCCURRENCE_FEATURES)
    if set(features)-allowed:raise ValueError('Feature d’occurrence interdite.')
    y=history.occurrence_target.dropna()
    if method=='all_transition':return np.ones(len(features))
    global_rate=float(y.mean())
    if method=='historical_global':return np.full(len(features),global_rate)
    if method!='historical_province':raise ValueError('Baseline d’occurrence inconnue.')
    stats=history.dropna(subset=['occurrence_target']).groupby('province').occurrence_target.agg(['count','mean'])
    reliable=stats.loc[stats['count'].ge(min_segment),'mean']
    return features.province.map(reliable).fillna(global_rate).to_numpy(dtype=float)


def occurrence_metrics(labels, probabilities):
    """Calibration globale descriptive ; accuracy seule insuffisante."""
    y=np.asarray(labels,dtype=float);p=np.asarray(probabilities,dtype=float)
    if y.shape!=p.shape or not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all() or not len(y):
        raise ValueError('Labels/probabilités d’occurrence invalides.')
    predicted=p>=.5
    tp=int((predicted & (y==1)).sum());fp=int((predicted & (y==0)).sum());fn=int((~predicted & (y==1)).sum())
    return {'accuracy':float((predicted==y).mean()),'precision':tp/(tp+fp) if tp+fp else np.nan,
            'recall':tp/(tp+fn) if tp+fn else np.nan,'Brier':float(np.mean((p-y)**2)),
            'predicted_rate':float(p.mean()),'observed_rate':float(y.mean()),
            'calibration_gap':float(p.mean()-y.mean())}


def aggregate_expected_contribution(probabilities, conditional_growth):
    """P1 : médiane de p × E[G|transition], pas E[médiane] ni rent roll."""
    p=np.asarray(probabilities,dtype=float);g=np.asarray(conditional_growth,dtype=float)
    if p.shape!=g.shape or p.ndim!=1 or not len(p) or not np.isfinite(g).all() or not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all():
        raise ValueError('Probabilités/croissances invalides.')
    return float(np.median(p*g))


def run_portfolio_baselines(leases, year):
    """P1 sur cohorte entière : prédire d'abord, révéler et scorer ensuite.

    G = moyenne des transitions admissibles par unité. Sa moyenne historique
    conditionnelle est cohérente avec E[O G]. Zéro sans événement ne décrit
    aucun mécanisme de loyer. Aucun calcul de forecast final 2026 ici.
    """
    from src.features.model_dataset import build_occurrence_features, reveal_occurrence
    cohort=build_prediction_cohort(leases,year)
    x=build_occurrence_features(leases,cohort,year)
    history=occurrence_history(leases,year)
    # G est une moyenne conditionnelle ; ne pas multiplier une médiane par p
    # et l'appeler espérance. La médiane historique reste un comparateur.
    growth=float(history.loc[history.occurrence_target.eq(1),'unit_growth'].mean())
    predictions=[]
    for method in ['all_transition','historical_global','historical_province']:
        p=predict_occurrence(history,x,method)
        predictions.append((method,p,aggregate_expected_contribution(p,np.full(len(x),growth))))
    # Accès à Y seulement après fixation de toutes les prédictions.
    revealed,_=build_model_dataset(known_leases(leases,pd.Timestamp(int(year),12,31)))
    labels=reveal_occurrence(cohort,revealed,year)
    observed_contribution=labels.unit_growth.where(labels.occurrence_target.eq(1),0).to_numpy()
    actual=float(np.median(observed_contribution))
    past_pairs,_=build_model_dataset(known_leases(leases,prediction_origin(year)))
    train,_=split_year(past_pairs,year)
    conditional_baseline=baseline_previous_year(train,year)
    rows=[]
    for method,p,predicted in predictions:
        row={'year':year,'baseline':method,'candidate_units':len(cohort),
             'realized_units':int(labels.occurrence_target.sum()),
             'history_qualified':int(history.occurrence_target.notna().sum()),
             'history_pending':int(history.occurrence_target.isna().sum()),
             'conditional_mean_pct':100*growth,'predicted_P1_pct':100*predicted,
             'observed_P1_pct':100*actual,'P1_error_pp':100*(predicted-actual),
             'conditional_baseline_P1_error_pp':100*(conditional_baseline-actual)}
        row.update(occurrence_metrics(labels.occurrence_target,p));rows.append(row)
    return pd.DataFrame(rows)
