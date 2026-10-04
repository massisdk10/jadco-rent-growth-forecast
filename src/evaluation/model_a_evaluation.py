"""Walk-forward Model A, résultats agrégés ; aucune prévision 2026."""
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.inspection import permutation_importance
from threadpoolctl import threadpool_limits
from src.evaluation.backtest import occurrence_history, occurrence_metrics, aggregate_expected_contribution
from src.features.model_a_features import build_model_a_features, assert_foundation_reference
from src.features.model_dataset import OCCURRENCE_FEATURES, ADMISSIBLE_FEATURES
from src.models.model_a_occurrence import predict_occurrence_models
from src.models.model_a_growth import predict_growth_models


def temporal_training(leases, year):
    """Fondation : cohortes à leur origine, labels mûrs à l'origine du test.

    Aucune paire future n'entre dans occurrence_history : son snapshot est
    préalablement limité à D. Les labels en attente ne deviennent pas zéro.
    Le G historique suit la moyenne des transitions admissibles alors connues,
    comme la baseline P1 figée ; sa fidélité CRM reste conditionnelle.
    """
    if year not in (2023, 2024, 2025):
        raise ValueError('Aucune construction de train/tuning 2026.')
    history = occurrence_history(leases, year)
    qualified = history.dropna(subset=['occurrence_target']).reset_index(drop=True)
    positive = qualified.loc[qualified.occurrence_target.eq(1) & np.isfinite(qualified.unit_growth)].reset_index(drop=True)
    return qualified, positive, int(history.occurrence_target.isna().sum())


def growth_metrics(observed, predicted):
    y = np.asarray(observed, dtype=float); p = np.asarray(predicted, dtype=float)
    if y.shape != p.shape or y.ndim != 1 or not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Croissances/prédictions finies et alignées requises.')
    e = 100*(p-y)
    return {'mae_pp': float(np.abs(e).mean()), 'rmse_pp': float(np.sqrt(np.mean(e**2))),
            'bias_pp': float(e.mean()), 'median_absolute_error_pp':float(np.median(np.abs(e))),
            'median_prediction':float(100*np.median(p)), 'median_realized':float(100*np.median(y))}


def p1_diagnostic(probabilities, growth, labels):
    """Toutes les candidates, contribution réalisée zéro sans événement."""
    if not set(['transition_occurred','conditional_growth']).issubset(labels):
        raise ValueError('Labels séparés requis.')
    o = labels.transition_occurred.to_numpy(dtype=float)
    g = labels.conditional_growth.to_numpy(dtype=float)
    if not np.isin(o,[0,1]).all() or not np.isfinite(g[o==1]).all():
        raise ValueError('Labels réalisés incomplets.')
    if len(probabilities)!=len(labels):
        raise ValueError('Une contribution par candidate requise.')
    realized = 100*float(np.median(np.where(o==1,g,0)))
    predicted = 100*aggregate_expected_contribution(probabilities,growth)
    return {'predicted_P1_pct':predicted,'realized_P1_pct':realized,'error_pp':predicted-realized}


def coefficient_summary(model, features, year, name):
    """Coefficients standardisés/one-hot ; aucune affirmation causale."""
    prep = model.named_steps['preprocessing']
    names = [*prep.transformers_[0][2], *prep.named_transformers_['cat'].named_steps['encodage'].get_feature_names_out(prep.transformers_[1][2])]
    coef = np.ravel(model.named_steps['model'].coef_)
    return pd.DataFrame({'year':year,'model':name,'feature':names,'coefficient':coef})


def evaluate_model_a(leases):
    """Réglages fixes préétablis ; labels test consultés après prédiction."""
    reference = assert_foundation_reference(leases)
    tables = {k:[] for k in ['training','occurrence','growth','p1','calibration','probabilities','coefficients','importance','extremes','segments']}
    for year in (2023,2024,2025):
        train, positive, pending = temporal_training(leases,year)
        candidate = build_model_a_features(leases,year,feature_set='core')
        occurrence, om = predict_occurrence_models(train[list(OCCURRENCE_FEATURES)],train.occurrence_target,candidate.X,year)
        growth, gm = predict_growth_models(positive[list(ADMISSIBLE_FEATURES)],positive.unit_growth,
                                          positive.occurrence_target,candidate.X_growth,year)
        # Révélation strictement après fixation des prédictions du fold.
        labels = build_model_a_features(leases,year,include_targets=True,feature_set='core').labels
        mask = labels.transition_occurred.eq(1).to_numpy()
        tables['training'].append({'year':year,'n_train_occurrence':len(train),
            'non_transitions':int(train.occurrence_target.eq(0).sum()),'transitions':int(train.occurrence_target.eq(1).sum()),
            'pending':pending,'n_train_growth':len(positive),'n_test':len(candidate.X),'n_test_growth':int(mask.sum())})
        for name,p in occurrence.items():
            m = occurrence_metrics(labels.transition_occurred,p)
            tables['occurrence'].append({'year':year,'model':name,'n_train':len(train),'n_test':len(p),
                'predicted_transition_rate':100*m['predicted_rate'],'realized_transition_rate':100*m['observed_rate'],
                'brier':m['Brier'],'rate_error':100*m['calibration_gap'],
                'log_loss':float(log_loss(labels.transition_occurred,np.clip(p,1e-12,1-1e-12),labels=[0,1])),
                'notes':'réglage fixe ; aucune calibration supplémentaire'})
            tables['probabilities'].append({'year':year,'model':name,'minimum':p.min(),'q05':np.quantile(p,.05),
                'median':np.median(p),'q95':np.quantile(p,.95),'maximum':p.max(),'extreme_rate_pct':100*np.mean((p<.01)|(p>.99))})
            bins = np.minimum((p*10).astype(int),9)
            for b in sorted(set(bins)):
                sel = bins==b; n = int(sel.sum())
                tables['calibration'].append({'year':year,'model':name,'probability_bin':int(b),'support':n,
                    'predicted_rate':float(p[sel].mean()) if n>=5 else np.nan,
                    'realized_rate':float(labels.transition_occurred.to_numpy()[sel].mean()) if n>=5 else np.nan})
            for building in candidate.X.building.unique():
                sel = candidate.X.building.eq(building).to_numpy(); n = int(sel.sum())
                tables['segments'].append({'year':year,'model':name,'building':building,'support':n,
                    'predicted_rate_pct':100*p[sel].mean() if n>=5 else np.nan,
                    'realized_rate_pct':100*labels.transition_occurred.to_numpy()[sel].mean() if n>=5 else np.nan})
        for name,g in growth.items():
            tables['growth'].append({'year':year,'model':name,'n_train':len(positive),'n_test':int(mask.sum()),
                **growth_metrics(labels.conditional_growth.to_numpy()[mask],g[mask])})
            observed = labels.conditional_growth.to_numpy()[mask]
            # Extrêmes définis exclusivement sur train, aucun retrait.
            q1,q3=positive.unit_growth.quantile([.25,.75]);iqr=q3-q1
            extreme=(observed<q1-3*iqr)|(observed>q3+3*iqr)
            tables['extremes'].append({'year':year,'model':name,'support':int(extreme.sum()),
                'mae_pp':100*float(np.abs(g[mask][extreme]-observed[extreme]).mean()) if extreme.sum()>=5 else np.nan})
        for name,model in {**om,**gm}.items():
            if name!='HGB':
                features=OCCURRENCE_FEATURES if name=='Logistic' else ADMISSIBLE_FEATURES
                tables['coefficients'].append(coefficient_summary(model,features,year,name))
        if 'HGB' in gm and mask.sum()>=30:
            with threadpool_limits(limits=1):
                imp=permutation_importance(gm['HGB'],candidate.X_growth.loc[mask],labels.conditional_growth.loc[mask],
                    scoring='neg_mean_absolute_error',n_repeats=3,random_state=42,n_jobs=1)
            tables['importance'].append(pd.DataFrame({'year':year,'feature':ADMISSIBLE_FEATURES,
                'increase_mae_pp':100*imp.importances_mean,'std_pp':100*imp.importances_std}))
        for oname,p in occurrence.items():
            for gname,g in growth.items():
                tables['p1'].append({'year':year,'occurrence_model':oname,'growth_model':gname,
                                     **p1_diagnostic(p,g,labels)})
    result={k:pd.concat(v,ignore_index=True) if v and isinstance(v[0],pd.DataFrame) else pd.DataFrame(v) for k,v in tables.items()}
    result['reference']=reference
    result['occurrence_summary']=result['occurrence'].groupby('model',sort=False).agg(
        folds=('year','count'),mean_brier=('brier','mean'),mean_abs_rate_error=('rate_error',lambda x:x.abs().mean()),
        worst_brier=('brier','max')).reset_index()
    result['growth_summary']=result['growth'].groupby('model',sort=False).agg(
        folds=('year','count'),mean_mae_pp=('mae_pp','mean'),mean_rmse_pp=('rmse_pp','mean'),
        mean_abs_bias_pp=('bias_pp',lambda x:x.abs().mean()),worst_year_mae_pp=('mae_pp','max'),
        stability_mae_std_pp=('mae_pp',lambda x:x.std(ddof=0))).reset_index()
    return result
