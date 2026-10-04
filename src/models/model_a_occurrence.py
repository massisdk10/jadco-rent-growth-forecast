"""Occurrence Model A : CORE uniquement, réglage fixe et train historique."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from src.features.model_dataset import OCCURRENCE_FEATURES

CATEGORICAL = ('property_code', 'building', 'province', 'unit_subtype')


def _numeric(values):
    return pd.DataFrame(values).apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)


def _categorical(values):
    return pd.DataFrame(values).astype(object).where(pd.notna(values), np.nan)


def preprocessor(features, scale=True):
    """Imputation/encodage appris sur train ; colonnes vides conservées."""
    numeric = [f for f in features if f not in CATEGORICAL]
    num_steps = [('conversion', FunctionTransformer(_numeric)),
                 ('imputation', SimpleImputer(strategy='median', keep_empty_features=True))]
    if scale:
        num_steps.append(('standardisation', StandardScaler()))
    cat = Pipeline([('conversion', FunctionTransformer(_categorical)),
                    ('imputation', SimpleImputer(strategy='constant', fill_value='manquant', keep_empty_features=True)),
                    ('encodage', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    return ColumnTransformer([('num', Pipeline(num_steps), numeric),
                              ('cat', cat, list(CATEGORICAL))], sparse_threshold=0,
                             verbose_feature_names_out=False)


def validate_training(X_train, X_test, year, features):
    """Refuser 2026, les années futures et tout schéma hors CORE."""
    if year not in (2023, 2024, 2025):
        raise ValueError('Entraînement limité aux backtests 2023–2025.')
    if tuple(X_train.columns) != tuple(features) or tuple(X_test.columns) != tuple(features):
        raise ValueError('Schéma CORE exact requis ; labels et extension exclus.')
    if X_train.empty or not X_train.forecast_year.lt(year).all():
        raise ValueError('Historique train vide ou non antérieur au test.')
    if not X_test.forecast_year.eq(year).all():
        raise ValueError('Année test incohérente.')


def predict_occurrence_models(X_train, labels, X_test, year):
    """Taux global qualifié et Logistic L2 C=1, sans tuning ni recalibration."""
    validate_training(X_train, X_test, year, OCCURRENCE_FEATURES)
    y = np.asarray(labels, dtype=float)
    if y.shape != (len(X_train),) or not np.isin(y, [0, 1]).all():
        raise ValueError('Labels historiques qualifiés 0/1 requis.')
    predictions = {'historical_global': np.full(len(X_test), y.mean())}
    fitted = {}
    # Deux classes et au moins dix observations de chacune : seuil préalable.
    if min(np.sum(y == 0), np.sum(y == 1)) >= 10:
        model = Pipeline([('preprocessing', preprocessor(OCCURRENCE_FEATURES)),
                          ('model', LogisticRegression(C=1.0, penalty='l2', class_weight=None,
                                                       max_iter=2000, random_state=42))])
        model.fit(X_train, y)
        predictions['Logistic'] = model.predict_proba(X_test)[:, 1]
        fitted['Logistic'] = model
    return predictions, fitted
