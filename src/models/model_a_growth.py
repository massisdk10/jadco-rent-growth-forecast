"""Croissance conditionnelle CORE : baselines et trois réglages fixes."""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from src.features.model_dataset import ADMISSIBLE_FEATURES
from src.models.model_a_occurrence import preprocessor, validate_training


def predict_growth_models(X_train, growth, occurrence, X_test, year):
    """Apprendre uniquement sur les transitions qualifiées ; fractions en sortie."""
    validate_training(X_train, X_test, year, ADMISSIBLE_FEATURES)
    y = np.asarray(growth, dtype=float)
    o = np.asarray(occurrence, dtype=float)
    if y.shape != (len(X_train),) or o.shape != y.shape or not np.isfinite(y).all() or not (o == 1).all():
        raise ValueError('Croissances finies de transitions positives exclusivement.')
    predictions = {
        'historical_mean': np.full(len(X_test), y.mean()),
        'historical_median': np.full(len(X_test), np.median(y)),
    }
    previous = y[X_train.forecast_year.eq(year-1).to_numpy()]
    if len(previous):
        predictions['previous_year_median'] = np.full(len(X_test), np.median(previous))
    fitted = {}
    if len(y) < 30:
        return predictions, fitted
    estimators = {
        'Ridge': Ridge(alpha=10.0),
        'ElasticNet': ElasticNet(alpha=.001, l1_ratio=.1, max_iter=20000, random_state=42),
        'HGB': HistGradientBoostingRegressor(learning_rate=.05, max_iter=100,
               max_leaf_nodes=7, min_samples_leaf=20, l2_regularization=1.0,
               early_stopping=False, random_state=42),
    }
    # Limiter le parallélisme évite un coût disproportionné au petit dataset.
    with threadpool_limits(limits=1):
        for name, estimator in estimators.items():
            model = Pipeline([('preprocessing', preprocessor(ADMISSIBLE_FEATURES, scale=name!='HGB')),
                              ('model', estimator)])
            model.fit(X_train, y)
            predictions[name] = model.predict(X_test)
            fitted[name] = model
    return predictions, fitted
