"""Experts A/B immuables et confiance contextuelle, sans cible réalisée de Y."""
from dataclasses import asdict, dataclass
import hashlib
import json
import numpy as np
import pandas as pd

from src.evaluation.backtest import split_year, occurrence_history
from src.features.model_dataset import MAIN_TARGET, known_leases, prediction_origin, build_model_dataset
from src.models.model_a import _champion_components
from src.models.model_b.experiments import P1_OCCURRENCE_CONFIG, P1_GROWTH_CONFIG, _p1_growth_training
from src.models.model_b.occurrence import HierarchicalOccurrenceEstimator
from src.models.model_b.growth import HierarchicalGrowthEstimator


@dataclass(frozen=True)
class TrustRules:
    """Constantes structurelles fixées avant scoring ; aucune grille."""
    support_scale: int = 30
    recent_minimum: int = 30
    recent_coverage_minimum: float = .80
    recent_window: str = 'previous_calendar_year'
    anchor_trust: float = 1.


RULES = TrustRules()


def rules_fingerprint():
    """Empreinte du protocole ; vérifiée avant/après les backtests."""
    from pathlib import Path
    payload = {'rules': asdict(RULES), 'B_occurrence': asdict(P1_OCCURRENCE_CONFIG),
               'B_growth': asdict(P1_GROWTH_CONFIG), 'engine_version': '1'}
    payload['implementation'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _iqr(values):
    values = pd.to_numeric(values, errors='coerce').dropna()
    return float(values.quantile(.75)-values.quantile(.25)) if len(values) else np.nan


def _supported_counts(history, candidates, columns):
    counts = history.groupby(columns, dropna=True).size()
    if len(columns) == 1:
        return candidates[columns[0]].map(counts).fillna(0).to_numpy(dtype=float)
    keys = pd.MultiIndex.from_frame(candidates[columns])
    return counts.reindex(keys, fill_value=0).to_numpy(dtype=float)


def _prepare_experts(leases, year):
    """Retour privé en mémoire, utilisable ultérieurement en 2026 sans changer R.

    Aucun label de Y n'est construit. Ce module ne déclenche aucun appel 2026.
    Train de A inchangé ; configurations P1_G1 de B explicitement réutilisées.
    """
    if year not in (2023, 2024, 2025, 2026):
        raise ValueError('Origine hors contrat.')
    origin = prediction_origin(year)
    past = known_leases(leases, origin)
    candidate, history, positive, pending, a = _champion_components(past, year)
    x = candidate.X.copy()
    pairs, _ = build_model_dataset(past)
    train, _ = split_year(pairs, year, MAIN_TARGET)
    growth_train = _p1_growth_training(past, train, year).reset_index(drop=True)
    p_b = HierarchicalOccurrenceEstimator(P1_OCCURRENCE_CONFIG).fit(
        history[list(x.columns)], history.occurrence_target.astype(float)).predict_proba(x)
    growth_model = HierarchicalGrowthEstimator(P1_GROWTH_CONFIG).fit(
        growth_train[['province']], growth_train[MAIN_TARGET].rename(MAIN_TARGET))
    g_b = growth_model.predict(x[['province']])
    province_n = _supported_counts(history, x, ['province'])
    child_n = _supported_counts(history, x, ['province', 'expiry_month'])
    # Support du dernier niveau effectivement utilisé, jamais du bâtiment
    # puisque la combinaison B figée ne possède pas ce niveau.
    p_support = np.where(child_n >= P1_OCCURRENCE_CONFIG.minimum_segment_size, child_n, province_n)
    p_fallback = province_n < P1_OCCURRENCE_CONFIG.minimum_segment_size
    g_support = _supported_counts(growth_train, x, ['province'])
    g_fallback = g_support < P1_GROWTH_CONFIG.minimum_segment_size
    all_history = occurrence_history(past, year)
    last = all_history.loc[all_history.forecast_year.eq(year-1)].copy()
    last_known = last.loc[last.occurrence_target.notna()]
    last_positive = last_known.loc[last_known.occurrence_target.eq(1) & np.isfinite(last_known.unit_growth)]
    coverage = len(last_known)/len(last) if len(last) else 0.
    p_c = float(last_known.occurrence_target.mean()) if len(last_known) else np.nan
    g_c = float(last_positive.unit_growth.median()) if len(last_positive) else np.nan
    p_c_valid = len(last_known) >= RULES.recent_minimum and coverage >= RULES.recent_coverage_minimum
    g_c_valid = len(last_positive) >= RULES.recent_minimum and coverage >= RULES.recent_coverage_minimum
    p_a, g_a = a['probabilities'], a['conditional_growth']
    p_scale = max(float(np.sqrt(p_a[0]*(1-p_a[0]))), 1/(len(history)+1))
    g_spread = _iqr(positive.unit_growth)
    g_scale = max(g_spread, abs(g_a[0]), np.finfo(float).eps)
    recent_pairs = pairs.loc[pairs.year.eq(year-1) & pairs.eligible_main & np.isfinite(pairs[MAIN_TARGET])]
    recent_mature = recent_pairs.loc[recent_pairs.label_available_date.le(origin)]
    return {'year': year, 'candidate': candidate, 'X': x, 'history': all_history,
            'positive': positive, 'pending': pending, 'pairs_recent': recent_pairs,
            'pairs_recent_mature': recent_mature, 'last': last, 'last_known': last_known,
            'last_positive': last_positive, 'coverage': coverage,
            'A_p': p_a.copy(), 'A_g': g_a.copy(), 'B_p': p_b, 'B_g': g_b,
            'C_p': p_c if p_c_valid else np.nan, 'C_g': g_c if g_c_valid else np.nan,
            'C_p_valid': p_c_valid, 'C_g_valid': g_c_valid,
            'p_context': pd.DataFrame({'b_support': p_support, 'b_fallback': p_fallback,
                'b_global': p_a[0], 'long_scale': p_scale, 'long_spread': p_scale,
                'c_support': len(last_known), 'c_coverage': coverage,
                'c_spread': np.sqrt(p_c*(1-p_c)) if np.isfinite(p_c) else np.nan}),
            'g_context': pd.DataFrame({'b_support': g_support, 'b_fallback': g_fallback,
                'b_global': growth_model._global_median, 'long_scale': g_scale,
                'long_spread': g_scale, 'c_support': len(last_positive), 'c_coverage': coverage,
                'c_spread': _iqr(last_positive.unit_growth)}),
            'child_supported': child_n >= P1_OCCURRENCE_CONFIG.minimum_segment_size}


CONTEXT_COLUMNS = frozenset({'b_support','b_fallback','b_global','long_scale',
                            'long_spread','c_support','c_coverage','c_spread'})


def mix_component(a, b, c, context, *, component):
    """Mélange pur, liste fermée du contexte ; aucun argument de label réalisé.

    A est toujours l'ancre. B/C invalides sont ignorés avant multiplication :
    zéro × NaN ne doit jamais contaminer la sortie. Retour unitaire en mémoire.
    """
    if component not in ('occurrence','growth') or set(context.columns) != CONTEXT_COLUMNS:
        raise ValueError('Composante ou contexte incompatible ; aucune cible autorisée.')
    a, b = np.asarray(a,dtype=float), np.asarray(b,dtype=float)
    if a.ndim != 1 or a.shape != b.shape or len(a) != len(context) or not len(a) or not np.isfinite(a).all():
        raise ValueError('Ancre A finie et vecteurs alignés nécessaires.')
    if component == 'occurrence' and not ((a>=0)&(a<=1)).all():
        raise ValueError('Probabilité A hors [0,1].')
    c = np.full(len(a), float(c)) if np.isscalar(c) else np.asarray(c,dtype=float)
    if c.shape != a.shape:
        raise ValueError('Expert C non aligné.')
    scale = context.long_scale.to_numpy(dtype=float)
    if not np.isfinite(scale).all() or (scale<=0).any():
        raise ValueError('Échelle historique positive nécessaire.')
    b_valid = np.isfinite(b) & np.isfinite(context.b_global.to_numpy(dtype=float))
    c_valid = np.isfinite(c)
    if component == 'occurrence':
        b_valid &= (b>=0)&(b<=1)
        c_valid &= (c>=0)&(c<=1)
    support = context.b_support.fillna(0).clip(lower=0).to_numpy(dtype=float)
    supported = ~context.b_fallback.fillna(True).to_numpy(dtype=bool)
    b_clean = np.where(b_valid,b,a)
    # Différence au propre global B ET à A : un repli B/global différent par
    # population de train n'est pas crédité comme information hiérarchique.
    deviation = np.minimum(np.abs(b_clean-a), np.abs(b_clean-context.b_global.fillna(pd.Series(a,index=context.index))))
    trust_b = supported*b_valid*(support/(support+RULES.support_scale))*(deviation/(deviation+scale))
    c_support = context.c_support.fillna(0).clip(lower=0).to_numpy(dtype=float)
    coverage = context.c_coverage.fillna(0).clip(0,1).to_numpy(dtype=float)
    spread = context.c_spread.to_numpy(dtype=float)
    long_spread = context.long_spread.to_numpy(dtype=float)
    gate = (c_valid & (c_support>=RULES.recent_minimum) & (coverage>=RULES.recent_coverage_minimum)
            & np.isfinite(spread) & (spread>=0) & np.isfinite(long_spread) & (long_spread>0))
    c_clean = np.where(c_valid,c,a)
    divergence = np.abs(c_clean-a)
    coherence = np.divide(long_spread,long_spread+np.where(np.isfinite(spread),spread,0),
                          out=np.zeros(len(a)),where=np.isfinite(long_spread)&(long_spread>0))
    trust_c = gate*(c_support/(c_support+RULES.support_scale))*coverage*coherence*(divergence/(divergence+scale))
    trusts = np.column_stack([np.ones(len(a))*RULES.anchor_trust, trust_b, trust_c]).astype(float)
    if not np.isfinite(trusts).all():
        raise ValueError('Confiance non finie ; contexte invalide.')
    weights = trusts/trusts.sum(axis=1,keepdims=True)
    result = weights[:,0]*a + weights[:,1]*b_clean + weights[:,2]*c_clean
    return result, weights


def _orchestrate_prepared(experts):
    p, wp = mix_component(experts['A_p'], experts['B_p'], experts['C_p'], experts['p_context'],component='occurrence')
    g, wg = mix_component(experts['A_g'], experts['B_g'], experts['C_g'], experts['g_context'],component='growth')
    return {**experts, 'p':p, 'g':g, 'contribution':p*g, 'weights_p':wp, 'weights_g':wg,
            'P1_pct':100*float(np.median(p*g)), 'rules_fingerprint':rules_fingerprint()}


def _predict_contextual(leases, year):
    """Préparer une cohorte et mélanger à cutoff ; aucune révélation de cible."""
    return _orchestrate_prepared(_prepare_experts(leases,year))
