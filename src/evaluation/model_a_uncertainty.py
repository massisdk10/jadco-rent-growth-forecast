"""Scénarios P1 : bootstrap de composition et enveloppe d'erreur historique."""
import numpy as np
from src.models.model_a import assemble_p1


def bootstrap_p1(contributions, n_bootstrap=1000, random_state=42):
    """Rééchantillonnage des unités avec remplacement ; P1 en pourcentage."""
    c=np.asarray(contributions,dtype=float)
    if c.ndim!=1 or not len(c) or not np.isfinite(c).all():
        raise ValueError('Contributions unitaires finies requises.')
    if not isinstance(n_bootstrap,int) or n_bootstrap<100 or random_state is None:
        raise ValueError('Au moins 100 réplications et seed explicite requis.')
    rng=np.random.default_rng(random_state)
    # Une réplication à la fois : pas de grande matrice mémoire ni export.
    return np.array([100*np.median(c[rng.integers(0,len(c),len(c))]) for _ in range(n_bootstrap)])


def historical_error_envelope(backtest):
    """Erreurs signées conservées ; rayon max absolu des trois années fixes."""
    if sorted(backtest.year.tolist())!=[2023,2024,2025]:
        raise ValueError('Une erreur par backtest 2023/2024/2025 ; aucune année 2026.')
    errors=backtest.error_pp.to_numpy(dtype=float)
    if not np.isfinite(errors).all():
        raise ValueError('Erreurs finies requises.')
    return {'signed_errors_pp':errors.tolist(),'model_error_radius_pp':float(np.max(np.abs(errors)))}


def uncertainty_scenarios(probabilities, growth, backtest, n_bootstrap=1000, random_state=42):
    """LOW=q10(bootstrap)−E, BASE=P1, HIGH=q90(bootstrap)+E, E=max|erreur|.

    Scénarios descriptifs, sans couverture probabiliste garantie. Ne corrige
    pas le biais, n'ajoute pas un pourcentage arbitraire au drift 2025. Aucun
    usage 2026 dans cette phase ; callers limités aux backtests par le notebook.
    """
    assembled=assemble_p1(probabilities,growth)
    boot=bootstrap_p1(assembled['contributions'],n_bootstrap,random_state)
    error=historical_error_envelope(backtest)
    q10,q90=np.quantile(boot,[.1,.9]);radius=error['model_error_radius_pp']
    return {'LOW':float(q10-radius),'BASE':assembled['p1_pct'],'HIGH':float(q90+radius),
            'bootstrap_q10_pct':float(q10),'bootstrap_q90_pct':float(q90),**error,
            'n_bootstrap':n_bootstrap,'random_state':random_state,
            'interpretation':'Scénarios de composition + enveloppe empirique ; aucun niveau de confiance garanti',
            'risk_factors':['Dérive du taux de transition observée en 2025','Dépendance entre unités/bâtiments non modélisée']}
