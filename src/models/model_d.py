"""Expérience fixe : croissance historique bâtiment → global, alpha ex ante."""
import numpy as np
import pandas as pd

ALPHA = 30.0
MIN_DISPLAY_SUPPORT = 5


def _buildings(values):
    """Clés de segmentation temporaires ; les entrées sources restent intactes."""
    result = values.astype('string')
    return result.where(result.str.strip().ne(''))


class BuildingShrunkGrowth:
    """Médianes rétrécies, sans hyperparamètre à sélectionner.

    L'appelant fournit exclusivement l'historique positif d'occurrence mûr.
    Une croissance négative est valide et ne doit pas être éliminée.
    """
    def __init__(self):
        self.global_growth = None
        self._segments = None

    def fit(self, buildings, growth):
        """Ajuster sur séries alignées ; refuser labels absents ou non finis."""
        if not isinstance(buildings, pd.Series) or not isinstance(growth, pd.Series):
            raise ValueError('Deux séries historiques alignées sont nécessaires.')
        if not buildings.index.equals(growth.index) or len(buildings) != len(growth):
            raise ValueError('Historique bâtiment/croissance non aligné.')
        values = pd.to_numeric(growth, errors='coerce').to_numpy(dtype=float)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError('Historique qualifié vide ou croissance non finie.')
        self.global_growth = float(np.median(values))
        data = pd.DataFrame({'building': _buildings(buildings), 'growth': values}, index=buildings.index)
        self._segments = {}
        for building, group in data.groupby('building', sort=True, dropna=True):
            n = len(group)
            raw = float(group.growth.median())
            weight = n/(n+ALPHA)
            self._segments[building] = (n, raw, weight,
                                       weight*raw+(1-weight)*self.global_growth)
        return self

    def _check_fitted(self):
        if self.global_growth is None:
            raise RuntimeError('Ajuster l’estimateur avant de prédire.')

    def predict(self, buildings):
        """Une fraction par candidate, uniquement en mémoire."""
        self._check_fitted()
        keys = _buildings(buildings)
        return np.asarray([self._segments[b][3] if pd.notna(b) and b in self._segments
                           else self.global_growth for b in keys], dtype=float)

    def diagnostics(self, candidates):
        """Médianes historiques et poids de cohorte, sans identifiants d'unité."""
        self._check_fitted()
        keys = _buildings(candidates).fillna('BÂTIMENT MANQUANT')
        counts = keys.value_counts()
        labels = sorted(set(self._segments).union(counts.index))
        rows = []
        for building in labels:
            n, raw, weight, shrunk = self._segments.get(
                building, (0, np.nan, 0., self.global_growth))
            count = int(counts.get(building, 0))
            # n=0 affiche le global public ; n=1..4 masque tout chiffre pouvant
            # reconstituer une croissance historique individuelle.
            publish = n == 0 or n >= MIN_DISPLAY_SUPPORT
            rows.append({'building': building, 'candidate_units': count,
                         'cohort_share_pct': 100*count/len(candidates) if len(candidates) else 0.,
                         'n_b': n, 'w_b': weight, 'g_global_pct': 100*self.global_growth,
                         'g_b_pct': 100*raw if n >= MIN_DISPLAY_SUPPORT else np.nan,
                         'g_b_shrunk_pct': 100*shrunk if publish else np.nan,
                         'fallback_global': n == 0,
                         'history_statistics_masked': 0 < n < MIN_DISPLAY_SUPPORT})
        return pd.DataFrame(rows)
