"""CORE conforme à foundation-v1 ; extension historique expérimentale séparée."""
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from src.features.model_dataset import (
    UNIT_KEY, ADMISSIBLE_FEATURES, OCCURRENCE_FEATURES, prediction_origin as contract_origin,
    build_prediction_cohort, build_occurrence_features, build_model_dataset,
    reveal_occurrence, available_external_context, known_leases, MAIN_TARGET,
)

YEARS = (2023, 2024, 2025, 2026)
CORE_FEATURES = OCCURRENCE_FEATURES
EXPERIMENTAL_FEATURES = ('previous_same_unit_growth', 'unit_historical_median_growth',
                        'recent_building_growth', 'recent_portfolio_growth')
ENRICHED_FEATURES = CORE_FEATURES + EXPERIMENTAL_FEATURES
MIN_RECENT_SUPPORT = 5
EXPERIMENT_SAFETY = {
    'prior_effective_rent': 'NOT_IDENTIFIABLE',
    'concession_gap': 'NOT_IDENTIFIABLE',
    'rent_position': 'NOT_IDENTIFIABLE',
    **{name: 'SAFE_WITH_CONDITIONS' for name in EXPERIMENTAL_FEATURES},
}
DEFERRED_FEATURES = (
    'current_effective_rent', 'effective_rent_per_sqft', 'rent_position',
    'concession_gap', 'previous_effective_growth', 'history_count',
    'unit_median_growth', 'recent_building_growth', 'recent_portfolio_growth',
    'recent_province_growth', 'history_available',
)


@dataclass
class ModelAFeatureDataset:
    """Tables alignées en mémoire : audit, X, labels optionnels, contexte séparé.

    Aucun identifiant/label ne fait partie de X. expiry_month est réservé à
    l'occurrence ; X_growth conserve les 12 noms CORE, plus quatre en ENRICHED.
    """
    audit: pd.DataFrame
    X: pd.DataFrame
    labels: Optional[pd.DataFrame]
    external_context: Optional[pd.DataFrame]
    feature_set: str = 'core'

    @property
    def X_growth(self):
        columns = tuple(ADMISSIBLE_FEATURES) + (EXPERIMENTAL_FEATURES if self.feature_set == 'enriched' else ())
        return self.X.loc[:, list(columns)].copy()


def select_model_a_features(dataset, features, purpose='occurrence'):
    """Refuser tout nom non autorisé, y compris des features économiques différées."""
    allowed = OCCURRENCE_FEATURES if purpose == 'occurrence' else ADMISSIBLE_FEATURES if purpose == 'growth' else None
    if allowed is not None and dataset.feature_set == 'enriched':
        allowed = tuple(allowed) + EXPERIMENTAL_FEATURES
    if allowed is None or set(features)-set(allowed) or len(features)!=len(set(features)):
        raise ValueError('Liste de features incompatible avec le contrat figé.')
    return dataset.X.loc[:, list(features)].copy()


def build_model_a_features(leases, target_year, prediction_origin=None,
                           include_targets=False, external=None, feature_set="core"):
    """Construire une cohorte à date ; révéler les labels uniquement sur demande.

    Le cutoff doit être celui du contrat. CORE ne consulte ni prix
    effectifs, ni historique de croissance. ENRICHED utilise exclusivement les
    transitions historiques mûres et exige la réserve de fidélité documentée. Aucun
    fallback/imputation appris. external est filtré puis conservé comme contexte
    d'audit : aucune feature externe n'est autorisée dans X à ce stade.
    Les labels sont rétrospectifs ; aucun label 2026 n'est fabriqué.
    """
    if feature_set not in ('core', 'enriched'):
        raise ValueError('Feature set inconnu : core ou enriched requis.')
    if target_year not in YEARS:
        raise ValueError('Année hors des quatre origines du contrat.')
    origin = contract_origin(target_year)
    if prediction_origin is not None and pd.Timestamp(prediction_origin) != origin:
        raise ValueError('Prediction origin incompatible avec le contrat.')
    if include_targets and target_year == 2026:
        raise ValueError('Aucun label futur 2026 ne peut être demandé.')
    cohort = build_prediction_cohort(leases, target_year)
    x = build_occurrence_features(leases, cohort, target_year).reset_index(drop=True)
    if tuple(x.columns) != OCCURRENCE_FEATURES:
        raise ValueError('Schéma de features différent du contrat figé.')
    if feature_set == 'enriched':
        extra = _history_features(leases, cohort, origin)
        x = pd.concat([x, extra], axis=1)
    audit = cohort[list(UNIT_KEY)].copy().reset_index(drop=True)
    audit['target_year'] = int(target_year)
    audit['prediction_origin'] = origin
    labels = None
    # Seule cette branche consulte les transitions révélées après la coupure.
    if include_targets:
        transitions, _ = build_model_dataset(leases)
        revealed = reveal_occurrence(cohort, transitions, target_year)
        labels = revealed[['occurrence_target', 'unit_growth']].rename(columns={
            'occurrence_target':'transition_occurred', 'unit_growth':'conditional_growth'}).reset_index(drop=True)
    context = available_external_context(external,target_year) if external is not None else None
    return ModelAFeatureDataset(audit, x, labels, context, feature_set)


def assert_foundation_reference(leases):
    """Garde explicite de l'extrait de référence ; arrêter en cas d'écart réel."""
    expected = {2023:(405,401),2024:(645,634),2025:(856,773),2026:(931,None)}
    rows=[]
    for year, (count, occurred) in expected.items():
        result = build_model_a_features(leases,year,include_targets=occurred is not None)
        actual = None if result.labels is None else int(result.labels.transition_occurred.sum())
        if len(result.X)!=count or actual!=occurred:
            raise AssertionError(f'Écart foundation-v1 en {year} : cohorte ou occurrences ; arrêt obligatoire.')
        if year==2026:
            if result.X.province.value_counts().to_dict()!={'Quebec':810,'Ontario':121}:
                raise AssertionError('Écart de composition provinciale 2026.')
            buildings={'Daniel-Johnson':128,'Le Carlyle':180,'Levesque':75,
                       'Saint-Elzear':268,'The Met':121,'Westpark':159}
            if result.X.building.value_counts().to_dict()!=buildings:
                raise AssertionError('Écart de composition des bâtiments 2026.')
        rows.append({'year':year,'candidates':count,'observed_units':actual,
                     'prediction_origin':str(contract_origin(year).date())})
    return pd.DataFrame(rows)


def feature_screening(datasets):
    """Recommandation descriptive, sans sélection selon une corrélation globale."""
    pooled = pd.concat([d.X for d in datasets.values()],ignore_index=True)
    redundant = {'property_code','bathrooms','floor','unit_subtype','forecast_year'}
    rows=[]
    for feature in OCCURRENCE_FEATURES:
        constant = pooled[feature].nunique(dropna=True)<=1
        status = 'DROP_FROM_MODEL_A' if constant else 'OPTIONAL' if feature in redundant else 'KEEP'
        growth = status!='DROP_FROM_MODEL_A' and feature in ADMISSIBLE_FEATURES
        rows.append({'feature':feature,'type':str(pooled[feature].dtype),
            'available_rate':float(pooled[feature].notna().mean()),
            'point_in_time_safe':True,'candidate_for_occurrence':status!='DROP_FROM_MODEL_A',
            'candidate_for_growth':growth,'recommendation':status,
            'reason':'constante' if constant else 'redondance/complexité à limiter' if feature in redundant else 'sens structurel ou contractuel',
            'warning':'sécurité conditionnelle à la fidélité CRM ; recommandations à tester sur train seul'
                + (' ; occurrence uniquement' if feature=='expiry_month' else '')})
    for feature in DEFERRED_FEATURES:
        rows.append({'feature':feature,'type':'non construite','available_rate':np.nan,
            'point_in_time_safe':False,'candidate_for_occurrence':False,'candidate_for_growth':False,
            'recommendation':'DROP_FROM_MODEL_A','reason':'hors liste fermée ou disponibilité non qualifiée',
            'warning':'nécessite une décision explicite ultérieure ; aucun calcul ni fallback'})
    return pd.DataFrame(rows)



def _history_features(leases, cohort, origin):
    """Extension expérimentale : transitions mûres, sans données futures.

    Borne conservatrice déjà définie par la fondation. Elle ne prouve pas la
    date de calcul source ni l'absence de révisions CRM. SAFE_WITH_CONDITIONS
    suppose que les valeurs des baux terminés étaient disponibles et fidèles.
    Fenêtre récente unique (D−365 jours, D], au moins cinq transitions.
    Aucun fallback temporel, aucune imputation, aucun retrait d'extrême.
    """
    past = known_leases(leases, origin)
    pairs, _ = build_model_dataset(past)
    mature = (pairs.eligible_main & pairs.label_available_date.le(origin)
              & pairs.new_lease_date.le(origin) & np.isfinite(pairs[MAIN_TARGET]))
    history = pairs.loc[mature].copy()
    keys = list(UNIT_KEY)
    # Ne jamais sauter un bail invalide dans le moteur ; seule la statistique
    # historique porte ici sur les transitions mathématiquement admissibles.
    ordered = history.sort_values(keys + ['new_lease_date'], kind='stable')
    previous = ordered.drop_duplicates(keys, keep='last')[keys + [MAIN_TARGET]].rename(
        columns={MAIN_TARGET:'previous_same_unit_growth'})
    medians = history.groupby(keys, as_index=False)[MAIN_TARGET].median().rename(
        columns={MAIN_TARGET:'unit_historical_median_growth'})
    out = cohort[keys].merge(previous,on=keys,how='left',validate='one_to_one')
    out = out.merge(medians,on=keys,how='left',validate='one_to_one')
    recent = history.loc[history.new_lease_date.gt(origin-pd.Timedelta(days=365))]
    building = recent.groupby('sBuilding')[MAIN_TARGET].agg(['count','median'])
    building_medians = building['median'].where(building['count'].ge(MIN_RECENT_SUPPORT))
    # Appartenance du candidat : snapshot du bail connu, pas bâtiment futur.
    candidate_buildings = build_occurrence_features(leases,cohort,origin.year+1).building
    out['recent_building_growth'] = candidate_buildings.map(building_medians).to_numpy()
    out['recent_portfolio_growth'] = recent[MAIN_TARGET].median() if len(recent)>=MIN_RECENT_SUPPORT else np.nan
    return out.loc[:,list(EXPERIMENTAL_FEATURES)].reset_index(drop=True)
