# Model A final — A11 à A13

## Forecast officiel et périmètre

**LOW 0,78 % — BASE 2,44 % — HIGH 4,10 %.** Valeur BASE de calcul : 2.4436448124776575 %.

Définition : Indice médian de contribution attendue des transitions à la croissance annualisée du loyer effectif des unités à échéance. Formule opérationnelle : 100×médiane(p_i×g_i), fractions dans le moteur. g historique médian est un estimateur robuste et non une espérance conditionnelle exacte.

Ce périmètre ne mesure pas la croissance complète du rent roll, occupancy, vacancy, NOI ou tous les loyers. Le champion est au niveau portefeuille, sans hétérogénéité prédictive entre unités. Les segments décrivent la composition ; ils ne modifient pas le forecast.

## Champion figé avant 2026

```json
{
  "occurrence_method": "historical_global",
  "growth_method": "historical_median",
  "recency_variant": {
    "occurrence": "equal",
    "growth": "equal"
  },
  "feature_set": "core"
}
```

La moyenne avait le meilleur MAE brut ; la médiane était dans la bande d'équivalence pratique de 0,05 pp, avec même simplicité et pire erreur/RMSE/biais plus faibles. Le champion a été choisi avant toute inspection de 2026 et n'a pas été retuné. Le packaging ne relance aucun tournoi et ne modifie aucun réglage ML, feature, target, P1 ou règle de cohorte.

## Données admissibles et cohorte

Origine 2025-12-31. Les baux commencés et signés avant cette coupure sont filtrés avant reconstruction de l'historique. Les labels positifs sont mûrs suivant le maximum des débuts/signatures/fins des deux baux ; les labels en attente sont exclus, jamais imputés à zéro. La croissance est la moyenne des transitions admissibles par unité/année historique, comme dans A3–A10. Fidélité temporelle CRM non entièrement prouvée.

Cohorte : 931 unités, Québec 810, Ontario 121. Bâtiments : Daniel-Johnson 128, Le Carlyle 180, Levesque 75, Saint-Elzear 268, The Met 121, Westpark 159. Échéances janvier–décembre : 38/52/65/62/91/143/90/91/108/61/69/61. Accents normalisés uniquement pour les agrégats et gardes, sans modification du stock ou des données.

L'échéance contractuelle connue peut être en 2026 : elle n'est pas une observation future. Aucun label 2026, bail commencé/signé après cutoff, listing actuel ou source externe non qualifiée n'entre dans le calcul. Un écart de cohorte déclenche un arrêt explicite, sans correction silencieuse.

## Composantes réelles du résultat

Historique occurrence : 2 372 labels qualifiés, 2 259 transitions, 113 non-transitions ; 782 labels en attente. Années admissibles : 2018–2025. p=0,9523608768971332, soit 95,23609 % ; probabilité commune aux 931 unités.

Historique growth : 2 259 observations positives. g=0,02565881139972113, soit 2,56588114 % ; q05 −5,22919 %, q95 11,25472 %. Effectifs source des deux baux, dates de début pour annualisation, signature/fin pour maturité : mêmes règles que la fondation. g est commun aux 931 unités. p×g=0,024436448124776574.

Aucune caractéristique bâtiment/chambres n'est un driver direct de ces baselines. CORE est conservé pour la compatibilité et les diagnostics, avec 13/12 features disponibles ; features_used est vide dans le rapport car aucun prédicteur individuel n'est appris. Asking/external sont conservés dans l'API, sans usage prédictif qualifié.

## Sensibilité runner-up

historical_global + historical_mean, equal, CORE : 2.65296483 %, écart +0.20932002 pp. Sensibilité uniquement ; ne remplace jamais le champion.

## Backtests du champion

| Année | Prédit (%) | Réalisé (%) | Erreur (pp) |
|---|---:|---:|---:|
| 2023 | 2,870060 | 2,036102 | +0,833957 |
| 2024 | 3,269179 | 1,607852 | +1,661326 |
| 2025 | 2,969328 | 3,300878 | −0,331551 |

MAE 0,942278 pp ; RMSE 1,090171 pp ; biais +0,721244 pp ; pire erreur absolue 1,661326381 pp. Années équipondérées. Valeurs recalculées par backtest, non hardcodées dans l'API. Sélection rétrospective sur trois années, non imbriquée : aucune significativité forte revendiquée.

## Scénarios et incertitude

Méthode inchangée : 1 000 bootstraps unitaires avec remplacement, seed=42, q10/q90. E=max des erreurs absolues P1 des trois backtests, calculé avec précision complète. LOW=q10−E ; BASE=P1 ; HIGH=q90+E. Valeurs : 0.78231843 / 2.44364481 / 4.10497119 %.

Le bootstrap transversal n'ajoute pas de largeur car le champion applique une contribution identique à chaque unité ; l'incertitude présentée provient donc de l'erreur historique de backtest.

Ce sont des scénarios, pas des intervalles de confiance. Aucune garantie de couverture ; pas de correction de biais ni marge arbitraire pour la dérive 2025. Non couvert : nouveaux chocs, changements réglementaires, dépendance entre unités, nouvelle erreur structurelle ou révisions CRM. Trois erreurs après sélection ne suffisent pas à estimer une loi statistique précise.

## Explicabilité et packaging comparable A/B

L'objet final contient forecast_year, method_name, predicted_growth (pourcentage), target_definition P1, backtest_by_year, aggregate_metrics, uncertainty, drivers, risk_factors et limitations. Il permet d'expliquer le résultat, la formule, la sélection antérieure, les composantes réelles et la fiabilité. Aucun LLM ni récit causal.

Les drivers sont p historique, g médian, éligibilité de cohorte et comportement des backtests. Les risques incluent dérive occurrence 2025, seulement trois années, fidélité CRM, maturité/censure et limites de l'indice/scénarios. A/B doit partager exactement le même target_definition, cohorte et origines.

## API et reproductibilité

```python
from src.models.model_a import estimate_2026, backtest
forecast_pct = estimate_2026(leases, asking, external=None)
report = estimate_2026(leases, asking, external=None, return_details=True)
result_2024 = backtest(leases, 2024)
```

Signature : estimate_2026(leases, asking, external=None, *, return_details=False), backtest(leases, target_year). Compatible avec les arguments et le scalaire pourcentage du starter ; starter inchangé. L'ancienne interface expérimentale predict_model_a reste limitée aux backtests pour préserver les expériences A7–A10. La nouvelle API publique couvre le champion final autorisé.

Exécuter notebooks/09_model_a_2026_forecast.ipynb depuis le noyau .venv existant. Il vérifie empreintes, références et cohorte, puis crée trois JSON agrégés dans outputs/reports et deux figures dans outputs/figures. Aucun export individuel, identifiant d'unité, dataset nettoyé ou modèle persisté.

**MODEL A COMPLETE : YES. READY FOR A-vs-B TOURNAMENT : YES**, sur le périmètre P1 commun et sous les réserves documentées. Aucun commit ou publication CRM.
