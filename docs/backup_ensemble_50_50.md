# Backup final JADCO — ensemble Model A / Model B 50/50

## Méthode et cible commune

La cible est exactement l’« Indice médian de contribution attendue des transitions à la croissance annualisée du loyer effectif des unités à échéance ».

Pour chaque unité candidate, `p_i` est la probabilité de transition, `g_i` la croissance annualisée attendue conditionnelle à une transition, et `c_i = p_i * g_i`. L’indice portefeuille est `P1 = médiane(c_i)`, exprimé en pourcentage.

Model A est appelé via ses interfaces finales existantes `backtest(leases, target_year)` et `estimate_2026(leases, asking, external=None)`. Model B est appelé via son tournoi `run_p1_combination_tournament(leases)` pour les folds historiques et ses composants existants pour appliquer P1_G1 au cutoff 2026. Le wrapper vérifie que les backtests ont les mêmes origines, tailles de cohorte et valeurs réalisées avant de comparer les prévisions.

## Règle d’ensemble fixe

La seule formule d’ensemble est :

```text
Forecast_backup = 0.50 * Forecast_A + 0.50 * Forecast_B
```

Les poids sont exactement 0.50 / 0.50 et n’ont jamais été optimisés. Aucun autre poids n’a été testé et aucun modèle supplémentaire n’a été créé.

## Backtests historiques

Origines imposées par la fondation : 2023 au 2022-12-31, 2024 au 2023-12-31, 2025 au 2024-12-31. B est la combinaison déjà existante P1_G1 : occurrence hiérarchique province → province + mois d’échéance avec shrinkage 20 ; croissance hiérarchique province avec shrinkage 1.

| Année | P1 réalisé (%) | Model A (%) | Erreur A (pp) | Model B P1_G1 (%) | Erreur B (pp) | Ensemble 50/50 (%) | Erreur ensemble (pp) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2023 | 2.036102 | 2.870060 | 0.833957 | 2.871314 | 0.835212 | 2.870687 | 0.834585 |
| 2024 | 1.607852 | 3.269179 | 1.661326 | 3.236099 | 1.628247 | 3.252639 | 1.644787 |
| 2025 | 3.300878 | 2.969328 | −0.331551 | 2.924144 | −0.376735 | 2.946736 | −0.354143 |

Metrics use equal-weighted annual fold errors in percentage points; bias is prediction minus realized.

| Modèle | MAE (pp) | RMSE (pp) | Biais (pp) | Pire erreur absolue (pp) |
|---|---:|---:|---:|---:|
| Model A | 0.942278 | 1.090171 | 0.721244 | 1.661326 |
| Model B P1_G1 | 0.946731 | 1.078687 | 0.695575 | 1.628247 |
| Ensemble 50/50 | 0.944505 | 1.084323 | 0.708410 | 1.644787 |

Ces résultats sont présentés tels quels. Ils n’ont pas servi à modifier les poids.

## Forecast 2026

Origine : **2025-12-31**. A est calculé par l’API finale de Model A. B applique P1_G1 avec les configurations existantes ; sa cohorte compte 931 unités.

| Sortie | Forecast (%) |
|---|---:|
| Model A 2026 | 2.443645 |
| Model B P1_G1 2026 | 2.510385 |
| **Backup ensemble 50/50 2026** | **2.477015** |

Le backup vaut exactement `0.50 * 2.4436448124776575 + 0.50 * 2.5103846530942655 = 2.4770147327859613 %`. Il s’agit d’un résultat agrégé ; l’API ne retourne ni probabilités/contributions par unité ni identifiants CRM.

## Implémentation et exécution

`src/models/ensemble_50_50.py` contient `backtest_ensemble_50_50` et `estimate_2026_ensemble`. Le notebook `notebooks/backup_ensemble_50_50.ipynb` charge les données locales, exécute les backtests, affiche la comparaison puis calcule le forecast 2026.

Le wrapper ne modifie aucun code Model A, Model B ou fondation et ne lit aucune donnée externe. Il n’utilise que les origines et données admissibles déterminées par les APIs/fonctions existantes. Les clés CRM sont utilisées en mémoire uniquement pour aligner les prédictions sur les unités candidates, jamais retournées.

## Limites

- Les métriques reposent sur trois folds historiques seulement ; elles ne donnent pas une estimation robuste de toutes les erreurs futures.
- L’indice est une médiane de contributions attendues sur la cohorte à échéance, pas une prévision du rent roll complet, du taux d’inoccupation, du revenu net ou de tous les loyers.
- La fidélité des dates CRM et la maturité des labels restent des limites documentées de la fondation.
- Aucune donnée externe et aucun résultat observé en 2026 ne sont utilisés.
- Les poids 50/50 sont une contrainte de ce backup, pas le résultat d’une optimisation ni une affirmation qu’ils minimisent l’erreur.
