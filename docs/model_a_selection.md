# Model A — sélection A7–A10 et méthode d'incertitude

## Décision reproductible

Champion et runner-up sélectionnés sans aucun forecast 2026, exclusivement sur les backtests 2023/2024/2025. Aucun hyperparamètre ML changé depuis A3–A6 ; aucune feature enrichie, donnée externe, pondération optimisée, ensemble ou calibration additionnelle. La fondation et les références réalisées sont inchangées.

**MODEL_A_CHAMPION**

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

**MODEL_A_RUNNER_UP**

```json
{
  "occurrence_method": "historical_global",
  "growth_method": "historical_mean",
  "recency_variant": {
    "occurrence": "equal",
    "growth": "equal"
  },
  "feature_set": "core"
}
```

Le champion médian est un estimateur robuste de croissance conditionnelle, pas une estimation littérale de E[G|transition]. Il prédit le même indice P1 ; sa contribution ne doit pas être décrite comme une espérance exacte. La moyenne demeure le runner-up cohérent avec cette interprétation.

## Récence : définition et expérience limitée

Âge = (prediction_origin − fin de l'année de l'observation).jours / 365,25, en années. Proxy annuel déterministe ; ne pas le confondre avec la date réelle de maturité du label. Les observations sont préalablement qualifiées par temporal_training : cohortes historiques à leurs propres origines, années terminées et labels admissibles à D selon la fondation. Une année future est refusée même avec lambda=0.

w=exp(−lambda×âge). equal : lambda=0 ; mild : lambda=ln(2)/3 ; moderate : lambda=ln(2)/1,5. Taux occurrence = somme(w×O)/somme(w) ; growth moyen = somme(w×G)/somme(w). Médiane non pondérée ; aucun weighted median approximatif. Logistic, Ridge et HGB ne reçoivent aucune récence supplémentaire.

| year | variant | occurrence_brier | predicted_transition_rate_pct | rate_error_pp | mae_pp | rmse_pp | bias_pp | median_absolute_error_pp | median_prediction | median_realized |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023 | equal | 0.009826 | 98.327759 | -0.684586 | 3.838847 | 4.926807 | 0.669141 | 3.054464 | 2.891718 | 2.085372 |
| 2023 | mild | 0.009897 | 97.927007 | -1.085339 | 3.873525 | 4.949131 | 0.817445 | 3.040022 | 3.040022 | 2.085372 |
| 2023 | moderate | 0.010024 | 97.446190 | -1.566155 | 3.906786 | 4.971427 | 0.943086 | 3.113875 | 3.165663 | 2.085372 |
| 2024 | equal | 0.016767 | 98.475120 | 0.180547 | 4.726305 | 5.913266 | 1.501548 | 3.873705 | 3.422782 | 1.699619 |
| 2024 | mild | 0.016765 | 98.161416 | -0.133157 | 4.822081 | 5.984543 | 1.761445 | 3.931508 | 3.682679 | 1.699619 |
| 2024 | moderate | 0.016789 | 97.790398 | -0.504175 | 4.907309 | 6.051388 | 1.976672 | 4.069941 | 3.897906 | 1.699619 |
| 2025 | equal | 0.093779 | 98.189499 | 7.885761 | 4.257661 | 6.081045 | -0.747152 | 2.623904 | 3.175832 | 3.716980 |
| 2025 | mild | 0.092909 | 97.617100 | 7.313361 | 4.251029 | 6.075108 | -0.697186 | 2.606638 | 3.225798 | 3.716980 |
| 2025 | moderate | 0.091852 | 96.854178 | 6.550440 | 4.255731 | 6.079314 | -0.732927 | 2.614114 | 3.190057 | 3.716980 |

La meilleure occurrence pondérée est moderate ; la meilleure growth pondérée est mild. Leur combinaison est le seul challenger de récence du tournoi. Ce choix utilise les scores des trois backtests et est **non imbriqué** : il est potentiellement optimiste, pas une validation indépendante. Le gain occurrence ne justifie pas automatiquement une pondération de growth : ses deux variantes pondérées ont un MAE moyen plus mauvais qu'equal. Aucun test 2026, aucune optimisation de poids.

## Assemblage P1 et interfaces

`model_a_config(occurrence_method, growth_method, recency_variant, feature_set="core")` retourne un dictionnaire JSON explicite. recency_variant accepte une chaîne commune ou un dictionnaire occurrence/growth. La récence est refusée pour les méthodes non expérimentées. `predict_model_a` accepte les mêmes choix et une année de backtest ; il refuse explicitement 2026 à cette phase. `fold_predictions` fixe toutes les prédictions avant révélation des labels test ; `predict_from_fold` réutilise ces prédictions sans nouvel entraînement.

c_i=p_hat_i×g_hat_i ; P1 fraction=médiane(c_i) ; P1 pourcentage=100×médiane(c_i). `assemble_p1` contrôle les probabilités et valeurs finies. Contributions et identifiants restent uniquement en mémoire ; aucun export individuel ni objet modèle persisté. Les références réalisées proviennent de p1_diagnostic, communes à tous les candidats : 2,036102 / 1,607852 / 3,300878 %.

## Tournoi : sept candidats et résultats annuels

Le candidat linéaire est Ridge, conservé comme référence de la phase précédente ; ElasticNet avait un comportement voisin. Aucun challenger à ensemble 50/50 : les résultats A3–A6 ne démontraient pas de complémentarité suffisante.

| year | candidate | predicted_P1_pct | realized_P1_pct | error_pp | absolute_error_pp |
| --- | --- | --- | --- | --- | --- |
| 2023 | global_mean | 2.843362 | 2.036102 | 0.807259 | 0.807259 |
| 2024 | global_mean | 3.370589 | 1.607852 | 1.762736 | 1.762736 |
| 2025 | global_mean | 3.118333 | 3.300878 | -0.182545 | 0.182545 |
| 2023 | recency_best | 2.962386 | 2.036102 | 0.926284 | 0.926284 |
| 2024 | recency_best | 3.601306 | 1.607852 | 1.993454 | 1.993454 |
| 2025 | recency_best | 3.124320 | 3.300878 | -0.176559 | 0.176559 |
| 2023 | global_median | 2.870060 | 2.036102 | 0.833957 | 0.833957 |
| 2024 | global_median | 3.269179 | 1.607852 | 1.661326 | 1.661326 |
| 2025 | global_median | 2.969328 | 3.300878 | -0.331551 | 0.331551 |
| 2023 | logistic_mean | 2.844777 | 2.036102 | 0.808674 | 0.808674 |
| 2024 | logistic_mean | 3.414993 | 1.607852 | 1.807141 | 1.807141 |
| 2025 | logistic_mean | 3.153593 | 3.300878 | -0.147285 | 0.147285 |
| 2023 | global_ridge | 5.042606 | 2.036102 | 3.006504 | 3.006504 |
| 2024 | global_ridge | 6.441329 | 1.607852 | 4.833477 | 4.833477 |
| 2025 | global_ridge | 3.887943 | 3.300878 | 0.587065 | 0.587065 |
| 2023 | global_hgb | 3.471871 | 2.036102 | 1.435769 | 1.435769 |
| 2024 | global_hgb | 4.579315 | 1.607852 | 2.971463 | 2.971463 |
| 2025 | global_hgb | 2.372805 | 3.300878 | -0.928074 | 0.928074 |
| 2023 | logistic_hgb | 3.075105 | 2.036102 | 1.039003 | 1.039003 |
| 2024 | logistic_hgb | 4.451927 | 1.607852 | 2.844075 | 2.844075 |
| 2025 | logistic_hgb | 2.209023 | 3.300878 | -1.091855 | 1.091855 |

| candidate | MAE_P1 | RMSE_P1 | bias_P1 | worst_year_absolute_error | max_to_min_error_spread | occurrence_brier | growth_mae_pp | complexity_level | interpretability_level | max_degradation_vs_baseline_pp | eligible_for_promotion | rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| global_median | 0.942278 | 1.090171 | 0.721244 | 1.661326 | 1.992877 | 0.040124 | 4.271796 | 0 | 0 | 0.149005 | True | 1 |
| global_mean | 0.917514 | 1.124312 | 0.795817 | 1.762736 | 1.945281 | 0.040124 | 4.274271 | 0 | 0 | 0.000000 | True | 2 |
| logistic_mean | 0.921033 | 1.146212 | 0.822843 | 1.807141 | 1.954426 | 0.074207 | 4.274271 | 2 | 2 | 0.044404 | True | 3 |
| recency_best | 1.032099 | 1.273189 | 0.914393 | 1.993454 | 2.170013 | 0.039555 | 4.315545 | 1 | 1 | 0.230718 | True | 4 |
| logistic_hgb | 1.658311 | 1.858354 | 0.930408 | 2.844075 | 3.935930 | 0.074207 | 4.612810 | 4 | 4 | 1.081339 | False | 5 |
| global_hgb | 1.778435 | 1.979254 | 1.159719 | 2.971463 | 3.899537 | 0.040124 | 4.612810 | 3 | 3 | 1.208727 | False | 6 |
| global_ridge | 2.809015 | 3.303847 | 2.809015 | 4.833477 | 4.246412 | 0.040124 | 5.090551 | 2 | 2 | 3.070741 | False | 7 |

## Règle de sélection et justification

Critère principal : MAE P1, années équipondérées. Équivalence opérationnelle fixée à 0,05 pp ; dans le groupe équivalent, choisir la simplicité, puis interprétabilité, pire erreur annuelle, RMSE, biais absolu, Brier occurrence, MAE growth, nom stable pour dernier départage. Les niveaux 0 à 4 croissants représentent une complexité plus forte et une interprétabilité plus faible.

Garde de promotion fixée : une dégradation annuelle absolue de plus de 0,5 pp par rapport au baseline global_mean empêche la promotion. Aucun candidat ni résultat annuel n'est masqué ; les candidatures incomplètes ne peuvent être classées sur moins de trois folds. Ces seuils sont des conventions opérationnelles explicites, pas des seuils de significativité.

La moyenne a le meilleur MAE brut (0,91751 pp), contre 0,94228 pp pour la médiane. Écart 0,02476 pp, inférieur à 0,05 ; leur simplicité est identique. La médiane gagne les départages : pire année 1,66133 contre 1,76274 pp ; RMSE 1,09017 contre 1,12431 ; biais 0,72124 contre 0,79582. Son erreur 2025 est toutefois plus forte (−0,33155 contre −0,18255 pp) : aucun gain universel n'est revendiqué.

La sélection réutilise seulement trois années déjà étudiées : comparaison empirique rétrospective, non preuve de généralisation ni significativité. Aucun poids d'ensemble, seuil ou hyperparamètre n'est réoptimisé pour minimiser les trois erreurs.

## Incertitude : méthode exacte proposée

Bootstrap des unités avec remplacement : 1 000 réplications, random_state=42, recalcul de la médiane des contributions prédites en pourcentage. q10 et q90 décrivent la distribution de composition conditionnelle aux contributions fixes ; aucun réentraînement ni simulation des occurrences/résidus.

Erreurs signées conservées :

| year | error_pp |
| --- | --- |
| 2023 | 0.833957 |
| 2024 | 1.661326 |
| 2025 | -0.331551 |

E=max(|e2023|,|e2024|,|e2025|) = 1.661326 pp.

Formule future figée : **LOW=q10−E ; BASE=P1 ; HIGH=q90+E**. Aucune correction de biais ; aucune marge arbitraire pour la dérive 2025. Le maximum absolu historique fournit une enveloppe conservatrice sur les trois erreurs observées, sans garantie pour des erreurs futures plus grandes.

Pour le champion global, toutes les contributions prédites sont identiques. Le bootstrap est donc **dégénéré**, q10=BASE=q90, largeur nulle ; LOW/HIGH dépendront entièrement du rayon E. Il ne faut pas présenter cette absence de dispersion comme une certitude économique. Les tests couvrent aussi une cohorte synthétique variable et démontrent la reproductibilité et l'absence de mutation des données.

Ces LOW/BASE/HIGH sont des **scénarios**, pas un intervalle de confiance à 95 % ou une couverture garantie. L'enveloppe est calibrée après sélection sur trois années : réutilisation des données et optimisme potentiel. Les démonstrations historiques du bootstrap ne sont pas un backtest prédictif de couverture. Aucune valeur 2026, y compris LOW/BASE/HIGH, n'est calculée.

Non mesuré : nouveaux événements, dérive structurelle, changement réglementaire, erreur de spécification future, incertitude des paramètres après réentraînement, dépendance des unités dans les bâtiments, mécanisme de censure ou révisions CRM. Le bootstrap naïf unité ne corrige pas ces dépendances. Les erreurs historiques en reflètent certains effets passés mais ne garantissent pas leur amplitude future.

## Dérive 2025 et explicabilité

Taux réalisé 90,304 % contre environ 98 % historiquement. moderate prévoit encore 96,854 % : la récence n'anticipe pas l'ampleur de la dérive. Ce risque est conservé dans risk_factors, sans ±X % inventé. La croissance pondérée détériore 2023–2024 et le P1 moyen ; elle n'est pas sélectionnée.

`run_tournament` produit un objet explainability sérialisable : configuration, backtests agrégés, métriques, runner-up, facteurs de risque et limites. Aucun LLM, texte marketing ou dossier unitaire. Le champion est figé avant toute inspection des prédictions 2026.

## Validation et suites

Notebook 08 : contrôles foundation-v1, cohortes 405/645/856/931, occurrences 401/634/773, références communes, agrégats descriptifs et bootstrap historique uniquement. Tests : récence/coupure, equal, contributions et médiane P1, sélection déterministe, équivalence/simple, garde annuelle, référence commune, JSON, CORE, 2026 refusé, injection future, bootstrap/seed/immutabilité et formule des scénarios.

**GO pour A11–A13 sous les réserves fondation et avec scénarios explicitement nommés.** Aucun forecast 2026, publication CRM, nouveau dataset, dépendance ou commit dans A7–A10. La sélection médiane et le runner-up moyenne restent documentés ; ne pas choisir entre eux après inspection de 2026.
