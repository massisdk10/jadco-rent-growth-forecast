# Model A — A3 à A6 : modélisation CORE

## Décision et périmètre

CORE uniquement : 13 features occurrence, 12 croissance. Aucune activation d'ENRICHED, aucune donnée externe ni calibration additionnelle. La décision humaine conserve les sept features économiques/historiques hors des modèles principaux. Le contrat commun et foundation-v1 ne sont pas modifiés.

Les résultats recommandent historical_global pour l'occurrence et historical_mean comme référence de croissance cohérente avec E[G|transition]. historical_median est un comparateur robuste presque équivalent en MAE. Les challengers linéaires et Logistic restent disponibles pour A7–A9 ; HGB n'est pas prioritaire. Ce diagnostic ne sélectionne pas le champion final P1.

## Protocole et limites temporelles

Walk-forward exclusif 2023/2024/2025, coupures 2022/2023/2024-12-31. `temporal_training` réutilise `occurrence_history` sans modification : chaque cohorte historique est reconstruite à son origine annuelle ; les baux sont d'abord limités aux débuts/signatures connus à D. Une croissance positive exige une transition admissible mûre à D : maximum des débuts/signatures/fins des deux baux. Les labels en attente sont exclus, jamais imputés à zéro.

L'historique G est la moyenne par unité des transitions admissibles alors connues, comme dans la baseline P1 fondation ; de nouvelles transitions peuvent modifier cette moyenne ultérieurement. Les label_available_date ne prouvent pas le moment réel de calcul ni l'absence de révisions CRM. La qualification des négatifs dépend aussi de la complétude de l'extrait. Simulation prudente sur snapshot rétrospectif, pas preuve d'un journal historique.

Les features train sont prises à l'origine de chaque année historique, non à l'origine du test. Labels test révélés seulement après prédiction. 2026 est interdit par les interfaces de training : seul le garde de cohorte vérifie ses features. Aucun split aléatoire, tuning, pondération de récence, bootstrap ou forecast final.

## Preprocessing et configurations fixes

Quatre catégories : property_code, building, province, unit_subtype ; conversion des manquants, imputation constante explicite et OneHotEncoder(handle_unknown="ignore") dense. Les neuf/huit autres features sont numériques : conversion des valeurs invalides en NaN, imputation médiane sur train uniquement et standardisation pour Logistic/Ridge/ElasticNet. Les colonnes totalement vides sont conservées (zéro numérique fixé, indépendant du futur). HGB reçoit une représentation dense compatible, sans standardisation.

- Logistic : L2, C=1, class_weight=None, max_iter=2000, random_state=42 ; pas de calibration supplémentaire.
- Ridge : alpha=10.
- ElasticNet : alpha=0,001, l1_ratio=0,1, max_iter=20000, random_state=42.
- HGB : learning_rate=0,05, max_iter=100, max_leaf_nodes=7, min_samples_leaf=20, l2_regularization=1, early_stopping=False, random_state=42.

Une configuration préétablie par modèle, aucune sélection sur les résultats des années test. Logistic exige au moins dix observations de chaque classe ; croissance ML au moins trente transitions ; sinon seules les baselines admissibles existent. Aucun training set fabriqué. Un historique vide doit être signalé non calculable ; les origines actuelles disposent toutes d'historique suffisant. Les transformateurs restent internes aux pipelines, ajustés sur train.

## Données d'entraînement

| year | n_train_occurrence | non_transitions | transitions | pending | n_train_growth | n_test | n_test_growth |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2023 | 897 | 15 | 882 | 351 | 882 | 405 | 401 |
| 2024 | 1246 | 19 | 1227 | 407 | 1227 | 645 | 634 |
| 2025 | 1657 | 30 | 1627 | 641 | 1627 | 856 | 773 |

Les 15/19/30 non-transitions limitent la calibration et la sélection d'hyperparamètres. L'accuracy n'est pas une métrique principale.

## Occurrence

Taux et erreurs de taux en pourcentage/points ; Brier sans unité. Agrégats à poids égal des années.

| year | model | n_train | n_test | predicted_transition_rate | realized_transition_rate | brier | rate_error | log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023 | historical_global | 897 | 405 | 98.32776 | 99.01235 | 0.00983 | -0.68459 | 0.05710 |
| 2023 | Logistic | 897 | 405 | 86.99528 | 99.01235 | 0.05861 | -12.01707 | 0.18445 |
| 2024 | historical_global | 1246 | 645 | 98.47512 | 98.29457 | 0.01677 | 0.18055 | 0.08645 |
| 2024 | Logistic | 1246 | 645 | 93.30132 | 98.29457 | 0.03200 | -4.99326 | 0.10379 |
| 2025 | historical_global | 1657 | 856 | 98.18950 | 90.30374 | 0.09378 | 7.88576 | 0.40547 |
| 2025 | Logistic | 1657 | 856 | 87.85629 | 90.30374 | 0.13200 | -2.44745 | 0.65847 |

| model | folds | mean_brier | mean_abs_rate_error | worst_brier |
| --- | --- | --- | --- | --- |
| historical_global | 3 | 0.04012 | 2.91696 | 0.09378 |
| Logistic | 3 | 0.07421 | 6.48593 | 0.13200 |

La calibration descriptive par bins de 0,1 et la distribution des probabilités figurent dans le notebook. Aucun Platt/isotonic : support négatif trop faible. Logistic a un Brier moins bon sur les trois folds ; l'amélioration du taux moyen 2025 ne prouve pas une meilleure calibration individuelle.

## Croissance conditionnelle

Trois baselines : moyenne historique positive (référence P1), médiane historique positive et médiane Y−1 positive. Une médiane est un comparateur, pas E[G|transition]. Contrairement aux anciens baselines conditionnels sur toutes les paires, le train utilise ici les unités/années des cohortes candidates qualifiées. Plusieurs transitions d'une unité sont moyennées suivant P1 ; aucune croissance artificiellement nulle pour les négatifs. Le test comprend seulement 401/634/773 unités ayant une transition ; toutes les candidates reçoivent cependant une prédiction avant scoring.

Toutes les métriques sont en points ; médianes en pourcentage. Aucune suppression d'extrêmes. Le notebook affiche leur MAE avec seuils Q1−3IQR / Q3+3IQR calculés sur train, statistiques masquées sous cinq observations.

| year | model | n_train | n_test | mae_pp | rmse_pp | bias_pp | median_absolute_error_pp | median_prediction | median_realized |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023 | historical_mean | 882 | 401 | 3.83885 | 4.92681 | 0.66914 | 3.05446 | 2.89172 | 2.08537 |
| 2023 | historical_median | 882 | 401 | 3.84514 | 4.93057 | 0.69629 | 3.02731 | 2.91887 | 2.08537 |
| 2023 | previous_year_median | 882 | 401 | 3.85777 | 5.00184 | -1.09214 | 2.81969 | 1.13044 | 2.08537 |
| 2023 | Ridge | 882 | 401 | 4.68329 | 5.67732 | 2.85802 | 3.86622 | 5.12270 | 2.08537 |
| 2023 | ElasticNet | 882 | 401 | 4.67761 | 5.67431 | 2.85330 | 3.83359 | 5.12486 | 2.08537 |
| 2023 | HGB | 882 | 401 | 4.15561 | 5.23300 | 1.40389 | 3.41007 | 3.51894 | 2.08537 |
| 2024 | historical_mean | 1227 | 634 | 4.72631 | 5.91327 | 1.50155 | 3.87371 | 3.42278 | 1.69962 |
| 2024 | historical_median | 1227 | 634 | 4.69024 | 5.88796 | 1.39857 | 3.80560 | 3.31980 | 1.69962 |
| 2024 | previous_year_median | 1227 | 634 | 4.60058 | 5.94400 | -1.61836 | 3.62610 | 0.30288 | 1.69962 |
| 2024 | Ridge | 1227 | 634 | 6.36143 | 7.41486 | 4.67585 | 5.71548 | 6.53660 | 1.69962 |
| 2024 | ElasticNet | 1227 | 634 | 6.23461 | 7.28635 | 4.50462 | 5.44447 | 6.40804 | 1.69962 |
| 2024 | HGB | 1227 | 634 | 5.23001 | 6.32740 | 2.64364 | 4.39861 | 4.64830 | 1.69962 |
| 2025 | historical_mean | 1627 | 773 | 4.25766 | 6.08104 | -0.74715 | 2.62390 | 3.17583 | 3.71698 |
| 2025 | historical_median | 1627 | 773 | 4.28001 | 6.10155 | -0.89890 | 2.72375 | 3.02408 | 3.71698 |
| 2025 | previous_year_median | 1627 | 773 | 5.04008 | 6.70988 | -2.93285 | 3.85695 | 0.99013 | 3.71698 |
| 2025 | Ridge | 1627 | 773 | 4.22693 | 6.02383 | -0.20226 | 2.53843 | 3.91059 | 3.71698 |
| 2025 | ElasticNet | 1627 | 773 | 4.21624 | 6.02033 | -0.06831 | 2.46096 | 4.02982 | 3.71698 |
| 2025 | HGB | 1627 | 773 | 4.45281 | 6.25473 | -1.32733 | 3.06884 | 2.41407 | 3.71698 |

| model | folds | mean_mae_pp | mean_rmse_pp | mean_abs_bias_pp | worst_year_mae_pp | stability_mae_std_pp |
| --- | --- | --- | --- | --- | --- | --- |
| historical_mean | 3 | 4.27427 | 5.64037 | 0.97261 | 4.72631 | 0.36249 |
| historical_median | 3 | 4.27180 | 5.64003 | 0.99792 | 4.69024 | 0.34506 |
| previous_year_median | 3 | 4.49948 | 5.88524 | 1.88112 | 5.04008 | 0.48794 |
| Ridge | 3 | 5.09055 | 6.37200 | 2.57871 | 6.36143 | 0.91776 |
| ElasticNet | 3 | 5.04282 | 6.32700 | 2.47541 | 6.23461 | 0.86351 |
| HGB | 3 | 4.61281 | 5.93838 | 1.79162 | 5.23001 | 0.45298 |

La stabilité est l'écart-type des trois MAE annuels (ddof=0), pas un intervalle de confiance. Les modèles linéaires surestiment 2023–2024 malgré une amélioration 2025. HGB ne bat pas robustement les références simples.

## P1 diagnostique

100 × médiane(p_hat × g_hat) sur toutes les candidates, contre 100 × médiane(O × G). Zéro sans événement définit l'indice et ne décrit pas le mécanisme économique d'une unité sans transition. Aucun champion final ; pas d'espérance de médiane ni de rent roll.

| year | occurrence_model | growth_model | predicted_P1_pct | realized_P1_pct | error_pp |
| --- | --- | --- | --- | --- | --- |
| 2023 | historical_global | historical_mean | 2.84336 | 2.03610 | 0.80726 |
| 2023 | historical_global | historical_median | 2.87006 | 2.03610 | 0.83396 |
| 2023 | historical_global | previous_year_median | 1.11153 | 2.03610 | -0.92457 |
| 2023 | historical_global | Ridge | 5.04261 | 2.03610 | 3.00650 |
| 2023 | historical_global | ElasticNet | 5.05080 | 2.03610 | 3.01470 |
| 2023 | historical_global | HGB | 3.47187 | 2.03610 | 1.43577 |
| 2023 | Logistic | historical_mean | 2.84478 | 2.03610 | 0.80867 |
| 2023 | Logistic | historical_median | 2.87149 | 2.03610 | 0.83539 |
| 2023 | Logistic | previous_year_median | 1.11208 | 2.03610 | -0.92402 |
| 2023 | Logistic | Ridge | 4.73543 | 2.03610 | 2.69933 |
| 2023 | Logistic | ElasticNet | 4.76031 | 2.03610 | 2.72421 |
| 2023 | Logistic | HGB | 3.07511 | 2.03610 | 1.03900 |
| 2024 | historical_global | historical_mean | 3.37059 | 1.60785 | 1.76274 |
| 2024 | historical_global | historical_median | 3.26918 | 1.60785 | 1.66133 |
| 2024 | historical_global | previous_year_median | 0.29826 | 1.60785 | -1.30959 |
| 2024 | historical_global | Ridge | 6.44133 | 1.60785 | 4.83348 |
| 2024 | historical_global | ElasticNet | 6.31545 | 1.60785 | 4.70760 |
| 2024 | historical_global | HGB | 4.57932 | 1.60785 | 2.97146 |
| 2024 | Logistic | historical_mean | 3.41499 | 1.60785 | 1.80714 |
| 2024 | Logistic | historical_median | 3.31225 | 1.60785 | 1.70439 |
| 2024 | Logistic | previous_year_median | 0.30219 | 1.60785 | -1.30567 |
| 2024 | Logistic | Ridge | 6.38999 | 1.60785 | 4.78214 |
| 2024 | Logistic | ElasticNet | 6.27057 | 1.60785 | 4.66272 |
| 2024 | Logistic | HGB | 4.45193 | 1.60785 | 2.84408 |
| 2025 | historical_global | historical_mean | 3.11833 | 3.30088 | -0.18255 |
| 2025 | historical_global | historical_median | 2.96933 | 3.30088 | -0.33155 |
| 2025 | historical_global | previous_year_median | 0.97221 | 3.30088 | -2.32867 |
| 2025 | historical_global | Ridge | 3.88794 | 3.30088 | 0.58707 |
| 2025 | historical_global | ElasticNet | 3.97009 | 3.30088 | 0.66921 |
| 2025 | historical_global | HGB | 2.37280 | 3.30088 | -0.92807 |
| 2025 | Logistic | historical_mean | 3.15359 | 3.30088 | -0.14729 |
| 2025 | Logistic | historical_median | 3.00290 | 3.30088 | -0.29798 |
| 2025 | Logistic | previous_year_median | 0.98320 | 3.30088 | -2.31768 |
| 2025 | Logistic | Ridge | 3.75194 | 3.30088 | 0.45106 |
| 2025 | Logistic | ElasticNet | 3.86915 | 3.30088 | 0.56828 |
| 2025 | Logistic | HGB | 2.20902 | 3.30088 | -1.09186 |

Global + historical_mean reproduit 2,84336 / 3,37059 / 3,11833 %, références réalisées 2,03610 / 1,60785 / 3,30088 %. La nouvelle médiane Y−1 utilise les cohortes positives qualifiées, contrairement au comparateur antérieur calculé sur toutes les transitions ; cette divergence est une différence de population explicitée, pas une modification de la target/P1.

## Interprétation et dérive 2025

Coefficients numériques standardisés et catégories one-hot affichés par année ; les variables de localisation sont partiellement redondantes. Les coefficients ne sont ni causaux ni des contributions en dollars. Un coefficient Logistic positif est associé à une probabilité plus élevée dans le modèle. Les coefficients de calendrier extrapolent une année absente de train et doivent être surveillés. La permutation HGB est exclusivement calculée sur les folds test positifs, trois répétitions, scoring MAE ; elle reste descriptive, sans sélection de features ni réglages futurs automatiques.

En 2025, Logistic prévoit 92,35 % de transitions à Saint-Elzear contre 69,91 % réalisées : il manque la concentration des absences. Il sous-estime en parallèle plusieurs bâtiments où les taux restent élevés. Le baseline global surestime le taux total de 7,89 points ; Logistic réduit l'écart à −2,45 mais dégrade le Brier. Aucun modèle n'est déclaré meilleur sur la seule explication rétrospective de Saint-Elzear.

## Reproductibilité et validation

`evaluate_model_a(leases)` retourne uniquement des agrégats en mémoire : supports, scores, coefficients, calibration, permutation et P1. Aucun CSV, objet modèle persisté ou prédiction individuelle exportée. Le notebook 07 exécute les gardes foundation-v1 et les empreintes des données, du contrat et des travaux précédents ; il reproduit explicitement l'ancienne baseline P1.

Tests synthétiques : imputation/encodage train uniquement, catégories inconnues, colonnes vides, 2026 refusé, années train antérieures, labels hors X, CORE uniquement, croissance sur positives, support insuffisant, injection de transitions futures, probabilités bornées, déterminisme, métriques et P1.

**GO pour A7–A9 sous les réserves fondation, CORE uniquement.** Conserver les baselines comme références ; aucun vainqueur P1 définitivement choisi, aucune nouvelle autorisation d'ENRICHED.
