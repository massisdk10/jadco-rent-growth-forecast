# Expérience D : occurrence A et croissance bâtiment → global

## Protocole fixé avant calcul de D

**Alpha = 30**, sans grille, sélection de fenêtre, poids optimisé ou changement après observation des résultats. Il représente 30 observations équivalentes de rappel vers le global : un bâtiment avec 30 observations reçoit un poids de 50 %, avec 10 un poids de 25 %. Ce choix de régularisation est plus conservateur que les defaults 10 proposés précédemment ; il répond à l'exigence explicite de cette expérience et ne dépend d'aucun score de D.

Population d'entraînement : **exactement les unités/années positives qualifiées de A**, pour isoler la seule modification de croissance. « Positive » signifie `occurrence_target=1`, **pas croissance supérieure à zéro** : les croissances négatives restent incluses. Chaque observation est la moyenne des transitions admissibles et mûres de cette unité/année, conformément au contrat P1. Aucune transition future ou résultat immature n'est ajouté. Cette population diffère du train par transition de B, comme documenté dans l'audit précédent.

`g_global = médiane(unit_growth)` sur cet historique. Pour chaque bâtiment : `g_b = médiane(unit_growth_b)`, `n_b = effectif qualifié`, `w_b=n_b/(n_b+30)`, `g_b_shrunk=w_b*g_b+(1−w_b)*g_global`. Bâtiment sans historique, absent ou manquant : **repli global**, sans autre niveau de hiérarchie et sans imputation de labels.

Les bâtiments de train sont reconstruits à leurs propres origines historiques par les helpers figés. Les candidates sont celles de A aux origines 2022-12-31, 2023-12-31, 2024-12-31. `p_i` est le vecteur exact du champion A, inchangé. `P1_D=100*médiane(p_i*g_i)` ; fractions en mémoire, forecasts en %, erreurs en pp. Les trois prédictions D sont fixées avant révélation de leurs labels test.

Diagnostics uniquement agrégés : supports, médianes, shrinkage et parts candidates par bâtiment. Les statistiques de croissance d'un historique de moins de cinq observations sont masquées dans les sorties pour préserver la confidentialité ; cela ne change pas l'estimateur. Aucun identifiant d'unité n'est exporté. La part du bâtiment est une part de **cohorte**, pas une contribution additive au P1 global, car la médiane n'est pas additive.

La sécurité reste conditionnelle à la fidélité historique CRM, sans journal de versions. Aucun asking, ancien effectif courant, gap, statut futur de renouvellement ou externe non qualifié n'est ajouté comme feature. Seuls les backtests 2023/2024/2025 seront exécutés. **Aucun forecast 2026**, aucune architecture adaptative, aucune sélection finale de champion.


## Résultats des trois backtests

A/B et 50/50 sont recalculés par l'évaluateur commun sans changement. Les références A sont reproduites. Les cohortes et les vecteurs réalisés de D sont ceux de A ; les probabilités d'occurrence sont exactement identiques, contrôlées avant scoring. Les neuf nouveaux tests ciblés ont passé avant ce calcul.

| Année | Méthode | Prévision (%) | Réalisé (%) | Erreur signée (pp) | Erreur absolue (pp) |
| --- | --- | --- | --- | --- | --- |
| 2023 | A | 2,870060 | 2,036102 | 0,833957 | 0,833957 |
| 2023 | Model B — full hierarchical | 2,871314 | 2,036102 | 0,835212 | 0,835212 |
| 2023 | A/B 50-50 | 2,870687 | 2,036102 | 0,834585 | 0,834585 |
| 2023 | D | 2,870927 | 2,036102 | 0,834825 | 0,834825 |
| 2024 | A | 3,269179 | 1,607852 | 1,661326 | 1,661326 |
| 2024 | Model B — full hierarchical | 3,236099 | 1,607852 | 1,628247 | 1,628247 |
| 2024 | A/B 50-50 | 3,252639 | 1,607852 | 1,644787 | 1,644787 |
| 2024 | D | 3,289002 | 1,607852 | 1,681150 | 1,681150 |
| 2025 | A | 2,969328 | 3,300878 | -0,331551 | 0,331551 |
| 2025 | Model B — full hierarchical | 2,924144 | 3,300878 | -0,376735 | 0,376735 |
| 2025 | A/B 50-50 | 2,946736 | 3,300878 | -0,354143 | 0,354143 |
| 2025 | D | 2,969328 | 3,300878 | -0,331551 | 0,331551 |

## Métriques communes

| Méthode | MAE (pp) | RMSE (pp) | Biais (pp) | Pire erreur absolue (pp) |
| --- | --- | --- | --- | --- |
| A | 0,942278 | 1,090171 | 0,721244 | 1,661326 |
| Model B — full hierarchical | 0,946731 | 1,078687 | 0,695575 | 1,628247 |
| A/B 50-50 | 0,944505 | 1,084323 | 0,708410 | 1,644787 |
| D | 0,949175 | 1,100473 | 0,728141 | 1,681150 |

D : MAE **0,949175 pp**, contre A **0,942278** ; pire erreur **1,681150**, contre **1,661326**. RMSE et biais absolu se dégradent aussi. Alpha reste 30 : aucune valeur alternative n'a été testée pour sauver l'expérience.

## Diagnostics des bâtiments

`n_b` compte des unités/années avec occurrence positive et croissance mûre, pas des unités distinctes ni toutes les lignes de baux. `g_b` et `g_b_shrunk` sont en %. Le tiret sans historique n'est pas une médiane inventée : le repli au global est indiqué explicitement. Les supports globaux sont 882 / 1 227 / 1 627 et les cohortes 405 / 645 / 856.

| Année | Bâtiment | Candidates | Part cohorte (%) | g_global (%) | g_b (%) | n_b | w_b | g_b_shrunk (%) | Repli global |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023 | Daniel-Johnson | 111 | 27,407407 | 2,918870 | 3,400080 | 96 | 0,761905 | 3,285506 | False |
| 2023 | Levesque | 81 | 20,000000 | 2,918870 | 2,165960 | 221 | 0,880478 | 2,255949 | False |
| 2023 | Saint-Elzear | 213 | 52,592593 | 2,918870 | 2,919799 | 565 | 0,949580 | 2,919752 | False |
| 2024 | Daniel-Johnson | 125 | 19,379845 | 3,319802 | 3,817923 | 153 | 0,836066 | 3,736264 | False |
| 2024 | Le Carlyle | 127 | 19,689922 | 3,319802 | — | 0 | 0,000000 | 3,319802 | True |
| 2024 | Levesque | 82 | 12,713178 | 3,319802 | 2,712845 | 300 | 0,909091 | 2,768023 | False |
| 2024 | Saint-Elzear | 215 | 33,333333 | 3,319802 | 3,340712 | 774 | 0,962687 | 3,339932 | False |
| 2024 | The Met | 96 | 14,883721 | 3,319802 | — | 0 | 0,000000 | 3,319802 | True |
| 2025 | Daniel-Johnson | 129 | 15,070093 | 3,024079 | 3,314405 | 262 | 0,897260 | 3,284577 | False |
| 2025 | Le Carlyle | 180 | 21,028037 | 3,024079 | — | 0 | 0,000000 | 3,024079 | True |
| 2025 | Levesque | 82 | 9,579439 | 3,024079 | 2,331426 | 382 | 0,927184 | 2,381862 | False |
| 2025 | Saint-Elzear | 216 | 25,233645 | 3,024079 | 3,088620 | 983 | 0,970385 | 3,086708 | False |
| 2025 | The Met | 120 | 14,018692 | 3,024079 | — | 0 | 0,000000 | 3,024079 | True |
| 2025 | Westpark | 129 | 15,070093 | 3,024079 | — | 0 | 0,000000 | 3,024079 | True |

**Mécanisme de composition et de médiane :**

- **2023** : Levesque est en dessous du global, Daniel-Johnson au-dessus. Les 213 candidates Saint-Elzear représentent 52,593 % de la cohorte et occupent la position médiane 203 ; leur croissance shrinkée 2,919752 % est légèrement supérieure au global 2,918870 %. D augmente le forecast de **0,000868 pp**.
- **2024** : les 82 candidates Levesque ont une croissance shrinkée inférieure au global. Le Carlyle et The Met, sans historique qualifié positif, ajoutent 223 candidates au global. Ces 305 candidates ne couvrent pas la position médiane 323 : elle appartient aux 215 Saint-Elzear, à **3,339932 %** contre global **3,319802 %**. D augmente le forecast de **0,019824 pp** et aggrave la surestimation. La composition connue a effectivement changé le forecast, dans le mauvais sens.
- **2025** : Le Carlyle 180, The Met 120 et Westpark 129 sont au repli global, soit **429 candidates (50,117 %)**. Avec les 82 Levesque sous ce niveau, cette masse couvre les deux positions médianes 428/429. La médiane de D est exactement le global de A : **aucun changement de forecast**. Le taux d'occurrence de Saint-Elzear reste celui de A et n'est pas corrigé.

Il ne s'agit pas d'une somme de contributions par bâtiment. Les parts candidates et les rangs expliquent la médiane sans publier d'identifiant ni de contribution individuelle. Aucune médiane réalisée future n'a été utilisée pour la segmentation ou le shrinkage.

## Réponses séparées aux huit questions

1. **D améliore-t-il réellement 2024 ?** Non : erreur absolue **1,681150 pp**, +0,019824 par rapport à A. Le découpage conserve des croissances historiques trop élevées et place Saint-Elzear à la médiane.
2. **D dégrade-t-il 2023 ?** Oui, très légèrement : erreur absolue **0,834825 pp**, contre **0,833957** pour A. Différence descriptive, sans signification statistique démontrée.
3. **D dégrade-t-il 2025 ?** Non : forecast et erreur exactement identiques à A (**2,969328 %**, **−0,331551 pp**). Il ne résout pas non plus l'occurrence de Saint-Elzear.
4. **L'amélioration éventuelle provient-elle de la composition bâtiment ?** Aucune amélioration observée. La composition intervient réellement dans les rangs de la médiane, mais augmente le forecast 2024 ; en 2025 les replis déterminent la médiane. À train, p et g_global constants par rapport à A, la seule différence est l'affectation du g shrinké par bâtiment.
5. **Les résultats justifient-ils la complexité ?** Non : MAE, RMSE, biais absolu et pire erreur sont tous moins bons que A ; aucun des trois backtests n'est amélioré en erreur absolue. Trois années ne permettent pas une conclusion générale, mais ce protocole ne fournit aucun bénéfice observé.
6. **D apporte-t-il une information différente de B ?** Structurellement oui : segmentation bâtiment et train unités/années A, contre province/mois et transitions pour B full. Prédictivement, aucun bénéfice ni complémentarité stable démontré : les erreurs P1 ont encore les mêmes signes (+,+,−), et D reste très proche de A.
7. **Les signaux étaient-ils disponibles au cutoff ?** Selon le contrat temporel, oui : bâtiment du bail connu à l'origine, croissance provenant seulement des labels historiques mûrs. Dates de train strictement antérieures à Y, mêmes helpers que A, aucun futur ajouté. La disponibilité strictement archivée demeure **UNKNOWN**, faute de journal CRM ; qualification **SAFE_WITH_CONDITIONS**, inchangée par rapport à la fondation. Les tests d'invariance aux baux futurs et aux effectifs immatures ne lèvent pas cette réserve de source.
8. **Existe-t-il un risque de sélection/tuning sur trois années ?** Oui : l'hypothèse de bâtiment a été formulée après l'analyse des mêmes années. Alpha 30 est fixé avant résultats D, sans grille ni sélection de poids, mais cette phase n'est donc pas un holdout indépendant. Rechercher un autre alpha après cet échec serait du tuning sur ces années ; ce n'est pas fait.

## Recommandation et arrêt

**REJECT_D_RETURN_TO_A_B.** Rejeter cette variante fixe pour le design final : elle n'améliore aucune année, dégrade 2024 et ne traite pas le risque d'occurrence 2025. Ne pas changer alpha, ajouter une hiérarchie ou construire un ensemble adaptatif pour corriger les résultats de cette expérience. Ce rejet ne prouve pas que toute segmentation bâtiment est inutile ; il réfute le bénéfice attendu de la variante précise testée sur ces trois folds.

Cette recommandation **n'est pas la sélection du modèle final**. Aucun forecast 2026, aucun système adaptatif, aucune modification de A/B, de la fondation ou des huit fichiers d'audit précédent. Aucun CSV modifié, aucun export de dossiers CRM, aucun commit ou push.

## Validation finale

Tests ciblés : **9/9** réussis avant calcul des résultats. Suite complète : **180 tests réussis**, comprenant les 171 tests préexistants ; 10 warnings non bloquants (9 sklearn et 1 joblib de détection des processeurs). Notebook 12 exécuté intégralement sous Python 3.11.17, quatre cellules de code, aucune erreur.

Comparaison SHA-256 de tous les fichiers préexistants protégés : aucune modification, y compris A/B, fondation, huit livrables antérieurs et CSV. Aucun nouveau CSV hors `data/raw/` ; le seul CSV extérieur reste le fichier public externe préexistant, intact. Les sorties du notebook sont agrégées et sans identifiant d'unité/bail.

Cinq fichiers créés uniquement : `src/models/model_d.py`, `src/evaluation/model_d_evaluation.py`, `tests/test_model_d.py`, `notebooks/12_model_d_experiment.ipynb`, `docs/model_d_experiment.md`. Aucun autre fichier créé pour cette expérience.
