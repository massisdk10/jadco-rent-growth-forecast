# Évaluation commune des modèles A et B

Les références A sont reproduites et les quatre combinaisons B sont comparables sur les cohortes et labels de l'extrait. **Model B — full hierarchical** désigne `P1_G1`, conformément au choix de représentation demandé ; ce n'est pas un champion. Aucun modèle final n'est sélectionné et aucun forecast 2026 n'est calculé dans cette phase.

La sécurité vérifiée est celle du contrat temporel figé : baux signés et commencés à la coupure, labels historiques mûrs, features reconstruites à chaque origine. Elle reste **conditionnelle à la fidélité des champs historiques CRM**, faute de journal de versions. Il serait incorrect de présenter les tests comme une preuve d'une disponibilité historique parfaitement archivée.

## Contrat mathématique et comparabilité

Les deux modèles estiment l'« indice médian de contribution attendue des transitions à la croissance annualisée du loyer effectif des unités à échéance », P1. Les fractions sont manipulées en mémoire ; les forecasts sont en %, les erreurs signées `prédit − réalisé` en points de pourcentage.

- Clé : `(sPropCode, sUnitCode)`, jamais `(sSite, sUnitCode)`.
- Candidate : dernier bail déjà signé **et** commencé à D, début non ambigu, fin valide en Y. La cohorte est fixée avant observation des transitions de Y.
- Transition : deux baux consécutifs de la même unité selon le moteur commun, sans sauter un bail invalide pour chercher un précédent favorable.
- Cible d'une transition : `(effective_nouveau/effective_ancien) ** (365,25/jours_entre_débuts) − 1`, avec `sRentEffective` fourni. Pas d'annualisation par `sTermMonths`, pas de reconstruction de concessions.
- `O_i=1` si au moins une transition admissible est observée en Y ; `G_i` est la **moyenne** des croissances admissibles de l'unité en Y. Sinon `O_i=0`, contribution réalisée nulle. Ce zéro signifie absence de transition dans l'extrait, pas stagnation du loyer ou vacance.
- `p_i` estime l'occurrence conditionnellement aux informations à D ; `g_i` estime la croissance conditionnelle ; `c_i=p_i*g_i` ; `P1=100*médiane_i(c_i)` sur toute la cohorte.

Les estimateurs de `g_i` utilisent des **médianes**, donc des approximations conventionnelles de croissance conditionnelle ; ils ne démontrent pas une espérance conditionnelle au sens probabiliste. P1 est un indice défini par le contrat, pas une prévision du rent roll ni des encaissements.

| Y | Origine D | Candidates | Unités avec transition | Statut |
|---|---|---:|---:|---|
| 2023 | 2022-12-31 | 405 | 401 | Clés et labels A/B identiques |
| 2024 | 2023-12-31 | 645 | 634 | Clés et labels A/B identiques |
| 2025 | 2024-12-31 | 856 | 773 | Clés et labels A/B identiques |
| 2026 | 2025-12-31 | 931 | Non observé | Cohorte seulement, aucun forecast |

A révèle les labels depuis l'extrait complet ; le tournoi B les révèle depuis les baux connus à fin de Y. Les vecteurs d'occurrence **et** de croissance unitaire sont explicitement comparés : ils concordent ici. Une divergence sur un autre extrait arrête la comparaison et interdit l'ensemble, sans réécrire B.

Différence d'entraînement importante : A estime `g` sur les **unités/années positives des cohortes historiques qualifiées** ; B sur **toutes les transitions historiques admissibles et mûres**, y compris hors des anciennes cohortes à échéance. Les unités ayant plusieurs transitions peuvent peser plusieurs fois dans B. Ce sont des méthodes différentes d'estimation, pas des populations de scoring différentes. Le diagnostic de croissance individuel du tournoi B initial est par transition ; notre diagnostic commun est par unité positive, avec le `G_i` moyen du contrat P1.

## Audit complet du code B et de ses tests

Lecture des cinq fichiers `src/models/model_b/` et des quatre fichiers `tests/model_b/`, ainsi que des helpers de fondation appelés. Les tests vérifient notamment le shrinkage, les replis, l'alignement, les listes fermées, la maturité, l'ordre prédiction/révélation et l'invariance aux baux futurs. L'inspection des chemins de données complète ces tests.

### Occurrence

`HierarchicalOccurrenceEstimator` calcule un taux global moyen sur les labels binaires qualifiés. À chaque niveau : `p_segment=(positifs + λ*p_parent)/(n+λ)`. Le parent est déjà rétréci vers son propre parent. Un segment absent, manquant ou sous le minimum de support se replie vers le parent connu, puis le global. Les probabilités sont bornées dans `[0,1]`.

Pour les combinaisons `P1_*` : **global → province → province × expiry_month**, `λ=20`, minimum **1** label. Les features effectivement utilisées sont `province` et `expiry_month` ; les 13 features autorisées peuvent être transmises, mais les autres ne segmentent pas cet estimateur. Aucun bâtiment ne figure dans cette configuration.

### Croissance conditionnelle

`HierarchicalGrowthEstimator` prend la médiane globale du train. Pour un segment : `g_segment = n/(n+λ)*médiane_segment + λ/(n+λ)*g_parent`. Pour `*_G1` : **global → province**, `λ=1`, minimum **10** transitions. Segment absent/manquant/insuffisant : repli au parent, puis médiane globale. La croissance peut être négative ; elle n'est pas bornée à zéro.

`*_G0` utilise la médiane des transitions commençant en Y−1 **déjà mûres à D**. Aucune médiane future, aucune imputation si elle est indisponible. Les valeurs de repli fictives de certains tests sont limitées à leurs fixtures synthétiques et ne sont pas utilisées en production.

| Variante B figée | Occurrence | Croissance |
|---|---|---|
| P0_G0 | Taux global qualifié | Médiane Y−1 mûre |
| P0_G1 | Taux global qualifié | Hiérarchie provinciale |
| P1_G0 | Hiérarchie province/mois | Médiane Y−1 mûre |
| P1_G1 | Hiérarchie province/mois | Hiérarchie provinciale |

Les defaults de `ModelB()` ne correspondent **pas** aux paramètres de cette shortlist : occurrence provinciale `λ=10`, croissance province/bâtiment `λ=10`, minimum 10. L'évaluation réutilise explicitement les configurations du tournoi P1 existant, pas ces defaults.

Les anciens tournois exploratoires comptent 7 hiérarchies × 6 forces de shrinkage pour chaque composante, plus leurs baselines. Nous **ne les relançons pas pour rechercher des hyperparamètres**. Leur exploration sur les mêmes trois années crée un risque de sélection optimiste : une shortlist figée ne transforme pas ces années en holdout indépendant.

### Features réellement actives et disponibilité

| Feature | Construction | Contrôle à l'origine | Réserve |
|---|---|---|---|
| province | `sState` du dernier bail signé et commencé | Snapshot à D pour candidates ; propre origine historique pour chaque ligne de train | Version historique CRM non archivée |
| expiry_month | Mois de la fin contractuelle du bail candidat connu | `known_lease_end`, jamais fin du nouveau bail de Y | Fidélité historique des échéances à confirmer |

Pas de listings actuels, d'asking, de concessions, d'ancien effectif comme feature, de `sRenewal` futur ou d'externe dans ces combinaisons. Les labels mûrs de croissance restent des cibles historiques, pas une autorisation de rendre l'effectif courant disponible comme prédicteur.

### Québec, Ontario, bâtiments et économie locative

La province segmente statistiquement ; **aucune règle TAL n'est appliquée à Ottawa**, aucun guideline Ontario au Québec. Aucune applicabilité juridique n'est supposée. The Met n'a aucun label d'occurrence qualifié dans les trains de ces folds : repli global. Ses six transitions de croissance provinciales qualifiées pour 2025 sont sous le minimum 10 : repli global aussi. L'absence d'historique ontarien interdit de présenter B comme ayant appris un régime provincial distinct sur ces folds.

Les configurations principales ne segmentent pas les bâtiments ni les chambres, même si ces possibilités existent dans les classes. « Full hierarchical » signifie ici les **deux composantes hiérarchiques figées**, pas tous les niveaux disponibles. Renouvellement/relocation n'est pas prédit séparément ; le futur `sRenewal` est exclu. L'effectif fourni entre dans les labels après maturité, pas une économie reconstruite à partir de PromoPay.

### Pipeline temporel et réserves

Entraînement expansif : année de début antérieure à Y, label admissible, `label_available_date <= D`. Cette date est le maximum des six dates début/signature/fin des deux baux, toutes valides. Les features de chaque exemple de croissance sont reconstruites à la fin de **son** année précédente. Les labels d'occurrence proviennent des anciennes cohortes reconstruites à leurs propres origines ; un positif encore immature ne devient pas automatiquement négatif.

| Y | Occurrence qualifiée | Labels en attente | Croissance A : unités/années | Croissance B : transitions |
|---|---:|---:|---:|---:|
| 2023 | 897 | 351 | 882 | 952 |
| 2024 | 1 246 | 407 | 1 227 | 1 318 |
| 2025 | 1 657 | 641 | 1 627 | 1 759 |

Les features provinciales manquent pour 8 / 10 / 13 transitions historiques B à leur propre origine : elles alimentent la médiane globale, pas un segment provincial inventé. Les retards de maturité créent un risque de sélection des baux courts dans la partie récente du train. Une absence en fin d'extrait peut être une transition retardée ou omise. Ni l'absence de fuite future dans le code ni les tests ne résolvent ces limites de source.

`ModelB.fit` reçoit des labels déjà qualifiés : la sécurité dépend de son appelant. Le tournoi existant utilise les helpers temporels figés ; l'adaptateur commun reproduit ses quatre forecasts à une tolérance de `1e-10` point. Tous ses vecteurs sont fixés avant révélation des labels test. Les familles B2/B3 externes sont déclaratives et `run_experiment` n'est pas implémenté. B ne fournit pas encore les fonctions finales `estimate_2026`/`backtest` du starter ; cette limite d'interface est documentée, sans modification de B dans cette phase.

## Résultats principaux

| Année | P1 réalisé (%) | A (%) | Erreur A (pp) | Model B — full hierarchical (%) | Erreur B (pp) | 50/50 (%) | Erreur 50/50 (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2023 | 2,036102 | 2,870060 | +0,833957 | 2,871314 | +0,835212 | 2,870687 | +0,834585 |
| 2024 | 1,607852 | 3,269179 | +1,661326 | 3,236099 | +1,628247 | 3,252639 | +1,644787 |
| 2025 | 3,300878 | 2,969328 | −0,331551 | 2,924144 | −0,376735 | 2,946736 | −0,354143 |

| Méthode | MAE (pp) | RMSE (pp) | Biais (pp) | Pire erreur absolue (pp) |
|---|---:|---:|---:|---:|
| A | 0,942278 | 1,090171 | +0,721244 | 1,661326 |
| Model B — full hierarchical | 0,946731 | 1,078687 | +0,695575 | 1,628247 |
| A/B 50-50 | 0,944505 | 1,084323 | +0,708410 | 1,644787 |

L'ensemble est **la moyenne des deux forecasts P1 scalaires**. Il n'est pas la médiane d'une moyenne de contributions unitaires : la médiane n'est pas linéaire. Les poids restent 0,5/0,5 ; aucun ajustement sur les trois années.

## Les quatre variantes B

| Variante | P1 2023 (%) | Erreur 2023 (pp) | P1 2024 (%) | Erreur 2024 (pp) | P1 2025 (%) | Erreur 2025 (pp) |
|---|---:|---:|---:|---:|---:|---:|
| P0_G0 | 2,226208 | +0,190105 | 1,115963 | −0,491889 | 2,200054 | −1,100824 |
| P0_G1 | 2,833380 | +0,797278 | 3,195010 | +1,587158 | 2,878542 | −0,422336 |
| P1_G0 | 2,256013 | +0,219911 | 1,130315 | −0,477538 | 2,234907 | −1,065971 |
| P1_G1 | 2,871314 | +0,835212 | 3,236099 | +1,628247 | 2,924144 | −0,376735 |

| Variante | MAE (pp) | RMSE (pp) | Biais (pp) | Pire erreur absolue (pp) |
|---|---:|---:|---:|---:|
| P0_G0 | 0,594273 | 0,704724 | −0,467536 | 1,100824 |
| P0_G1 | 0,935591 | 1,054054 | +0,654033 | 1,587158 |
| P1_G0 | 0,587807 | 0,686221 | −0,441200 | 1,065971 |
| P1_G1 | 0,946731 | 1,078687 | +0,695575 | 1,628247 |

**P1_G0 domine P1_G1 sur les quatre métriques agrégées** (biais en valeur absolue), mais **pas sur chaque année** : son erreur 2025 est presque trois fois plus grande. P0_G0 fait aussi mieux sur ces métriques. La hiérarchie complète ne démontre donc pas un gain sur les variantes simples. Leur meilleur score sur trois observations ne suffit pas à sélectionner un champion final.

## Complémentarité : occurrence, croissance et P1

| Y | Erreur occurrence A/B (pp) | Erreur médiane conditionnelle A/B (pp) | Brier A/B |
|---|---|---|---|
| 2023 | −0,684586 / −0,458536 | +0,833499 / +0,796195 | 0,009826 / 0,008452 |
| 2024 | +0,180547 / +0,183023 | +1,620183 / +1,544866 | 0,016767 / 0,015389 |
| 2025 | +7,885761 / +7,510860 | −0,692901 / −0,785360 | 0,093779 / 0,092031 |

Les médianes conditionnelles sont comparées sur les **mêmes unités positives** ; les taux d'occurrence portent sur toutes les candidates. Le notebook donne aussi les MAE et biais individuels de croissance, distincts de l'erreur de médiane et de l'erreur P1.

**B apporte-t-il réellement une information complémentaire à A ?** La segmentation province/mois apporte une amélioration du Brier sur chaque année, mais aucun changement de signe des erreurs. Les erreurs de croissance et de P1 ont aussi les mêmes signes dans les trois années. Les forecasts P1 A/B diffèrent d'au plus 0,045184 point. La complémentarité prédictive P1 n'est pas démontrée.

**Le 50/50 réduit-il une vraie faiblesse structurelle ou seulement le bruit sur trois observations ?** Il ne résout ni 2024 ni le risque d'occurrence de Saint-Elzéar en 2025. Sa MAE est légèrement supérieure à A et ses autres métriques sont intermédiaires. Trois années ne permettent pas de conclure à une robustesse hors échantillon ; aucune inférence statistique ou optimisation de poids n'est justifiée.

## Prochaine action recommandée

**D — tester une seule amélioration structurelle : croissance conditionnelle par bâtiment avec shrinkage vers le global, sur les unités/années positives qualifiées du contrat P1.** Maintenir l'occurrence globale de A et l'agrégation médiane des `p_i*g_i` sur la cohorte connue. Fixer le protocole avant le test, réutiliser les defaults existants de shrinkage/support 10/10 plutôt que rechercher des paramètres sur ces trois années, et ne pas faire de forecast 2026 pour choisir la variante.

Il s'agit d'une hypothèse à réfuter, pas d'une amélioration acquise : les taux historiques de croissance 2024 étaient déjà élevés, et un découpage bâtiment peut échouer à traiter le changement temporel. Rejeter ce candidat si la pire erreur ou le biais se dégrade, ou si l'amélioration globale repose seulement sur 2024 au prix de 2025. Les détails et les deux autres hypothèses non retenues figurent dans [l'analyse des erreurs](error_forensics.md). Aucun modèle A, B ou C n'est modifié ou construit ici.
