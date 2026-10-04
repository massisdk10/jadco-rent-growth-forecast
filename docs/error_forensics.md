# Analyse des erreurs 2023, 2024 et 2025

L'erreur commune A/B est surtout une erreur de croissance conditionnelle en 2023–2024. En 2025, une forte erreur d'occurrence coexiste avec une sous-estimation de croissance : leurs effets se compensent partiellement dans P1. Cette analyse est descriptive et ne prouve aucune causalité. Les chiffres sont des agrégats de l'extrait, jamais du portefeuille réel.

Le tableau principal utilise **Model B — full hierarchical**, `P1_G1`, sans sélection de champion. Les quatre combinaisons et leurs métriques sont dans [la comparaison A/B](model_ab_comparison.md). Les notebooks 10/11 recalculent les résultats sans modifier les modèles ni exporter de lignes CRM.

## Méthode et disponibilité

Les diagnostics de composition et d'historique utilisent les baux signés et commencés à D, les anciennes cohortes reconstruites à leur propre origine, et les labels d'entraînement mûrs à D. Les comparaisons réalisées sont construites **après** fixation des forecasts. Les marginales sont séparées pour éviter de décrire des dossiers ; les statistiques numériques sont masquées sous cinq observations. Les modalités rares sont regroupées.

`PREDICTABLE_AT_CUTOFF` exige une disponibilité démontrée. Le calendrier déterministe satisfait ce critère. Pour la composition, les échéances et les historiques CRM, le filtrage daté est vérifié mais les versions archivées ne le sont pas : la classification stricte est **UNKNOWN**, avec usage conditionnel déjà autorisé par le contrat. Cela ne modifie pas les décisions méthodologiques existantes ; cela conserve leur réserve explicite. `RETROSPECTIVE_ONLY` ne peut jamais devenir une feature.

## Composition connue selon le contrat temporel

| Y | Candidates | Québec / Ontario | Superficie médiane (pi²) | Étage médian | Ancien loyer contractuel médian ($) | Terme médian (mois) |
|---|---:|---|---:|---:|---:|---:|
| 2023 | 405 | 405 / 0 | 1 165 | 10 | 1 930 | 12 |
| 2024 | 645 | 549 / 96 | 1 105 | 9 | 2 100 | 12 |
| 2025 | 856 | 736 / 120 | 1 050 | 7 | 2 215 | 12 |

Le notebook détaille aussi chambres, types d'unité, mois d'échéance, quartiles, valeurs manquantes et temps écoulé depuis le bail connu. Ces cinq caractéristiques numériques sont renseignées pour toutes les candidates des trois années ; cela ne prouve pas leur archivage historique.

La distance de variation totale entre parts de bâtiments successives vaut **0,123 / 0,346 / 0,164**. En 2024, Le Carlyle (127 candidates) et The Met (96) entrent dans la cohorte à échéance et représentent **34,574 %** du total ; en 2025 Westpark y entre avec 129 candidates. Leurs premières dates de début présentes dans l'extrait connu sont respectivement 2023, 2023 et 2024 : ce n'est pas une preuve indépendante de leur date d'ouverture.

## Marginales réalisées par bâtiment

Ces chiffres sont **RETROSPECTIVE_ONLY**. Le « P1 bâtiment » est la médiane des contributions de ce sous-groupe, pas sa contribution additive au P1 global.

| Y | Bâtiment | Candidates | Part (%) | Non-transitions | Taux réalisé (%) | Croissance conditionnelle médiane (%) | P1 bâtiment (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| 2023 | Daniel-Johnson | 111 | 27,407 | 1 | 99,099 | 2,331 | 2,300 |
| 2023 | Levesque | 81 | 20,000 | 0 | 100,000 | 2,097 | 2,097 |
| 2023 | Saint-Elzear | 213 | 52,593 | 3 | 98,592 | 1,969 | 1,879 |
| 2024 | Le Carlyle | 127 | 19,690 | 3 | 97,638 | 1,540 | 1,473 |
| 2024 | Daniel-Johnson | 125 | 19,380 | 2 | 98,400 | 1,939 | 1,738 |
| 2024 | Levesque | 82 | 12,713 | 0 | 100,000 | 2,487 | 2,487 |
| 2024 | The Met | 96 | 14,884 | 0 | 100,000 | 1,745 | 1,745 |
| 2024 | Saint-Elzear | 215 | 33,333 | 6 | 97,209 | 1,457 | 1,303 |
| 2025 | Le Carlyle | 180 | 21,028 | 4 | 97,778 | 3,794 | 3,677 |
| 2025 | Daniel-Johnson | 129 | 15,070 | 3 | 97,674 | 3,884 | 3,820 |
| 2025 | Levesque | 82 | 9,579 | 5 | 93,902 | 3,187 | 3,030 |
| 2025 | The Met | 120 | 14,019 | 2 | 98,333 | 3,388 | 3,366 |
| 2025 | Saint-Elzear | 216 | 25,234 | 65 | 69,907 | 4,196 | 0,913 |
| 2025 | Westpark | 129 | 15,070 | 4 | 96,899 | 3,682 | 3,639 |

Le notebook calcule une sensibilité « retirer un bâtiment puis recalculer la médiane ». Elle n'est ni additive ni causale. Les erreurs de **nombre attendu de transitions** sont, elles, additives par bâtiment, car elles portent sur des sommes de probabilités.

## 2023 : croissance historique trop élevée pour l'année réalisée

A prévoit **2,870060 %**, B **2,871314 %**, contre **2,036102 %** réalisés. Le taux réalisé est **99,012 %**, légèrement supérieur à A 98,328 % et B 98,554 %. L'occurrence ne justifie donc pas la surestimation de P1.

La croissance conditionnelle réalisée médiane est **2,085372 %**, contre A **2,918870 %**, B **2,881567 %**. Saint-Elzear représente 52,593 % de la cohorte avec une médiane réalisée de 1,969 %. Les trois bâtiments ont une croissance conditionnelle réalisée inférieure aux estimations globales. À D, les médianes qualifiées par bâtiment sont Daniel-Johnson 3,400 %, Levesque 2,166 %, Saint-Elzear 2,920 % : une hétérogénéité historique est visible selon le contrat, mais la baisse future n'est pas démontrée prévisible.

Les variantes G0, fondées sur seulement **7 transitions Y−1 mûres**, sont plus proches en 2023. Ce faible support ne suffit pas à conclure que leur récence est un signal fiable.

## 2024 : diagnostic détaillé de la pire année

**1. Pourquoi P1 est-il bas ?** La médiane conditionnelle de croissance effective des unités positives est déjà faible : **1,699619 %**. Les 11 non-transitions et la distribution complète des contributions donnent ensuite P1 **1,607852 %**. P1 n'est pas `taux réalisé × médiane conditionnelle`.

**2. Quelle composante explique principalement l'erreur ?** A estime la croissance à **3,319802 %**, B à **3,244485 %**, alors que leurs taux d'occurrence moyens 98,475 % et 98,478 % sont proches des 98,295 % réalisés. Les erreurs de médiane conditionnelle sont **+1,620183 / +1,544866 pp**. Remplacer ex post les probabilités par les occurrences réalisées, tout en conservant la croissance prévue, donne encore **3,319802 / 3,244485 %** : l'erreur ne disparaît pas. Cette substitution est un diagnostic interdit comme forecast.

Pour A, une identité descriptive clarifie l'ordre de grandeur : `1,620183` (écart des médianes conditionnelles) `+ 0,091767` (médiane conditionnelle réalisée moins P1 réalisé) `− 0,050623` (réduction liée à p historique) `= 1,661326 pp`. Ce n'est pas une décomposition causale ni une attribution additive aux bâtiments.

**3. Quels bâtiments/segments sont associés au niveau bas ?** Saint-Elzear (33,333 % de la cohorte) et Le Carlyle (19,690 %) ont des P1 de sous-groupes **1,302539 / 1,473402 %**. Sans eux, séparément, P1 serait **1,699619 / 1,708589 %**, soit seulement +0,091767 / +0,100737 pp. Le niveau bas ne se réduit donc pas à un seul bâtiment. Levesque a la médiane la plus haute, 2,487170 % ; le retirer abaisse P1 à 1,501974 %. The Met est à 1,745485 %, proche du niveau global.

Les unités de deux chambres représentent **395/645** candidates, avec une croissance conditionnelle de **1,685771 %** ; celles d'une chambre **197/645**, à **1,760051 %**. Le petit groupe de trois chambres (13 candidates, 12 transitions) est à −0,272503 %, sans preuve qu'il pilote la médiane globale. Les échéances de mars (40 candidates) sont associées ex post à −1,195789 % ; leur mois était à cutoff, **leur croissance future ne l'était pas**. Ces marginales se recouvrent et ne sont pas des effets indépendants.

**4. Était-ce visible au 2023-12-31 ?** Le changement de composition était visible selon le contrat : nouvelles cohortes Le Carlyle/The Met, part de Saint-Elzear réduite de 52,593 % à 33,333 %. Mais aucune occurrence historique de ces nouveaux bâtiments n'était qualifiée. Les anciennes médianes de croissance mûre par bâtiment restaient **3,818 % Daniel-Johnson, 2,713 % Levesque, 3,341 % Saint-Elzear** : elles n'annoncent pas clairement la faiblesse 2024. Le train de croissance Y−1 mûr a une médiane **1,133244 %**, mais seulement **13 transitions** et un biais potentiel de sélection des baux terminés tôt. Les quatre unités/années positives 2023 qualifiées à D sont trop peu nombreuses pour publier une médiane descriptive fiable.

**5–6. B capte-t-il mieux cette information, et par quelle logique ?** Son erreur **+1,628247 pp** améliore A de seulement **0,033080 pp**. Il ne segmente pas les bâtiments. Sa médiane de croissance provinciale vient d'une autre population de train (toutes transitions mûres), avec repli global pour l'Ontario non appris. Cette médiane est un peu plus basse ; cela ne démontre pas un effet prédictif des nouveaux immeubles. Avec croissance B et occurrence globale (`P0_G1`), P1 serait 3,195010 % ; ajouter la hiérarchie d'occurrence le remonte à 3,236099 % et **augmente** l'erreur 2024 de 0,041089 pp.

**7. Quel signal aurait potentiellement aidé ?** La statistique récente Y−1 mûre est un avertissement disponible selon le contrat, mais son support 13 et sa maturité sélective empêchent de la généraliser. Une croissance conditionnelle par bâtiment, avec repli et support explicites, est une hypothèse structurelle testable pour exploiter l'hétérogénéité et les poids connus ; **elle pourrait aussi échouer**, puisque les historiques des grands bâtiments sont élevés. Ni les loyers futurs faibles ni les publications externes postérieures à D ne constituent une solution leakage-safe.

## 2025 : forte dérive d'occurrence, compensée en partie par la croissance

A prévoit **2,969328 %**, B **2,924144 %**, contre **3,300878 %** réalisés. La croissance conditionnelle réalisée **3,716980 %** dépasse A **3,024079 %** et B **2,931619 %**. Le taux d'occurrence réalisé chute à **90,303738 %**, contre A **98,189499 %**, B **97,814598 %**. Les bons scores P1 relatifs cachent donc deux erreurs de composantes opposées : surestimation d'occurrence et sous-estimation de croissance. On ne doit pas les présenter comme deux prédictions individuellement bien calibrées.

**Saint-Elzear : 65 des 83 non-transitions**, soit **78,313 %**, sur **216 candidates** (25,234 % du total). Son taux réalisé est **69,907407 %**, contre A **98,189499 %** et B **98,277070 %**. B ne détecte pas mieux ce risque local, même si son Brier global est légèrement meilleur. L'erreur attendue de nombre de transitions à Saint-Elzear est **+61,089 / +61,278** pour A/B.

À D, Saint-Elzear dispose de **1 003 labels qualifiés**, **216 en attente**, et d'un taux qualifié historique de **98,005982 %**. La cohorte passe seulement de **215 à 216** candidates. Ni ce taux historique ni ce nombre stable ne démontrent la chute future à 69,9 %. L'historique qualifié de croissance y est 3,088620 %, contre 4,195614 % réalisés parmi les unités positives. La concentration des absences est donc un constat **rétrospectif**, pas une feature autorisée en 2025.

Le P1 de Saint-Elzear vaut 0,912654 %, mais retirer ce bâtiment donne un P1 global de **3,481768 %**, seulement +0,180889 pp. Cette sensibilité ne mesure pas une contribution additive. Les autres bâtiments ont des médianes conditionnelles réalisées de 3,187 % à 3,884 %, au-dessus de l'estimation A. Westpark entre dans la cohorte avec 129 candidates et n'a pas d'historique qualifié d'occurrence dans le train. Le Carlyle n'a que **3 labels d'occurrence qualifiés / 124 en attente**, The Met **0 / 96** : les nouveaux segments sont surtout des cas de repli, pas des preuves d'apprentissage hiérarchique.

Les 83 absences sont celles de l'extrait se terminant en 2025 : transition retardée ou omise, complétude et convention source restent à investiguer. Il ne s'agit pas d'une mesure de vacance ou d'une preuve de non-renouvellement réel.

## Économie locative mûre à cutoff, sans nouvelle feature

Seuls les baux signés, commencés **et terminés** à D sont examinés. Gap individuel descriptif : `100*(sRent−sRentEffective)/sRent`, pour loyers sources positifs et finis ; les gaps sont ensuite agrégés. Les médianes des deux loyers ne sont pas divisées pour fabriquer une médiane de gap. `sConcession` est uniquement un indicateur source 0/1 : il ne mesure ni montant, ni récurrence, ni encaissement. Aucun PromoPay n'est étalé et aucun ancien effectif/gap n'est réintroduit dans X.

À D=2023-12-31, les **43 baux mûrs commençant en 2023** ont un gap médian **5,752 %** et un flag concession présent pour **67,442 %** ; à D=2024-12-31, les **41 baux mûrs commençant en 2024** ont **7,236 % / 80,488 %**. Les supports sont faibles et sélectionnés par fin précoce. Dans le stock de baux mûrs à cette dernière coupure, les gaps médians sont Le Carlyle **5,790 %**, The Met **5,839 %**, Westpark **7,711 %** (10 baux seulement), Saint-Elzear **2,748 %**.

Ce contexte est compatible avec des économies locatives différentes, mais ne prouve pas la cause des erreurs P1 ni une disponibilité historique fiable de chaque effectif. Classification **UNKNOWN**, conformément à l'exclusion de ces features dans la fondation. Les séries asking rétrospectives ne possèdent pas de qualification des versions à cutoff et ne sont pas utilisées pour expliquer prédictivement ces erreurs.

## Récence, dispersion et sélection des labels

| Y prédit | Transitions mûres commençant en Y−1 | Médiane (%) | Q25 / Q75 (%) |
|---|---:|---:|---|
| 2023 | 7 | 2,264068 | −0,571958 / 6,561011 |
| 2024 | 13 | 1,133244 | −0,865529 / 2,470848 |
| 2025 | 28 | 2,240621 | −1,555474 / 4,209603 |

Les trains complets de transitions mûres ont des médianes 2,894218 / 3,255591 / 2,948542 %. Les fenêtres de trois années de début mûres donnent 3,301403 / 3,996277 / 3,013192 %, donc un simple raccourcissement à trois ans n'annonce **pas** la faiblesse 2024. Les labels en attente 351 / 407 / 641 et les supports récents faibles empêchent d'assimiler cette récence à la tendance de toutes les unités. Les changements réalisés de 2024 à 2025 sont des changements de distribution observés ; « changement de régime causal » n'est pas identifié.

## Couverture et exclusion de fausses absences

| Y | Paires annuelles dans la cohorte | Paires admissibles | Unités positives | Transitions supplémentaires au-delà d'une par unité |
|---|---:|---:|---:|---:|
| 2023 | 404 | 404 | 401 | 3 |
| 2024 | 636 | 636 | 634 | 2 |
| 2025 | 779 | 779 | 773 | 6 |

Aucune paire annuelle présente dans ces cohortes n'est exclue pour inéligibilité mathématique principale. Cela distingue les absences de transition présentes dans l'extrait d'un effet des exclusions, sans prouver sa complétude. Le regroupement des paires admissibles en moyenne par unité est conservé pour tous les scores P1.

## Tableau des explications et de leur disponibilité

Le notebook contient le tableau complet `signal / year / observed_effect / availability_class / evidence / potential_model_use` pour chaque année. Les principaux cas sont :

| Signal | Années | Effet observé | Classe | Preuve / utilisation potentielle |
|---|---|---|---|---|
| Calendrier et cutoff | 2023–2025 | Origines au 31 décembre Y−1 | PREDICTABLE_AT_CUTOFF | Calendrier déterministe, sans futur |
| Cohorte, caractéristiques, échéances | 2023–2025 | Mix connu reconstruit, notamment +223 candidates de nouveaux segments en 2024 | UNKNOWN | Dates filtrées ; fidélité des versions CRM à confirmer. Usage conditionnel autorisé par contrat |
| Croissance historique mûre / récence | 2023–2025 | Différences de niveaux et faible support récent | UNKNOWN | Maturité contrôlée ; disponibilité source non archivée. Pas de qualification stricte inventée |
| Gap effectif et concessions mûres | 2023–2025 | Écarts par année/bâtiment | UNKNOWN | Hors features autorisées ; diagnostic seulement |
| Croissance effective de Y | 2023–2025 | Faiblesse 2024, hausse 2025 | RETROSPECTIVE_ONLY | Nouveau bail et intervalle observés après D ; aucune feature |
| Taux de transition de Y | 2023–2025 | Chute 2025 à 90,304 % | RETROSPECTIVE_ONLY | Labels post-prédiction ; aucune feature |
| Absences Saint-Elzear | 2025 | 65/83, sans avertissement historique démontré | RETROSPECTIVE_ONLY | Aucun encodage de cette concentration pour prédire 2025 |
| Statut futur renouvellement/relocation | 2023–2025 | Non exploité par les modèles | RETROSPECTIVE_ONLY | `sRenewal` du nouveau bail interdit en prédiction |

## Signaux externes déjà présents

Aucune collecte, recherche web, interpolation ou jointure supplémentaire. Le filtre existant exige valeur présente, booléen strict `availability_verified=True`, date vérifiée `available_date<=D` : **zéro ligne admissible pour les trois backtests**.

| Famille présente | Classe | Justification |
|---|---|---|
| SCHL : loyers, croissance, inoccupation, rotation / sans rotation | NOT_HISTORICALLY_QUALIFIED | Dates/versions des valeurs non qualifiées ; univers différent de P1 JADCO |
| Statistique Canada : IPC loyers annuel | NOT_HISTORICALLY_QUALIFIED | Publications mensuelles et versions non reconstituées ; année complète Y inconnue à D |
| Statistique Canada : IPC partiel 2026 | NOT_HISTORICALLY_QUALIFIED | Version non qualifiée et période postérieure aux origines étudiées |
| Ontario guideline | NOT_HISTORICALLY_QUALIFIED | Dates d'annonces non qualifiées ; applicabilité The Met à confirmer |
| Neuf composantes TAL legacy 2025 | SAFE_BUT_WEAK | Publication 2025-01-21, donc admissibles seulement à l'origine 2026 parmi celles étudiées ; pas de série utile aux backtests et applicabilité non acquise |
| Trois nouveaux paramètres TAL 2026 | NOT_RELEVANT | Publication vérifiée 2026-01-19, postérieure même à l'origine 2026 ; exclusion de ce périmètre, pas preuve d'inutilité générale |
| Anciennes composantes TAL 2026 | NOT_HISTORICALLY_QUALIFIED | Disponibilité non qualifiée ; rupture méthodologique à conserver |

**Aucun signal n'est SAFE_AND_USEFUL sur preuve de ces backtests.** Le tableau détaillé du notebook est par nom de signal ; une composante legacy contenant 2025 et 2026 distingue ses lignes admissibles de ses lignes exclues. Les paramètres TAL ne sont pas un plafond universel, ni une règle transférable à Ottawa ; les niveaux SCHL ne sont pas des taux de croissance same-unit.

## Trois hypothèses d'amélioration, aucune implémentée

| Candidat | Intuition / signal | Disponibilité et leakage | Overfit / complexité | Bénéfice attendu, sans promesse | Test falsifiable |
|---|---|---|---|---|---|
| 1. Croissance bâtiment → global sur les unités/années positives P1 | Hétérogénéité historique et changement des parts candidates ; comparer à population de train constante | Bâtiment historique et labels mûrs, sous réserve CRM ; jamais croissance ni poids futurs | Modéré ; shrinkage/support 10/10 déjà defaults existants, pas de grille ; faible complexité | Tester l'ajustement à la composition plutôt que confondre segmentation et changement de population A/B | Garder p de A, refaire les trois folds ; rejeter si pire erreur ou biais absolu augmente, ou si le gain est seulement 2024 et dégrade 2025 |
| 2. Occurrence bâtiment → global avec support / labels en attente explicités | Concentration locale ex post et supports inégaux | Seulement taux historiques qualifiés à D ; 65 absences futures jamais feature | Modéré ; petit nombre de bâtiments et censure ; faible complexité | Tester une calibration locale ; ne pas promettre de prévoir la rupture Saint-Elzear, son taux antérieur était 98 % | Rejeter si le Brier et la calibration locale ne s'améliorent pas sans aggravation du P1 ; signal historique insuffisant = repli, pas correction imposée |
| 3. Croissance récente mûre rétrécie vers l'historique global | Signal Y−1 faible en 2024, mais supports 7/13/28 sélectionnés | Labels mûrs uniquement, aucun effectif courant/asking non qualifié ; réserve CRM | Élevé : sélection des baux courts, seulement trois années ; complexité modérée | Tester une adaptation temporelle prudente, plutôt qu'extrapoler automatiquement une petite médiane récente | Protocole/support fixés avant test ; rejeter si le biais de maturité ou la dégradation 2025 domine le gain 2024 ; aucune recherche de fenêtre ou de poids sur ces folds |

La hiérarchie ne saurait transformer l'absence de preuve d'un avertissement en 2025 en signal prédictif. Les trois hypothèses requièrent le même contrat et une validation des sources ; elles ne sont pas implémentées pendant cette phase.

## Une seule prochaine action

**D — tester le candidat 1 : croissance conditionnelle par bâtiment avec shrinkage vers le global, entraînée sur les unités/années positives qualifiées P1, en gardant l'occurrence de A.**

Motifs : A reste simple et légèrement meilleur que B full en MAE ; B améliore faiblement RMSE/pire année mais sans complémentarité P1 ni résolution de 2024/2025 ; le 50/50 ne résout rien de structurel. P1_G0 gagne les métriques agrégées, mais manque nettement la croissance 2025 et repose sur très peu de labels récents. Tester une seule hypothèse explicable isole la segmentation de croissance de la différence de train A/B, sans sélectionner un champion final sur trois observations.

La cohorte 2026 reste **931**, Québec/Ontario **810/121**, bâtiments Daniel-Johnson **128**, Le Carlyle **180**, Levesque **75**, Saint-Elzear **268**, The Met **121**, Westpark **159**. Elle contient les six segments avec des poids différents ; cela justifie d'évaluer la cohérence d'une méthode de repli et de composition, **pas d'affirmer un gain 2026**. Aucun forecast 2026 n'a servi à la recommandation. Si le test falsifie cette hypothèse, aucune promotion ni correction de modèle ne devra être automatique.
