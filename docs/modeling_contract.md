# Contrat commun proposé — Foundation Freeze

## Cible et unité statistique

Une observation est une transition entre deux baux consécutifs de la même unité, obtenue **exclusivement** par `build_same_unit_pairs` ; `UNIT_KEY = (sPropCode, sUnitCode)`. Les anciennes/nouvelles clés et dates restent en mémoire pour traçabilité. Aucun export de lignes CRM. Les dates ambiguës et loyers non positifs ne deviennent pas valides ; les flags de comparabilité restent visibles. Aucun retrait automatique d'extrêmes ni filtre supplémentaire de durée.

**Target recommandée : médiane des croissances same-unit annualisées du loyer effectif fourni.** Pour la transition i débutant en Y :

`Δ_i = (date_nouvelle − date_ancienne).jours / 365,25`

`g_i = (sRentEffective_nouveau / sRentEffective_ancien)^(1/Δ_i) − 1`

`Croissance médiane annualisée du loyer effectif des transitions same-unit observées(Y) = 100 × médiane({g_i : nouveau bail commence en Y, transition mathématiquement valide})`

Les modules manipulent des **fractions** ; le résultat business est en **pourcentage**, les erreurs en **points de pourcentage**. La définition est une statistique des transitions observables dans l'extrait, pas une hausse du rôle total, une variation de revenu encaissé ou une moyenne à poids égal des unités. Une unité ayant plusieurs transitions compte plusieurs fois.

Quatre cibles sont conservées : `target_contractual_raw`, `target_effective_raw`, `target_contractual_annualized`, `target_effective_annualized`. Les ratios bruts sont `nouveau/ancien − 1`. Le contractuel reste un diagnostic/target secondaire obligatoire ; aucune conversion en effectif et aucune reconstruction de concessions ne remplace le champ source. L'annualisation utilise l'intervalle **entre débuts**, jamais `sTermMonths`. Elle suppose un rythme composé équivalent, pas une trajectoire mensuelle observée.

La recommandation est économique (effet net selon la convention source) et comparable à l'annualisation du starter ; elle n'est pas sélectionnée sur la facilité de prédiction. Le notebook compare couverture, dispersion, années, termes atypiques et fenêtre starter. Les 62 intervalles hors fenêtre restent dans le périmètre principal ; la fenêtre stricte `0,5 < Δ < 2,5` fournit seulement une sensibilité. Les trois transitions 2017 sont comptées mais leurs valeurs ne sont pas affichées.

## Agrégation et segmentation

La médiane de toutes les transitions est la règle commune principale, robuste aux extrêmes et sans poids extrapolé du portefeuille réel. Chaque transition a le même poids. Comparaisons descriptives : moyenne des transitions, moyenne des médianes par bâtiment, moyenne des médianes de bâtiment pondérée par unités distinctes connues à la coupure, et médianes provinciales pondérées par unités distinctes connues. Les poids sont calculés uniquement sur les baux commencés et signés au plus tard à l'origine, jamais sur les listings actuels ni les transitions futures. Ils décrivent l'extrait, pas le stock réel.

Pour les variantes pondérées : `Σ_b w_b médiane(g_i dans b) / Σ_b w_b`, somme sur les segments présents et pondérés ; la masse couverte et les transitions sans poids sont rapportées. Un nouveau bâtiment absent à la coupure n'obtient pas un poids inventé. La moyenne égale par bâtiment donne beaucoup d'influence aux bâtiments avec peu de transitions et change fortement le résultat 2023 ; elle n'est pas retenue. Les variantes pondérées restent des sensibilités, sans les confondre avec la cible principale.

Diagnostics obligatoires : renouvellement/relocation (`sRenewal` nouveau, connu **après coupure**, utilisable pour évaluation seulement), Québec/Ontario, bâtiment, code de propriété, année. Pas de segmentation prédictive par statut futur de renouvellement. Les comparaisons same-unit évitent l'effet du niveau de loyer des nouveaux immeubles, mais **n'éliminent pas les changements de fréquence des transitions et de mix renouvellement/relocation**. La médiane globale n'est pas une médiane à mix fixe. Les futurs modèles devront documenter leurs hypothèses de composition, sans utiliser la composition future réalisée. Une cible à composition fixe nécessiterait une autre question business explicitement approuvée avant modification du contrat.

## Politique des features et audit du leakage

Liste fermée de **12 features candidates admissibles** : `property_code`, `building`, `province`, `bedrooms`, `bathrooms`, `sqft`, `floor`, `unit_subtype`, `prior_contractual_rent`, `prior_term_months`, `forecast_year`, `months_since_known_start`.

`KNOWN_AT_PREDICTION=True` signifie : attribut du **dernier bail déjà commencé ET signé** à l'origine de l'année prédite, ou calendrier de l'origine. Les champs du bail immédiatement précédent réel ne sont pas utilisés s'il commence après cette origine. Les caractéristiques viennent du bail historique et non du nouveau bail ou des listings 2025. Les unités inconnues restent manquantes ; le modèle doit gérer cette absence sans imputation future. Les débuts égaux rendent le snapshot ambigu ; il est écarté. Les transformations apprises seront ajustées sur train seul.

Cette qualification est **conditionnelle à la fidélité des champs historiques CRM** : le fichier est un extrait rétrospectif sans journal de versions. Les dates de signature/début ne prouvent pas que toutes les caractéristiques n'ont jamais été corrigées ensuite. Les règles implémentées empêchent le recours aux baux futurs ; elles ne prouvent pas une reconstruction parfaite de la base historique. Confirmer la convention source avant de présenter les résultats comme un backtest strict des informations réellement connues.

**19 colonnes explicitement interdites**, `KNOWN_AT_PREDICTION=False` : les huit `new_sRent`, `new_sRentEffective`, `new_sConcession`, `new_sTermMonths`, `new_sSignDate`, `new_sLeaseFrom`, `new_sLeaseTo`, `new_sRenewal` ; `sRenewal`, `days_between`, `months_between`, `old_sRentEffective`, `old_concession_gap`, `previous_effective_growth`, `future_segment_weight` ; les quatre cibles. Les anciens loyers effectifs et croissances effectives restent non qualifiés comme features : leur disponibilité historique n'est pas démontrée. Cette liste n'est pas exhaustive : **tout nom hors de la liste fermée est refusé** par `validate_features`. Les identifiants d'unité servent à l'appariement en mémoire, pas de feature automatique. Les flags calculés avec le nouveau bail servent aux diagnostics, jamais aux prédicteurs.

Le notebook audite chaque colonne du dataset : toutes les colonnes issues des paires restent hors de la matrice X. Les features sont reconstruites séparément par `build_features` à chaque origine historique de train et à l'origine du test. `run_backtest` remet au callable seulement X_train, sa cible et X_test ; pas les loyers, dates, concessions ou statut futur du test.

## Origines, entraînement et fenêtres de test

Le starter précise « comme si vous étiez à la fin de target_year − 1 ». Convention commune : résolution journalière, coupure inclusive au **31 décembre Y−1**, test du **1er janvier au 31 décembre Y** selon `sLeaseFrom` nouveau.

| Backtest | Train cutoff = prediction origin | Fenêtre test |
|---|---|---|
| 2023 | 2022-12-31 | 2023-01-01 / 2023-12-31 |
| 2024 | 2023-12-31 | 2024-01-01 / 2024-12-31 |
| 2025 | 2024-12-31 | 2025-01-01 / 2025-12-31 |

Train expansif : transitions dont le nouveau début est dans une année antérieure à Y ET `label_available_date <= prediction_origin`. Borne prudente du label : **maximum des dates de début, signature et fin des deux baux**, toutes valides et fin non antérieure au début pour chaque bail. Sans ces dates, pas de label d'entraînement. On attend la fin du bail pour ne pas affirmer que l'économie effective réalisée est connue à la signature ; le champ fourni pourrait être un calcul ex ante, ce qui reste à confirmer. Aucun label en cours n'est automatiquement mûr sur la seule année de début.

Le test est une mesure rétrospective du champ source fourni, incluant les baux démarrant en Y ; il ne prétend pas prouver des encaissements futurs effectivement réalisés. Ses labels ne sont jamais livrés au prédicteur ; la correction ci-dessous supprime aussi la sélection ex post de X_test. Les changements futurs des paramètres de validité du moteur devront faire l'objet de tests de stabilité à la coupure. À défaut de versions CRM historiques, le qualificatif correct est **simulation temporelle prudente sur extrait rétrospectif**, avec disponibilité source restant à confirmer.

## Données externes

Bloc additionnel indépendant, non nécessaire aux baselines. Filtre obligatoire : booléen strict `availability_verified=True`, date présente `available_date <= prediction_origin`, valeur présente. Aucun fallback sur `retrieved_date`, année ou publication. Sur les 12 lignes qualifiées, **aucune n'est disponible aux origines des trois backtests** (la publication TAL 2025 intervient après le 31 décembre 2024). Les 101 autres lignes restent exclues. Aucun remplissage ni recherche supplémentaire.

Le contexte est fourni séparément, sans jointure implicite ni feature externe activée dans la liste fermée. Un futur ajout exigera qualification, nom déclaré et correspondance provinciale explicite. TAL uniquement Québec ; guideline Ontario uniquement Ontario, avec admissibilité The Met non confirmée. Jamais un plafond universel ni transfert entre provinces.

## Baselines et métriques communes

Deux références sans tuning : médiane des labels admissibles de Y−1 ; médiane des labels admissibles de Y−3 à Y−1. Le délai de maturité peut modifier la population de Y−1 : leurs résultats ne sont pas les médianes descriptives calculées avec tous les baux. Pas de repli silencieux si historique insuffisant ; statut non calculable. Pas de baseline segmentée utilisant le renouvellement futur.

MAE, RMSE et biais moyen individuel (`prédit − observé`), tous en points de pourcentage. Erreur portefeuille : `100 × (médiane des prédictions test − médiane des g_i observés test)` et valeur absolue. Comparaison annuelle 2023/2024/2025 ; les agrégats multiannuels futurs donneront le même poids aux années, en déclarant toute autre règle. Cette erreur d'agrégat conditionnel n'est pas une erreur de prévision du portefeuille complet ; elle est l'objectif principal du périmètre conditionnel ; les erreurs individuelles restent complémentaires. Aucun gagnant choisi uniquement sur RMSE.

## Interface commune et compatibilité starter

`run_backtest(model, data, leases, year, features=(), external=None)` attend un callable `model(train, X_test, year, external_context)` retournant une fraction par unité candidate, définie à la coupure avant révélation des transitions. Toute méthode doit restituer ultérieurement :

```python
{
    "forecast_year": ...,              # année prévue
    "target_definition": "median_same_unit_effective_annualized",
    "predicted_growth": ...,           # pourcentage, pas fraction
    "method_name": ...,
    "features_used": [...],
    "backtest_by_year": {...},
    "aggregate_metrics": {...},
    "assumptions": [...],
    "drivers": [...],
    "uncertainty": {...}
}
```

`backtest(leases, target_year)` du starter pourra construire les paires, déléguer au protocole et restituer ces métriques ; `estimate_2026(leases, asking, external=None)` pourra retourner le scalaire `predicted_growth` du rapport commun. Aucun adaptateur de prévision ni forecast 2026 n'est réalisé ici. Le starter ne définit pas les futures transitions à prédire : Model A et Model B devront proposer explicitement leur mécanisme d'agrégation annuel avec composition inconnue, en conservant la définition commune. Asking reste hors des features jusqu'à qualification de ses dates/versions.

## Décisions encore ouvertes avant validation finale

1. Confirmer la signification et le moment de disponibilité de `sRentEffective`, et la fidélité temporelle des champs CRM ; lever cette réserve pour revendiquer un backtest strict.
2. Valider métier la médiane effective annualisée comme « Overall Portfolio Rent Growth », qui n'est pas une hausse du revenu total ; le brut et le contractuel restent diagnostics.
3. Définir les hypothèses de composition et de population des transitions futures sans connaître leurs réalisations ; aucune pondération future ne pourra être introduite silencieusement.

Le notebook fournit les résultats reproductibles et les sensibilités qui étayent ces propositions. Aucune ancienne analyse, donnée brute, valeur externe, target du moteur historique ni starter n'est modifiée. Aucun modèle A/B, tuning, prévision finale, export CRM ou commit.

## Dernière validation : cohorte et portée du Foundation Freeze

**Cohort-selection leakage détecté : OUI**, si l'ancien pipeline était présenté comme un forecast portefeuille. `split_year` sélectionne ex post les transitions de Y : connaître leurs unités et leur nombre à la coupure revient à connaître une réalisation future. Cette sélection reste légitime pour décrire la croissance **conditionnelle à une transition**, pas pour constituer la population à prédire.

Correction : `build_prediction_cohort(leases, Y)` prend le dernier bail commencé et signé au 31 décembre Y−1, exclut les débuts ambigus, exige une fin contractuelle valide **dans Y**, et conserve une ligne par UNIT_KEY. Les termes ne remplacent pas une fin manquante. Les dates du nouveau bail, ses prix, concessions et sRenewal ne servent jamais à la sélection. La date d'échéance est un signal de possibilité de transition ; ni départ, renouvellement ni nouveau début ne sont garantis. Les échéances corrigées rétroactivement ne peuvent pas être détectées sans archives CRM.

`run_backtest` reconstruit aussi le train sur les seuls baux connus à la coupure, empêchant une anomalie de bail futur de modifier la succession historique. Il prédit maintenant toutes les unités candidates **avant** le rapprochement avec les transitions révélées. Les cibles restent dans le pipeline d'évaluation, jamais dans la cohorte ou X_candidates. L'évaluation porte ensuite sur l'intersection cohorte/transitions valides. Les candidats sans transition n'ont pas de label imputé à zéro. Les transitions hors cohorte sont comptées dans le total révélé, sans étendre la population prédite après coup. Plusieurs transitions d'un candidat reçoivent la même prédiction conditionnelle par unité : aucun nombre futur de transitions n'est remis au callable. Les baselines antérieures restent des références conditionnelles sur toutes les transitions ex post, et non des scores du nouveau périmètre de cohorte.

| Année | Unités candidates au cutoff | Candidates avec transition valide | Transitions correspondantes | Toutes unités avec transition en Y |
|---|---:|---:|---:|---:|
| 2023 | 405 | 401 | 404 | 420 |
| 2024 | 645 | 634 | 636 | 685 |
| 2025 | 856 | 773 | 779 | 789 |

**Nom business recommandé : « Croissance médiane annualisée du loyer effectif des transitions same-unit observées ».** Pour le score de cohorte, ajouter « parmi les unités candidates à échéance à la coupure ». La formule ne change pas, mais son ensemble d'évaluation doit toujours être annoncé. « Overall Portfolio Rent Growth » est trop large : une projection réelle du portefeuille exige des hypothèses sur probabilités de transition, dates, fréquences et composition, sans transformer les unités sans transition en croissance nulle. Aucun tel modèle n'est construit ici.

**ANNUALISATION VALIDÉE pour la statistique conditionnelle et sur le domaine observé**, pas comme preuve d'un rythme mensuel ni d'un forecast portefeuille. Médianes effectives brut / annualisé : 2023 2,0961 % / 2,0975 % ; 2024 1,8958 % / 1,8651 % ; 2025 3,7144 % / 3,6391 %. Écarts : +0,0015 / −0,0306 / −0,0753 point. Les intervalles minimaux sont 182 jours dans ces trois années : pas d'intervalle de quelques jours. Les groupes courts (<0,5 an) contiennent 8 / 8 / 15 transitions ; l'annualisation amplifie leurs valeurs absolues maximales de 11,1682 à 23,6736 %, de 19,3758 à 42,6793 %, et de 12,5557 à 26,7908 %. Les valeurs restent finies ; leur amplification est réelle et conservée sans retrait arbitraire. La médiane robuste et les sensibilités du notebook rendent ce choix défendable ; aucune garantie d'absence d'extrêmes sur de futurs intervalles plus courts.

**NO-GO pour un Foundation Freeze présenté comme forecast portefeuille strict.** GO technique pour le contrat de croissance conditionnelle et sa sélection à la coupure, sous réserves explicites de fidélité des dates/versions CRM et d'interprétation de l'effectif. La portée business et les hypothèses de transitions futures restent à valider avant la séparation définitive Model A / Model B. sRenewal nouveau demeure interdit, sauf scénario séparé explicitement connu/supposé, qui n'est pas implémenté.

## FOUNDATION FREEZE CANDIDATE v1

Proposition de Phase 7C, **soumise à revue humaine ; aucun freeze déclaré**. Cette section complète le contrat conditionnel sans changer la formule same-unit. Elle définit un indice sur la cohorte connue, pas une variation du rent roll réel.

### Target, population et deux problèmes

- **Business target proposée :** « Indice médian de contribution attendue des transitions à la croissance annualisée du loyer effectif des unités à échéance ».
- **Population :** `C_Y`, une ligne par unité de l'extrait dont le dernier bail commencé et signé à la coupure a une échéance contractuelle valide dans Y. Origine inclusive : 31 décembre Y−1. Pas de listings actuels, nouvelles unités futures ni statut futur de renouvellement.
- **Occurrence :** `O_i = 1` si au moins une transition same-unit admissible est observée dans Y, `0` sinon. La cohorte est fixée avant révélation. Zéro signifie seulement « aucune transition admissible observée dans cet extrait » ; ce n'est ni un loyer constant, ni une unité vacante, ni une absence de revenu.
- **Growth target conservée :** `g_ij = (R_effectif_nouveau / R_effectif_ancien)^(365,25 / jours_entre_débuts) − 1` pour chaque transition admissible j de l'unité i. Pour une unité ayant plusieurs transitions : `G_i = moyenne_j(g_ij)`. Chaque unité a ainsi le même poids dans P1 ; le nombre futur de transitions n'est pas une feature.
- **Deux prévisions :** `p_i = P(O_i=1 | I_cutoff)` et `m_i = E(G_i | O_i=1, I_cutoff)`. Une médiane conditionnelle n'est pas une espérance : elle ne peut pas remplacer m_i en prétendant que le produit est une contribution attendue.
- **Mathematical target P1 :** `100 × médiane_{i∈C_Y}(p_i × m_i)`. Il s'agit de la médiane des contributions individuelles attendues, **pas de l'espérance de la médiane des contributions réalisées**. Cette distinction découle de la non-linéarité de la médiane ; aucune identité entre ces deux grandeurs n'est affirmée.
- **Référence réalisée du backtest :** `100 × médiane_{i∈C_Y}(O_i × G_i)`, avec contribution événementielle nulle si O_i=0. Ce zéro définit l'indice événementiel ; aucune croissance économique réelle nulle n'est imputée. Le score compare l'indice prévu à cette réalisation bruitée et n'est pas une preuve de calibration de l'espérance d'une médiane.

### P1 contre P2

**P1 recommandé pour un périmètre honnêtement restreint**, parce qu'il se calcule sur toutes les unités candidates sans connaître les occurrences futures et incorpore explicitement leur probabilité. L'agrégation médiane est robuste mais n'additionne ni revenus ni dollars.

**P2 rejeté à ce stade.** Un retour au loyer nouveau nécessiterait notamment un ancien effectif réellement connu à la coupure, un intervalle futur identifié et la transformation `(1+g)^(jours/365,25)`. Aucun de ces éléments ne peut être complété par une durée future inventée. L'absence de transition n'identifie pas l'évolution du loyer ni les encaissements. Le statut de `sRentEffective` et le volume réel du portefeuille ne sont pas qualifiés. Les données ne permettent donc pas d'annoncer une prévision de croissance du rent roll complet.

### Occurrence policy, features et leakage

Trois baselines sans tuning : toutes les candidates transitionnent (`p=1`) ; taux global historique qualifié ; taux par province uniquement à partir de **30 labels connus**, sinon repli global. Le seuil est fixé avant scoring. Le taux est la moyenne des O_i historiques. Le taux par province donne ici les mêmes résultats que le global : il ne démontre aucun gain.

L'historique d'occurrence est reconstruit pour chaque ancienne année sur sa propre cohorte à cutoff, uniquement depuis les baux connus à l'origine du backtest courant. Les années non terminées ne fournissent pas de labels. Un positif exige au moins une paire admissible mûre selon la politique début/signature/fin des deux baux ; une paire potentiellement positive encore immature rend le label inconnu plutôt que négatif. Les labels inconnus sont exclus explicitement, non imputés. Les labels qualifiés / en attente sont 897 / 351 en 2023, 1 246 / 407 en 2024, 1 657 / 641 en 2025. Ce retard peut biaiser la composition de l'historique qualifié ; les taux sont des références prudentes, pas des probabilités parfaitement identifiées.

Les features d'occurrence autorisées sont les 12 features à cutoff plus **expiry_month**, tiré de l'échéance contractuelle déjà connue. Aucun ancien effectif, concession gap ou historique d'effectif n'est ajouté comme prédicteur faute de qualification. Le loyer contractuel ancien et le temps écoulé depuis le bail connu restent des diagnostics admissibles. La fonction de probabilité refuse tout champ hors de cette liste, notamment occurrence_target, unit_growth et sRenewal futur. La baseline globale n'utilise aucune variable segmentante.

La croissance P1 de référence est une **moyenne historique des G_i positifs qualifiés**, pour respecter le sens d'une contribution attendue. Pas de modèle ML ni recherche de poids. L'ancienne médiane admissible Y−1 reste un comparateur conditionnel : son agrégation avec `p=1` sur la même cohorte est rapportée séparément. On ne compare pas des scores sur des populations différentes sans le préciser.

L'ordre de `run_portfolio_baselines` est : cohorte et X à cutoff → historique qualifié → probabilités et croissance → agrégat P1 → révélation des transitions de Y → scoring. Les vrais O_i de Y n'interviennent jamais dans la prédiction. Les informations externes ne sont pas utilisées : elles restent optionnelles et soumises au filtre existant. Les restrictions Québec/Ontario et l'interdiction de sRenewal futur sont inchangées.

### Backtests et metrics

Coupures 2022-12-31, 2023-12-31, 2024-12-31 ; années test 2023, 2024, 2025. Simulation rétrospective conditionnelle à la fidélité des champs datés CRM ; pas une preuve de versions historiques archivées.

| Année | Occurrences / candidates | Taux réalisé | Brier global historique | Brier p=1 | P1 global prévu (%) | Référence réalisée (%) | Erreur P1 (pp) | Comparateur médiane Y−1, p=1 : erreur P1 (pp) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2023 | 401 / 405 | 99,012 % | 0,00983 | 0,00988 | 2,84336 | 2,03610 | +0,80726 | +0,22797 |
| 2024 | 634 / 645 | 98,295 % | 0,01677 | 0,01705 | 3,37059 | 1,60785 | +1,76274 | −0,47461 |
| 2025 | 773 / 856 | 90,304 % | 0,09378 | 0,09696 | 3,11833 | 3,30088 | −0,18255 | −1,06026 |

Accuracy et precision des trois baselines : taux réalisé de chaque année ; recall : 1. Toutes prédisent la classe positive au seuil fixé 0,5 : cela illustre pourquoi accuracy seule ne distingue pas les baselines. Calibration globale historique : probabilité moyenne 98,328 / 98,475 / 98,189 %, écart au taux observé −0,685 / +0,181 / +7,886 points. **Global historique recommandé comme référence d'occurrence** : Brier légèrement meilleur que p=1 sur les trois années, simplicité ; la segmentation ne change pas ces scores. La dégradation 2025 empêche de prétendre que la probabilité est stable. Aucun paramètre n'est ajusté sur ces années pour corriger cette dérive.

Les erreurs individuelles de croissance restent celles du contrat conditionnel. Le score P1 est un agrégat de cohorte entier ; il doit être nommé comme tel, jamais « erreur de rent roll ». Les calibrations de petits groupes ne justifient pas de conclure à la prévisibilité des absences.

### Non-transitions : constats limités

4 / 11 / 83 candidates sans transition admissible observée. Les quatre de 2023 et les onze de 2024 sont québécoises ; toutes leurs échéances sont de septembre à décembre (2023 : novembre–décembre). Leur faible nombre ne permet pas de définir une segmentation prédictive fiable.

En 2025, 65 des 83 concernent Saint-Elzear, 81 le Québec, 67 des unités de deux chambres, 79 des anciens termes de 12 mois. Ces nombres sont des ventilations marginales, pas 83 dossiers publiés ni des groupes indépendants. Le loyer contractuel ancien médian est 2 310 $ chez les non-transitions contre 2 195 $ chez les transitions ; temps depuis ancien début 4,50 contre 5,68 mois à l'origine. Ces associations utilisent uniquement des caractéristiques à cutoff ; elles ne démontrent aucune causalité ni prévisibilité. Les prix effectifs anciens sont exclus de cette analyse faute de disponibilité qualifiée. L'extrait s'arrête en 2025 : une transition retardée à 2026 ou omise n'est pas une preuve de stagnation du loyer.

### Composition 2026, sans prévision

**931 candidates**, construites au 2025-12-31. Québec : 810 ; Ontario : 121. Bâtiments : Daniel-Johnson 128, Le Carlyle 180, Levesque 75, Saint-Elzear 268, The Met 121, Westpark 159.

Propriétés : carlyle 180 ; dj1 56 ; dj2 72 ; levesque 75 ; metcalfe 121 ; stelz1 37 ; stelz2 113 ; stelz3 118 ; wsp1 159.

Mois d'échéance de janvier à décembre : **38, 52, 65, 62, 91, 143, 90, 91, 108, 61, 69, 61**. Chambres 0/1/2/3 : 49 / 316 / 544 / 22. Termes connus 5,9 / 12 / 17,9 / 23,9 mois : 3 / 897 / 15 / 16. Superficie médiane 1 030 pi² ; ancien loyer contractuel médian 2 320 $. Aucun ancien effectif ni cible future ne sert à cette construction. Ces volumes concernent l'extrait, pas le portefeuille réel. Aucun +X % 2026 n'a été calculé.

### Limites et recommandation de revue

L'indice P1 rend occurrence et croissance séparées et est calculable avec la cohorte connue. Il ne résout pas l'absence de journal CRM, la qualification de l'effectif, la censure/complétude des transitions, les retards de maturité et la dérive 2025. L'ancien périmètre conditionnel reste intact pour diagnostic ; le changement de population/agrégation P1 est explicite.

**GO méthodologique recommandé pour soumettre ce contrat P1 restreint à la revue humaine ; NO-GO pour le présenter comme croissance du rent roll complet.** La décision finale de Foundation Freeze appartient à la revue humaine, notamment sur l'intitulé business, les hypothèses de disponibilité interne et le biais possible des labels historiques qualifiés. Aucun freeze déclaré et aucun forecast final 2026.
