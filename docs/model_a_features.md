# Model A — moteur de features A0/A1/A2

Référence : tag `foundation-v1`, contrat inchangé. Les cohortes 405/645/856/931 et occurrences 401/634/773 sont contrôlées explicitement ; tout écart arrête le garde. Les noms sources sans accents (`Levesque`, `Saint-Elzear`) restent conservés. La croissance est une **fraction** ; l'affichage seul multiplie par 100. Les unités à plusieurs transitions gardent la moyenne de leurs croissances selon la fondation P1.

## Interface et séparation

`build_model_a_features(leases, target_year, prediction_origin=None, include_targets=False, external=None, feature_set="core")` retourne un `ModelAFeatureDataset` :

- `audit` : sPropCode, sUnitCode (UNIT_KEY officiel), target_year, prediction_origin ; jamais prédicteurs.
- `X` : 13 features, dont expiry_month réservé à l'occurrence.
- `X_growth` : copie des 12 features autorisées pour la croissance.
- `labels` : transition_occurred et conditional_growth, seulement sur demande rétrospective ; aucune cible requise/permise pour 2026.
- `external_context` : données vérifiées disponibles à la coupure, séparées de X, sans jointure ni activation automatique.

Origines : 2022-12-31 / 2023-12-31 / 2024-12-31 / 2025-12-31. Un cutoff différent est refusé. La cohorte est construite avant X et avant la révélation des labels. Le moteur réutilise les fonctions figées ; CORE n'appelle pas le moteur de transitions si include_targets=False ; ENRICHED consulte uniquement les transitions historiques admissibles. Les tables individuelles restent uniquement en mémoire : aucun CSV, dump de records ou export de clés.

## Définitions des features autorisées

Source commune : dernier bail **commencé et signé** au plus tard à l'origine, sans début ambigu, obtenu par les helpers de la fondation. Aucune lecture des listings actuels pour compléter l'historique. Disponibilité conditionnelle à la fidélité des valeurs CRM historiques ; aucune preuve de versions archivées n'est inventée.

| Feature | Définition / source | Usage recommandé |
|---|---|---|
| property_code | sPropCode du bail connu ; pas de numéro d'unité | Optionnel, redondant avec bâtiment |
| building | sBuilding | Occurrence et croissance |
| province | sState ; Québec/Ontario séparés | Occurrence et croissance ; confondu partiellement avec bâtiment |
| bedrooms | sBeds | Occurrence et croissance |
| bathrooms | sBaths | Optionnel, lié aux caractéristiques physiques |
| sqft | sSqft, superficie source | Occurrence et croissance |
| floor | sFloor | Optionnel, risque de complexité supplémentaire |
| unit_subtype | sUnitSubtype | Optionnel, surveiller les catégories rares |
| prior_contractual_rent | sRent du bail connu, dollars mensuels source | Occurrence et croissance ; jamais converti en effectif |
| prior_term_months | sTermMonths source, sans arrondi | Occurrence et croissance ; pas l'intervalle d'annualisation |
| forecast_year | année Y demandée, connue à l'origine | Optionnel ; extrapolation temporelle à contrôler |
| months_since_known_start | (origine − début connu).jours / (365,25 / 12) | Occurrence et croissance ; temps à l'origine, pas intervalle futur |
| expiry_month | mois de l'échéance contractuelle connue | Occurrence uniquement selon la liste fermée |

Manquants : NaN conservé, aucune unité supprimée pour une feature absente, aucune imputation/fallback appris. Un futur pipeline pourra imputer sur son train uniquement. Les unités avec début ambigu/fin manquante suivent la règle de cohorte déjà figée. Les codes/unités servent à identifier et auditer, pas à prédire.

## Features économiques différées : divergence du plan de conception

Le PDF `docs/modele_A.pdf` a été lu via PDFKit local, sans dépendance installée. Il propose rent_position, effectif courant, gap, historiques et dynamiques récentes. **Ces propositions dépassent la liste fermée du contrat**, qui reste l'autorité expressément demandée. Le mode CORE ne les construit pas ; le contrat n'est pas modifié. A2.5 ci-dessous autorise une extension expérimentale séparée.

Sont différés : current_effective_rent, effective_rent_per_sqft, rent_position (même une variante contractuelle serait hors liste), concession_gap, previous_effective_growth, history_count, unit_median_growth, recent_building_growth, recent_portfolio_growth, recent_province_growth, history_available. Aucun fallback de segment ou fenêtre récente n'est inventé. Les tests vérifient leur absence et le refus de les sélectionner ; un test de leur formule serait sans objet tant qu'elles sont interdites. L'ancien effectif demeure la base de la **target**, pas une feature disponible par simple présence dans le CSV.

Le contexte externe est facultatif et filtré par availability_verified=True et available_date <= origine ; aucun repli sur retrieved_date. Les trois backtests ont zéro observation externe admissible ; neuf lignes TAL sont disponibles à l'origine 2026 mais restent hors X faute de noms/features autorisés et de correspondance d'applicabilité validée.

## Diagnostic et recommandations

Le notebook 06 ne contient aucun modèle entraîné. Il fournit missingness, cardinalités, distributions/composition par origine, groupes rares, associations descriptives aux labels révélés 2023–2025 et screening. Les agrégats de moins de cinq observations ont leurs statistiques masquées ; les corrélations exigent vingt couples complets et sont descriptives, non une sélection apprise. Aucun outlier n'est supprimé. 2026 sert uniquement au drift des features ; aucun label ni forecast.

La table de screening recommande KEEP / OPTIONAL / DROP_FROM_MODEL_A sur sens structurel, redondance et autorisation. Son available_rate est descriptif sur les quatre cohortes, pas un paramètre appris pour les backtests. Les associations aux labels ne doivent pas servir à choisir les features sur l'année test ; la sélection/encodage futurs seront appris à l'intérieur des folds. Le booléen point_in_time_safe décrit le contrôle de coupure sous l'hypothèse de fidélité CRM, sans lever cette réserve historique.

Compatibilité starter conservée : UNIT_KEY officiel, conventions Yardi, moteurs same-unit/rent economics non modifiés, effectif/contractuel distincts, concessions non reconstruites, sRenewal futur interdit, aucun taux d'occupancy déduit. Les tables pourront alimenter occurrence/growth puis P1 dans de futurs adaptateurs backtest(...) et estimate_2026(...).

Décision requise : conserver ce périmètre restreint pour A3–A6 ou autoriser séparément l'extension économique avec preuve de disponibilité. Aucun modèle final, tuning, modification des fondations, forecast 2026 ou commit dans cette phase.


## A2.5 — Extension économique et historique

### CORE et MODEL A EXPERIMENTAL EXTENSION

CORE reste le défaut : 13 features occurrence / 12 croissance, cohortes et labels identiques. `feature_set="enriched"` demande explicitement 17 / 16 features. Le sélecteur Model A accepte les quatre historiques uniquement dans ce mode ; aucun changement au contrat commun A/B. Le screening A0–A2 reste celui de CORE.

La coupure historique est **conservatrice** : filtrer les baux commencés et signés à D avant de former les paires, conserver les transitions admissibles dont `label_available_date <= D`. Cette borne fondation est le maximum des débuts, signatures et fins des deux baux ; elle ne démontre pas la date réelle de calcul de `sRentEffective`. La croissance conserve exactement la formule annualisée fondation, sans retrait d'extrêmes.

**Condition de fidélité** : les effectifs des baux terminés doivent avoir été effectivement connus à cette borne, sans révision CRM ultérieure. Le snapshot disponible ne prouve pas cette condition. Les tests établissent l'invariance aux ajouts futurs, pas la fidélité des versions historiques. Aucun historique ne devient SAFE sans cette réserve.

| feature | status | definition | information_source | cutoff_rule | historical_fidelity_risk | missingness | fallback | approved_for_experiment |
|---|---|---|---|---|---|---|---|---|
| prior_effective_rent | NOT_IDENTIFIABLE | Effectif du bail courant connu | sRentEffective du bail de la cohorte | Début/signature ne prouvent pas la disponibilité de l'effectif ; bail non terminé à D | Date de calcul et versions inconnues | Non construit | Aucun ; pas de substitution d'un ancien bail | Non |
| concession_gap | NOT_IDENTIFIABLE | (contractuel − effectif) / contractuel | sRent et effectif courant | Prérequis effectif non qualifié | Même risque ; aucune reconstruction de concession | Non construit | Aucun | Non |
| rent_position | NOT_IDENTIFIABLE | Effectif courant / médiane des effectifs comparables | Effectifs courants du stock connu | Prérequis non qualifié pour numérateur et références | Même risque et population de référence | Non construit | Aucun fallback de segment tant que le prérequis manque | Non |
| previous_same_unit_growth | SAFE_WITH_CONDITIONS | Dernière croissance annualisée admissible par UNIT_KEY | Deux baux historiques, moteur fondation | Tous les débuts/signatures/fins <= D | Effectifs terminés supposés connus et non révisés | NaN sans historique | Aucun | Oui, sous condition de fidélité |
| unit_historical_median_growth | SAFE_WITH_CONDITIONS | Médiane de toutes les croissances admissibles par UNIT_KEY | Mêmes paires historiques | Même borne conservatrice | Même réserve CRM | NaN sans historique | Aucun | Oui, sous condition de fidélité |
| recent_building_growth | SAFE_WITH_CONDITIONS | Médiane récente par bâtiment | Paires historiques et bâtiment du bail connu | Borne précédente et nouveau début dans (D−365 jours, D] | Effectifs et appartenance historiques | NaN si moins de cinq transitions | NaN ; aucune fenêtre élargie | Oui, diagnostic uniquement |
| recent_portfolio_growth | SAFE_WITH_CONDITIONS | Médiane récente du portefeuille | Paires historiques admissibles | Même fenêtre et borne | Effectifs historiques | NaN si moins de cinq transitions | NaN | Oui, diagnostic uniquement |

Le seuil de cinq transitions et la fenêtre unique de 365 jours sont fixés sans consulter les targets. Aucun historique supplémentaire, aucun comptage auxiliaire ajouté à X. Les NaN ne suppriment aucune unité. Une future imputation éventuelle doit être apprise sur train seulement, avec politique explicite pour une colonne entièrement absente ; aucune imputation ici.

### Résultats descriptifs et décision A3–A6

| Feature | Disponibilité 2023 / 2024 / 2025 / 2026 (%) | Décision |
|---|---|---|
| previous_same_unit_growth | 86,17 / 55,35 / 50,58 / 63,37 | APPROVE FOR A3-A6, sous validation explicite de fidélité CRM et gestion des NaN sur train |
| unit_historical_median_growth | 86,17 / 55,35 / 50,58 / 63,37 | APPROVE FOR A3-A6, mêmes conditions ; redondance à examiner sur train |
| recent_building_growth | 52,59 / 52,71 / 0 / 0 | REJECT FOR A3-A6 : support nul aux deux dernières origines |
| recent_portfolio_growth | 100 / 100 / 0 / 100 | REJECT FOR A3-A6 : absence totale 2025 et une seule valeur commune par origine |

Les trois features non identifiables sont REJECT FOR A3-A6. La borne de fin des deux baux réduit fortement le support de la fenêtre récente et sélectionne des transitions de baux courts ; ces médianes ne représentent pas toute la dynamique locative récente. La disponibilité historique décroissante peut refléter la composition et la maturité, sans démonstration causale. Les corrélations ex post restent des diagnostics, jamais des critères de sélection sur l'année test. Aucune target 2026 n'est consultée.

**Verdict : NO-GO pour lancer immédiatement A3–A6 avec ENRICHED sans validation de fidélité historique. CORE conserve son périmètre déjà validé.** Les deux historiques unitaires sont des approbations conditionnelles, pas une preuve d'observabilité CRM. Les indicateurs récents restent implémentés uniquement pour l'expérience descriptive.
