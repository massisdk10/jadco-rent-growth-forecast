# Signaux externes officiels

Collecte vérifiée le **3 octobre 2026**. Le fichier long `data/external/external_signals.csv` contient uniquement des statistiques publiques : aucune unité, aucun bail ni identifiant CRM JADCO. Les travaux existants, la cible same-unit, les variantes d'économie locative et les règles du starter restent inchangés. Aucune jointure, sélection finale de variables, estimation ou simulation de backtest n'a été effectuée.

| Source | Lignes | Valeurs présentes | Années de référence |
|---|---:|---:|---|
| SCHL | 42 | 20 | 2019–2025 |
| Statistique Canada | 40 | 40 | 2017–2026, 2026 partielle |
| Gouvernement de l’Ontario | 10 | 10 | 2017–2026 |
| TAL | 21 | 21 | 2025–2026 |
| Total | 113 | 91 | |

## SCHL : marché locatif construit à cette fin

Source : [Rapport sur le marché locatif 2025](https://www.cmhc-schl.gc.ca/professionals/housing-markets-data-and-research/market-reports/rental-market-reports-major-centres), publié le 11 décembre 2025. Les valeurs viennent des textes et tableaux des figures du rapport officiel. Aucun marché de copropriétés locatives n'est mélangé avec le marché construit à cette fin.

Vérification des 20 valeurs retenues :

| Géographie et signal | Années et valeurs publiées |
|---|---|
| Montréal, vacancy_rate, ensemble des logements (%) | 2025 : 2,9 |
| Ottawa, vacancy_rate, ensemble des logements (%), figure 8 | 2023 : 2,1 ; 2024 : 2,6 ; 2025 : 3,0 |
| Montréal, average_rent_growth, deux chambres (%) | 2025 : 7,2 |
| Ottawa, average_rent_growth, deux chambres (%) | 2025 : 3,4 |
| Montréal, average_rent, deux chambres ($/mois) | 2025 : 1 346 |
| Ottawa, average_rent, deux chambres ($/mois) | 2025 : 1 926 |
| Montréal, turnover_average_rent, deux chambres ($/mois), figure 1 | 2022 : 1 235 ; 2023 : 1 310 ; 2024 : 1 407 ; 2025 : 1 644 |
| Ottawa, turnover_average_rent, deux chambres ($/mois), figure 1 | 2022 : 1 829 ; 2023 : 1 903 ; 2024 : 2 118 ; 2025 : 2 155 |
| Montréal, turnover_rent_growth (%), figure 9 | 2024 : 18,7 ; 2025 : 17,2 |
| Montréal, non_turnover_rent_growth (%), figure 9 | 2024 : 4,7 ; 2025 : 6,0 |

La croissance à échantillon constant publiée par la SCHL porte sur des immeubles communs aux enquêtes et inclut logements occupés, vacants et nouvellement loués. Ce n'est ni un ratio calculé entre les niveaux moyens, ni une mesure identique à la cible same-unit JADCO. Le loyer moyen à la rotation est un **niveau**, pas un taux de rotation. Ottawa désigne le marché ontarien ; Gatineau n'est pas inclus dans ces valeurs.

Le [catalogue historique officiel](https://www.cmhc-schl.gc.ca/chic/Listing?item_ID=%7BA6C8DBDA-51BA-4EE7-9432-5D0522FE2A8D%7D) référence les éditions antérieures, mais son téléchargement automatisé a échoué (HTTP 403 ; lien Excel non récupéré). Les deux signaux prioritaires pour 2019–2024 restent donc manquants, sauf les taux d'inoccupation d'Ottawa 2023–2024 expressément reproduits dans le rapport 2025 : **22 lignes sans valeur**. Ces lignes conservent la provenance de recherche et une explication, sans date de disponibilité. Aucune interpolation. Le taux de rotation et les croissances par rotation d'Ottawa ne sont pas collectés ; ne pas les déduire des niveaux.

## Statistique Canada : IPC des loyers

Source : [tableau 18-10-0004-01](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1810000401), [archive CSV officielle](https://www150.statcan.gc.ca/n1/en/tbl/csv/18100004-eng.zip). Sélection exacte : `Rent`, `Quebec` et `Ontario`, indices mensuels non désaisonnalisés, base 2002 = 100. Vecteurs : Québec `v41691818`, Ontario `v41691954`. Dernier mois observé : août 2026. Les mois de 2016 servent uniquement de dénominateur à la croissance 2017.

Pour chaque province et chaque année complète 2017–2025 :

- `rent_cpi_annual_mean` = moyenne arithmétique des 12 indices mensuels de l'année ; unité `index_2002_100`.
- `rent_cpi_annual_growth` = `100 × (moyenne annuelle Y / moyenne annuelle Y−1 − 1)` ; unité `percent`.

Pour 2026, deux signaux séparés : `rent_cpi_ytd_mean`, moyenne janvier–août ; `rent_cpi_ytd_growth`, `100 × (moyenne janvier–août 2026 / moyenne janvier–août 2025 − 1)`. Ils ne sont pas annualisés et ne représentent pas une année complète. Les 40 valeurs dérivées ont été recalculées et comparées aux mois officiels avant export. Les données mensuelles temporaires ne sont pas conservées dans le dépôt. Aucun mois manquant ou futur n'a été ajouté.

Les dates originales de publication de chaque mois et les versions historiques ne sont pas reconstituées. Une série historique téléchargée aujourd'hui peut contenir des révisions : elle ne devient pas automatiquement admissible aux backtests passés.

## Ontario : guideline historique

Source : [Residential rent increases](https://www.ontario.ca/page/residential-rent-increases), tableau officiel des guidelines, vérifié pour chaque année :

| Année | Pourcentage |
|---|---:|
| 2017 | 1,5 |
| 2018 | 1,8 |
| 2019 | 1,8 |
| 2020 | 2,2 |
| 2021 | 0,0 |
| 2022 | 1,2 |
| 2023 | 2,5 |
| 2024 | 2,5 |
| 2025 | 2,5 |
| 2026 | 2,1 |

Signal `ontario_guideline`, unité `percent`. Le zéro de 2021 correspond au gel publié. Les notes précisent : ne s'applique pas au turnover ; certains logements occupés pour la première fois après le 15 novembre 2018 sont exemptés ; applicabilité exacte à The Met à confirmer. Aucune contrainte universelle sur les baux n'est construite. La date de création ou de mise à jour de la page ne remplace pas la date d'annonce de chaque guideline : ces dates restent inconnues.

## TAL : composantes et rupture de méthode

Source : [Pourcentages applicables aux critères de fixation de loyer](https://www.tal.gouv.qc.ca/fr/reconduction-du-bail-et-fixation-de-loyer/pourcentages-applicables-aux-criteres-de-fixation-de-loyer). Les anciennes composantes sont conservées séparément sous le préfixe `tal_legacy_` :

| Composante | 2025 (%) | 2026 (%) |
|---|---:|---:|
| electricity | 2,9 | 2,9 |
| gas | −5,8 | 13,7 |
| fuel_oil_other_energy | −2,9 | 4,9 |
| maintenance | 6,9 | 5,7 |
| building_services | 4,5 | 3,4 |
| personal_services_rpa | 5,9 | 6,3 |
| management | 8,2 | 7,2 |
| net_income | 6,9 | 5,7 |
| capital_expenditure | 4,7 | 4,0 |

Nouvelle méthode 2026, trois signaux distincts : pourcentage de base **3,1 %**, services personnels **6,7 %**, dépenses d'immobilisation **5,0 %**. Le [communiqué officiel du 19 janvier 2026](https://www.tal.gouv.qc.ca/fr/actualites/detail?code=le-calcul-de-l-ajustement-des-loyers-en-2026) précise la rupture méthodologique : la nouvelle méthode concerne les avis donnés à compter du 1er janvier 2026 ; l'ancienne méthode reste pertinente pour les avis antérieurs. Les périodes des baux indiquées dans la table sont du 2 avril 2025 au 1er avril 2026 et du 2 avril 2026 au 1er avril 2027.

Aucun « guideline Québec » universel n'est calculé. La base 2025 affichée dans la nouvelle table n'est pas utilisée pour fabriquer une série rétrospective continue. Les composantes TAL antérieures à 2025 ne sont pas collectées ici. L'applicabilité exige de confirmer les dates des avis et les éléments propres aux baux ; les composantes ne s'additionnent pas arbitrairement.

## Publication, disponibilité et prévention du leakage

- **REFERENCE PERIOD** (`reference_period`) : période économique décrite, distincte de la date de connaissance.
- **PUBLICATION DATE** (`publication_date`) : date officielle de publication, uniquement lorsqu'elle est vérifiée.
- **RETRIEVED DATE** (`retrieved_date`) : date de récupération par notre équipe, ici le 3 octobre 2026. Elle ne constitue jamais une disponibilité historique.
- **AVAILABLE DATE** (`available_date`) : première date vérifiée à laquelle cette valeur/version aurait réellement pu être utilisée. Sans preuve, champ manquant et `availability_verified=False`.

**12 lignes sont vérifiées** : les neuf composantes TAL 2025, publiées et mises à disposition le [21 janvier 2025](https://www.tal.gouv.qc.ca/fr/actualites/detail?code=le-calcul-de-l-ajustement-des-loyers-en-2025), et les trois nouveaux paramètres TAL 2026, publiés le 19 janvier 2026. `availability_notes` conserve la preuve officielle correspondante. La rupture de méthode reste explicite. Aucun paramètre 2024 ne figure dans ce fichier : aucune ligne ni date n'est ajoutée pour cette année.

**101 lignes restent non vérifiées**, avec `available_date` manquant : toutes les lignes SCHL, Statistique Canada et Ontario, ainsi que les neuf anciennes composantes TAL 2026. Une date de publication du rapport SCHL est connue mais la disponibilité historique de chacune des valeurs reproduites dans la version actuelle n'est pas qualifiée. Les dates d'annonce individuelles Ontario restent inconnues. Les dates des publications mensuelles et des versions IPC ne sont pas reconstituées. Les valeurs restent disponibles pour description, sans autorisation de les utiliser en backtest.

`get_available_external_data(data, as_of_date)` exige un booléen strict `availability_verified=True`, une valeur présente et une date connue inférieure ou égale à la coupure inclusive. Aucun repli vers `retrieved_date`, `publication_date` ou `year`. Une chaîne telle que `"False"` ne constitue pas une vérification. Pour une origine au 31 décembre 2025, seules les neuf composantes TAL 2025 sont admissibles. Il s'agit d'un contrôle de disponibilité, pas d'un backtest exécuté ni d'une confirmation d'applicabilité juridique à chaque bail.

Une moyenne annuelle complète de Y ne peut **jamais** être connue au début de Y. Sa disponibilité exige la publication de tous ses mois, y compris décembre, et la qualification de la version employée ; un taux annuel exige aussi la version admissible du dénominateur. Les agrégats janvier–août 2026 exigent tous ces huit mois. Retrouver ces preuves avant toute utilisation historique ; aucune imputation ni antidatation sur l'année de référence.

## Ce que ces données peuvent expliquer

- **Pression du marché** : l'inoccupation et les loyers de la SCHL contextualisent Montréal et Ottawa ; leur univers ne représente pas nécessairement le parc JADCO.
- **Inflation des loyers** : l'IPC des loyers décrit une évolution provinciale du coût du logement loué, distincte des nouveaux loyers et de la cible same-unit.
- **Réglementation** : paramètres TAL pour le Québec et guideline ontarien pour les logements admissibles ; aucun transfert automatique entre provinces.
- **Turnover** : niveaux à la rotation et croissances SCHL distinguant rotation et maintien des occupants, sans déduire les mouvements individuels JADCO.
- **Contexte Québec/Ontario** : garder province, géographie et période explicites ; Montréal n'est pas une statistique locale de Laval, Ottawa n'est pas Gatineau. Tout rattachement aux propriétés nécessite une validation ultérieure.

## Contrôles et décisions restantes

Le module vérifie le schéma exact, les clés `geography/province/signal/reference_period`, la provenance officielle HTTPS, les dates, les valeurs numériques et des plages exploratoires. Il signale les anomalies sans supprimer silencieusement de valeurs. Les lignes manquantes restent traçables. Les tests ciblent ces contrôles et le filtrage temporel.

Décisions humaines restantes : qualifier les versions/dates historiques ; confirmer l'admissibilité de The Met et les avis concernés par le TAL ; accepter ou préciser la correspondance des univers SCHL/IPC avec le portefeuille. Les niveaux, indices et taux ne sont pas directement interchangeables. Les signaux restent candidats, sans pondération ni modification de cible.

La protection globale `*.csv`, `data/raw/` et `data/processed/` est conservée. Une exception ancrée autorise uniquement `/data/external/external_signals.csv`, public. Les tests vérifient que les chemins CRM restent ignorés. Aucun commit ni ajout à l’index Git.
