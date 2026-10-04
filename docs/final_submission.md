# Soumission finale — architecture gelée et résultats 2026

Le starter officiel est la référence. Le moteur A/B/C, la fondation, les constantes, fenêtres, cohortes et règles de cutoff sont intégralement préservés. L’adaptateur final expose les signatures du starter ; aucun réglage après observation de 2026. Les sources asking et externes sont acceptées par l’interface, mais restent exclues des prédicteurs gelés.

## Résultat officiel

**P1 2026 = 2.445012251 %**, à la coupure du 2025-12-31, sur **931 candidates**, dont 810 au Québec et 121 en Ontario.
Occurrence attendue : **94.921771 %**. Médiane de croissance conditionnelle prédite sur toutes les candidates : **2.566148 %**. P1 est `100 × médiane(p_i × g_i)`, pas le produit de ces deux résumés. Les contributions individuelles restent exclusivement en mémoire.

La cible rétrospective conserve la cohorte ex ante : zéro sans transition qualifiée, moyenne annuelle des croissances qualifiées de chaque unité sinon. L’annualisation effective est composée entre débuts des baux, en jours/365,25. Cette cible P1 diffère de la médiane descriptive de toutes les transitions et ne représente pas le revenu ni l’occupation du portefeuille réel.

## Explication et confiance

A stabilise les deux composantes à partir de l’historique mûr. B conserve P1_G1 (« Model B — full hierarchical »), occurrence province/mois et croissance province, et ajuste uniquement selon son support et sa pertinence contextuelle. Les confiances sont des heuristiques, pas des probabilités de précision.

occurrence : A=95.236088 %, B=95.038734 % ; poids moyens A/B/C=92.689787/7.310213/0.000000 %. Support B médian=169, minimum=1, maximum=356 ; 0 replis globaux.

growth : A=2.565881 %, B=2.607293 % ; poids moyens A/B/C=98.436024/1.563976/0.000000 %. Support B médian=2308, minimum=106, maximum=2308 ; 0 replis globaux.

**C inactif pour les deux composantes** : 89/856 labels récents connus (10.397196 %), dont 6 positifs. La couverture requise est 80 %, le support requis 30 par composante. Le support d’occurrence suffit seul, mais pas sa couverture ; le support de croissance ne suffit pas non plus. Aucun assouplissement.

Les poids sont normalisés séparément pour occurrence et croissance. L’ancre A a confiance 1 ; B utilise support n/(n+30) et divergence au propre global et à A, avec poids nul en repli global. C applique qualification, couverture, cohérence de dispersion et divergence. Les formules exactes restent dans `docs/contextual_orchestration.md` et le module inchangé.

Empreinte gelée : `f8ce4bd1355a1ab3ee238bdbbb8754e38a70d332b72f5548c1187952dc51fa07`.

## Diagnostics agrégés

Les bâtiments ci-dessous sont des segments de l’extrait, pas des stocks réels. Les médianes segmentées ne se somment pas pour reconstruire P1. Les statistiques de groupes de moins de cinq candidates sont masquées par l’adaptateur.

| Bâtiment | Candidates | Occurrence (%) | Croissance conditionnelle médiane (%) | Médiane contribution (%) |
| --- | ---: | ---: | ---: | ---: |
| Le Carlyle | 180 | 94.722007 | 2.566148 | 2.445164 |
| Daniel-Johnson | 128 | 94.799520 | 2.566148 | 2.445164 |
| Levesque | 75 | 94.605077 | 2.566148 | 2.445164 |
| The Met | 121 | 95.318876 | 2.511377 | 2.393582 |
| Saint-Elzear | 268 | 94.967923 | 2.566148 | 2.445164 |
| Westpark | 159 | 95.015726 | 2.566148 | 2.445164 |

Québec : occurrence 94,862450 %, croissance conditionnelle 2,566148 %, médiane contribution 2,445164 %. Ontario / The Met : occurrence 95,318876 %, croissance conditionnelle 2,511377 %, médiane contribution 2,393582 %. Les provinces demeurent distinctes ; aucune règle réglementaire n’est transférée ni appliquée automatiquement. The Met dispose désormais de support dans les deux composantes B ; aucun repli global n’est observé en 2026.

Échéances janvier à décembre : 38, 52, 65, 62, 91, 143, 90, 91, 108, 61, 69, 61. Ces mois sont ceux des baux connus à la coupure, pas des transitions futures observées.

## Scénarios descriptifs

LOW / BASE / HIGH = **0.781020 % / 2.445012 % / 4.109156 %**.

Réutilisation directe de `uncertainty_scenarios` : 1 000 bootstraps des contributions unitaires avec remplacement, seed=42 ; E=max des erreurs absolues de l’orchestration sur 2023/2024/2025 ; LOW=q10−E, BASE=P1, HIGH=q90+E. Aucun ajustement du biais ou plancher arbitraire.
E=1.663992580 pp ; q10=2.445012251 %, q90=2.445163783 %. Le bootstrap est très étroit car les contributions sont concentrées ; l’enveloppe historique domine les scénarios. Cela ne démontre pas une faible incertitude prédictive.

Ces scénarios défavorable / central / favorable ne sont pas des intervalles de confiance calibrés. Les dépendances entre unités et bâtiments, les changements de régime et les ruptures ne sont pas probabilistiquement couverts.

## Historique et limites

| Année | Prévision P1 (%) | Réalisé (%) | Erreur (pp) |
| --- | ---: | ---: | ---: |
| 2023 | 2.872357 | 2.036102 | 0.836255 |
| 2024 | 3.271845 | 1.607852 | 1.663993 |
| 2025 | 2.972696 | 3.300878 | -0.328182 |

La MAE orchestrée est 0,942810 pp contre 0,942278 pour A. Les erreurs A/B sont corrélées. D demeure rejeté ; les variantes B simples peuvent dominer certaines métriques sans entraîner de sélection de champion ou de modification de la représentation P1_G1. Le notebook reproduit A, B, 50/50, D et orchestration.

Aucun signal fiable à cutoff n’a anticipé le ralentissement 2024 ni la rupture Saint-Elzear 2025. Trois années ne valident pas une précision générale. C peut rester absent. Le CRM rétrospectif n’offre pas de journal de versions : les simulations supposent la fidélité historique des champs. Le loyer effectif est une convention source non entièrement reconstruite par les détails de concessions ; PromoPay n’est pas mensualisé sur tout le bail. Asking et données publiques non qualifiées ne sont pas forcés dans le modèle.

## Exécution et validation

Exécuter `notebooks/14_final_submission.ipynb` intégralement avec l’environnement existant. Il charge les quatre CSV depuis `data/raw/`, exécute les backtests et interfaces, puis écrit uniquement `outputs/reports/final_2026_summary.json`. `estimate_2026(leases, asking, external=None)` renvoie un scalaire en pourcentage ; `backtest(leases, target_year)` renvoie un résumé agrégé pour 2023, 2024 ou 2025. Aucun paramètre d’entraînement n’est exposé pour une optimisation.

Validation : 32 tests ciblés réussis, dont neuf tests nouveaux de soumission ; suite complète **212 tests réussis**, dix avertissements non bloquants préexistants. Notebook final : quatre cellules de code exécutées, aucune erreur. Les tests contrôlent cohorte, unités, poids, produit p×g, médiane, disponibilité temporelle, absence d’influence des baux futurs et des sources optionnelles, reproduction historique et JSON agrégé strict.

Fichiers créés : module final, tests finaux, notebook final, ce document et résumé JSON agrégé. Aucun fichier préexistant modifié, aucune donnée brute exportée, aucun commit, push ou merge.
