# JADCO / Collection Équinoxe — prévision 2026

## Objectif

Estimer, pour les unités à échéance, l’**Indice médian de contribution attendue des transitions à la croissance annualisée du loyer effectif**. Cet indicateur ne mesure pas la croissance totale du rent roll.

Pour une unité candidate `i` :

- `p_i` : probabilité de transition ;
- `g_i` : croissance annualisée attendue du loyer effectif conditionnelle à une transition ;
- `c_i = p_i * g_i` ;
- `P1 = median(c_i)`, rapporté en pourcentage.

Le backup final combine les modèles gelés par `Forecast_backup = 0.50 * Forecast_A + 0.50 * Forecast_B_P1_G1`. **The 50/50 weights are fixed ex ante and were not optimized on the historical backtests.**

## Résultats

| Année | P1 observé (%) | Model A (%) | Model B P1_G1 (%) | Ensemble 50/50 (%) |
|---:|---:|---:|---:|---:|
| 2023 | 2.036102 | 2.870060 | 2.871314 | 2.870687 |
| 2024 | 1.607852 | 3.269179 | 3.236099 | 3.252639 |
| 2025 | 3.300878 | 2.969328 | 2.924144 | 2.946736 |

| Modèle | MAE (pp) | RMSE (pp) | Biais (pp) | Pire erreur absolue (pp) |
|---|---:|---:|---:|---:|
| Model A | 0.942278 | 1.090171 | +0.721244 | 1.661326 |
| Model B P1_G1 | 0.946731 | 1.078687 | +0.695575 | 1.628247 |
| Ensemble 50/50 | 0.944505 | 1.084323 | +0.708410 | 1.644787 |

**Prévision 2026 au cutoff 2025-12-31 :**

- Model A : **2.443645 %**
- Model B P1_G1 : **2.510385 %**
- Ensemble final 50/50 : **2.477015 %**

Ces résultats proviennent des APIs et du wrapper du dépôt. Le notebook recalcule et affiche les résultats à l’exécution.

## Architecture

- `notebooks/backup_ensemble_50_50.ipynb` — livrable principal : contexte métier, définitions, protocole anti-leakage, backtests et prévision calculés.
- `src/models/ensemble_50_50.py` — wrapper 50/50 appelant Model A et Model B P1_G1 sans dupliquer leurs pipelines.
- `src/models/model_a.py`, `src/models/model_b/` — implémentations finales des modèles; configurations inchangées par le wrapper.
- `src/features/`, `src/evaluation/` — foundation commune pour la cohorte, les cibles et les cutoffs.
- `app.py` — dashboard de démonstration facultatif.
- `docs/backup_ensemble_50_50.md`, `docs/backup_dashboard.md`, `docs/external_sources.md` — résultats, dashboard et provenance des signaux publics.

## Exécution

Python testé : **3.12.10**. Les dépendances principales et versions testées sont listées dans `requirements.txt` (notamment pandas 2.3.3, NumPy 2.0.2, scikit-learn 1.6.1 et Streamlit 1.65.0).

1. Installer les dépendances dans un environnement virtuel Python 3.12 :

   ```powershell
   python -m pip install -r requirements.txt
   ```

2. Obtenir l’autorisation d’utiliser les données confidentielles et placer localement `equinoxe_lease_history.csv` dans `data/raw/`. Les fichiers CRM ne sont pas distribués avec le dépôt et doivent rester hors de Git.

3. Ouvrir `notebooks/backup_ensemble_50_50.ipynb` depuis le dépôt et exécuter les cellules de haut en bas. Le notebook trouve la racine du projet relativement à son répertoire courant; il ne dépend pas d’un chemin utilisateur Windows codé en dur. Sans le fichier CRM autorisé, la cellule d’entrée s’arrête avec une explication.

4. Pour le dashboard facultatif :

   ```powershell
   python -m streamlit run app.py
   ```

   Le mode upload lit les fichiers CSV en mémoire et permet l’utilisation sans fichiers CRM dans le dépôt. Les téléchargements ne contiennent que des résultats agrégés.

Le wrapper ne modifie pas les modèles, leurs configurations, les poids, la cible, les cutoffs ni les règles anti-leakage. Pour la méthode et ses limites, voir [la note du backup](docs/backup_ensemble_50_50.md).

## Données, confidentialité et méthode

La clé same-unit est `sPropCode + sUnitCode`; `sSite + sUnitCode` n’est pas la clé officielle. Le target principal s’appuie sur le loyer effectif fourni, distinct du loyer contractuel; les concessions peuvent rendre leurs évolutions différentes. Les renouvellements et relocations peuvent aussi réagir différemment. Le notebook explique l’effet de composition, les limites de l’agrégat, ainsi que la séparation réglementaire entre les cinq immeubles du Québec et The Met en Ontario.

Cutoffs historiques : 2023 → 2022-12-31, 2024 → 2023-12-31, 2025 → 2024-12-31; cutoff du forecast 2026 : 2025-12-31. Aucun résultat post-cutoff ne sert à produire la prédiction.

Sources publiques étudiées :

- [SCHL / CMHC](https://www.cmhc-schl.gc.ca/)
- [Statistique Canada, table 18-10-0004-01](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1810000401)
- [Tribunal administratif du logement du Québec](https://www.tal.gouv.qc.ca/)
- [Ontario rent increase guideline](https://www.ontario.ca/page/rent-increase-guideline)

Les sources externes ne sont intégrées quantitativement que lorsque leur disponibilité point-in-time est vérifiée; les signaux non qualifiés restent du contexte/sensibilité. `data/raw/` est ignoré par Git. **Aucun CSV CRM, identifiant de locataire ou export de prédictions par unité ne doit être publié.** Le notebook et le dashboard ne présentent que des dimensions de données et des résultats agrégés.

## Limites

- Trois folds historiques seulement; les estimations d’erreur sont fragiles.
- Une dérive du taux de transition a été observée en 2025.
- P1 est un indice événementiel sur les unités à échéance et non la croissance complète du rent roll.
- L’historique CRM n’est pas un journal versionné; la disponibilité réelle de chaque valeur rétrospective n’est pas toujours prouvable.
- Les données externes historiquement disponibles sont limitées; les signaux réglementaires et de marché ne sont pas interchangeables avec le target.
- Model A et Model B reposent sur des hypothèses distinctes; le 50/50 est un fallback à poids fixes, pas un ensemble optimisé.
- Les données CRM sont confidentielles et doivent rester locales.

## AI tools used

- ChatGPT / OpenAI
- GitHub Copilot

Les outils d’IA ont assisté le développement et la documentation; les valeurs publiées par le notebook sont recalculées par les pipelines du dépôt.