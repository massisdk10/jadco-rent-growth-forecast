# JADCO backup dashboard

## Lancement

Depuis la racine du dépôt, installer la dépendance UI isolée puis démarrer Streamlit :

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Le manifeste principal inclut les dépendances testées des modèles et Streamlit. `requirements-dashboard.txt` reste une option minimale pour les environnements dans lesquels ces dépendances sont déjà installées. Les déploiements Streamlit Community Cloud utilisent `requirements.txt` à la racine.

## Sources de données

La sidebar permet de choisir les CSV locaux du dépôt ou de téléverser les quatre fichiers attendus. Les octets téléversés sont lus directement en mémoire. Aucun téléversement n’est écrit dans le dépôt, et aucune donnée CRM ligne à ligne n’est rendue à l’écran ni incluse à l’export.

Pour un déploiement public depuis GitHub, utiliser la branche `backup-ensemble` et `app.py` comme fichier principal. Aucun dataset CRM n’est nécessaire dans le dépôt : choisir **Upload CSVs** dans la sidebar et fournir les fichiers autorisés en mémoire. Les fichiers ne sont pas persistés par l’application; vérifier les règles de confidentialité applicables avant tout déploiement ou téléversement.

Le forecast utilise uniquement `equinoxe_lease_history.csv`. Listings, concessions et asking history sont acceptés pour compatibilité, mais n’entrent pas dans les modèles finaux ; l’API Model A conserve l’argument asking sans l’utiliser comme prédicteur. Seules la structure du CSV, le nombre de lignes/colonnes et le statut de chargement sont affichés.

La prévision est désactivée si lease history est absente, illisible ou n’a pas les colonnes attendues par les APIs finales. Les erreurs détaillées du parseur ne sont pas affichées afin d’éviter de révéler des extraits de lignes.

## Résultats

Le bouton **Run 2026 Forecast** appelle les APIs Model A, Model B P1_G1 et le wrapper fixe `estimate_2026_ensemble`. Le backtest affiche seulement les résultats agrégés 2023–2025 et des métriques portefeuille. Le téléchargement JSON contient les trois forecasts, poids fixes et métriques agrégées uniquement.

Les poids restent exactement 50/50, non optimisés. Les fichiers/confidentialité CRM restent sous responsabilité de l’utilisateur : exécuter le dashboard localement et ne pas partager les CSV téléversés.
