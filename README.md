# Prévision de consommation électrique

MVP de prévision day-ahead de la consommation électrique à pas de 30 minutes à partir des données régionales RTE éCO2mix.

## Objectif

Prévoir la consommation future en respectant strictement l'ordre temporel :

- aucun mélange aléatoire des observations ;
- les trois derniers mois sont conservés pour le test final ;
- `lag_48` représente 24 heures ;
- `lag_336` représente 7 jours ;
- les variables de production ne sont pas utilisées comme features pour éviter la fuite de données.

## Structure

```text
data/                         Données locales et données préparées
notebooks/                    EDA, features et entraînement MLflow
.github/workflows/ci.yml      Validation GitHub Actions
dvc.yaml                      Pipeline de préparation DVC
requirements.txt              Dépendances Python
scripts/start_mlflow.ps1      Démarrage du serveur MLflow local
```

## Installation

Créer et activer l'environnement :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Exécution des notebooks

Exécuter dans cet ordre :

1. `01_eda_preparation_consommation.ipynb` : contrôle qualité et préparation des séries nationale et régionales ;
2. `02_features_split_temporel.ipynb` : création des lags et séparation temporelle ;
3. `03_entrainement_mlflow.ipynb` : comparaison baseline, LightGBM intermédiaire et production.

## MLflow

Le tracking utilise SQLite dans `mlflow.db`. Pour démarrer l'interface :

```powershell
.\scripts\start_mlflow.ps1
```

Puis ouvrir <http://127.0.0.1:5000>.

Les métriques suivies sont RMSE, MAE et R2. Le modèle production utilise une validation temporelle et l'early stopping pour limiter l'overfitting.

## DVC

Le pipeline de préparation est défini dans `dvc.yaml` :

```powershell
dvc dag
dvc repro
```

Le stockage distant DVC doit être configuré séparément selon l'emplacement choisi pour les données volumineuses. Les données brutes et générées ne doivent pas être poussées directement dans GitHub.

## GitHub Actions

Le workflow `.github/workflows/ci.yml` se déclenche sur chaque `push` et `pull_request`. Il installe les dépendances, exécute le notebook de features et vérifie le graphe DVC.

## Limites actuelles

Ce MVP prévoit la consommation France entière. Les données régionales sont également exportées dans `data/eco2mix_regional.csv` pour une prochaine version de prévision par région. Les données météo et les jours fériés pourront ensuite améliorer le modèle, à condition de respecter leur disponibilité au moment de la prévision.