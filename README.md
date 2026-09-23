# Prévision de consommation électrique

MVP de prévision day-ahead de la consommation électrique France à pas de 30 minutes, à partir des données régionales RTE éCO2mix.

## Objectif

Prévoir les 48 demi-heures de demain en respectant strictement l'ordre temporel :

- aucun mélange aléatoire des observations ;
- les trois derniers mois sont conservés pour le test final ;
- seuls des lags d'au moins 24 h sont utilisés (`lag_48` = 24 h, `lag_96` = 48 h, `lag_336` = 7 jours) : toute la journée de demain se prédit d'un coup, sans réutiliser ses propres prédictions ;
- les variables calendaires sont calculées en heure de Paris ;
- les variables de production ne sont pas utilisées comme features pour éviter la fuite de données.

## Architecture

```text
GitHub ──► Streamlit Cloud (app.py)
                 │ charge le dernier run stage=production
                 ▼
DagsHub ── MLflow (runs, métriques, modèle)
        └─ DVC    (données volumineuses)
```

## Structure

```text
app.py                        Application Streamlit
src/prepare_data.py           Préparation des séries nationale et régionales
src/features.py               Variables calendaires et lags
src/train.py                  Entraînement LightGBM + baseline, log MLflow
tests/                        Tests pytest
exemples/historique_7_jours.csv  Historique d'exemple utilisé par l'app
notebooks/                    EDA, features et comparaison de modèles
dvc.yaml                      Pipeline DVC (préparation puis entraînement)
requirements.txt              Dépendances de l'application (Streamlit Cloud)
requirements-dev.txt          Dépendances de développement
scripts/start_mlflow.ps1      Serveur MLflow local
```

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

## Connexion DagsHub

Le remote DVC `dagshub` est déjà déclaré dans `.dvc/config`. Les identifiants restent en local (jamais commités) :

```powershell
dvc remote modify --local dagshub user adem.debbahi
dvc remote modify --local dagshub password TON_TOKEN_DAGSHUB

$env:MLFLOW_TRACKING_URI = "https://dagshub.com/adem.debbahi/conso_energie_prediction.mlflow"
$env:MLFLOW_TRACKING_USERNAME = "adem.debbahi"
$env:MLFLOW_TRACKING_PASSWORD = "TON_TOKEN_DAGSHUB"
```

Sans `MLFLOW_TRACKING_URI`, le tracking se fait en local dans `mlflow.db`.

## Mettre à jour le modèle

```powershell
dvc pull            # récupère les données depuis DagsHub
dvc repro           # prépare les données puis entraîne (log dans MLflow DagsHub)
dvc push            # envoie les données sur DagsHub
git add dvc.lock metrics.json
git commit -m "Nouveau modèle"
git push            # Streamlit Cloud se met à jour
```

`src.train` enregistre deux runs : `baseline` (persistance 24 h) et `production` (LightGBM avec early stopping). Les métriques (RMSE, MAE, R2) sont aussi écrites dans `metrics.json` (`dvc metrics show`).

## Streamlit Cloud

Sélectionner ce dépôt GitHub et le fichier `app.py`, puis ajouter dans **Settings > Secrets** :

```toml
MLFLOW_TRACKING_URI = "https://dagshub.com/adem.debbahi/conso_energie_prediction.mlflow"
MLFLOW_TRACKING_USERNAME = "adem.debbahi"
MLFLOW_TRACKING_PASSWORD = "TON_TOKEN_DAGSHUB"
MLFLOW_EXPERIMENT_NAME = "conso_energie_day_ahead"
```

L'application charge le dernier run `stage=production` et propose deux modes :

- **Prévision rapide** : une date, une heure et trois curseurs (consommation 24 h, 48 h et 7 jours avant) pour prédire une demi-heure ;
- **Prévoir demain** : les 48 demi-heures de demain à partir d'un CSV `timestamp,consommation_mw` couvrant les 7 derniers jours ; sans fichier, `exemples/historique_7_jours.csv` est utilisé.

Pour lancer l'app en local : `streamlit run app.py`.

## Notebooks

1. `01_eda_preparation_consommation.ipynb` : contrôle qualité des données ;
2. `02_features_split_temporel.ipynb` : lags et séparation temporelle ;
3. `03_entrainement_mlflow.ipynb` : comparaison baseline / LightGBM.

Les notebooks servent à l'exploration. Les entrées de référence sont `python -m src.prepare_data` et `python -m src.train`.

## GitHub Actions

`.github/workflows/ci.yml` s'exécute à chaque `push` et `pull_request` : installation des dépendances, tests pytest et vérification du graphe DVC.

## Limites actuelles

Ce MVP prévoit la consommation France entière. Les données régionales sont exportées dans `data/eco2mix_regional.csv` pour une future prévision par région. La météo et les jours fériés pourront améliorer le modèle, à condition de respecter leur disponibilité au moment de la prévision.
