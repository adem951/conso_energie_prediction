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

Les notebooks servent à l'exploration et à la lecture des résultats. Les entrées de production sont `python -m src.prepare_data` et `python -m src.train`.

## MLflow

Le tracking utilise SQLite dans `mlflow.db`. Pour démarrer l'interface :

```powershell
.\scripts\start_mlflow.ps1
```

Puis ouvrir <http://127.0.0.1:5000>.

## Déploiement portable de l'application

L'application ne lit pas `mlflow.db`, `mlruns` ou les données locales en production. Elle charge le dernier run marqué `stage=production` depuis un serveur MLflow distant. Cela la rend portable sur Streamlit Community Cloud ou dans Docker.

### Streamlit Community Cloud

Déployer le fichier `app.py` depuis GitHub, puis ajouter ces secrets dans **Settings > Secrets** :

```toml
MLFLOW_TRACKING_URI = "https://dagshub.com/UTILISATEUR/DEPOT.mlflow"
MLFLOW_TRACKING_USERNAME = "UTILISATEUR_DAGSHUB"
MLFLOW_TRACKING_PASSWORD = "TOKEN_DAGSHUB"
MLFLOW_EXPERIMENT_NAME = "conso_energie_day_ahead"
```

Le token ne doit jamais être écrit dans GitHub, le code ou `requirements.txt`.

L'interface ne demande pas de CSV : elle demande quatre valeurs de référence (`lag_1`, `lag_48`, `lag_96`, `lag_336`) et calcule automatiquement les variables calendaires. Le bouton **Prévoir la consommation de demain** génère les 48 demi-heures futures de façon récursive. Pour une prévision industrielle, ces quatre valeurs doivent être alimentées automatiquement par une API de mesures ou une base de données récente ; les champs sont un mode MVP de démonstration.

### Docker

Construire et lancer l'application :

```powershell
docker build -t conso-energie-app .
docker run --rm -p 8501:8501 `
	-e MLFLOW_TRACKING_URI="https://dagshub.com/UTILISATEUR/DEPOT.mlflow" `
	-e MLFLOW_TRACKING_USERNAME="UTILISATEUR_DAGSHUB" `
	-e MLFLOW_TRACKING_PASSWORD="TOKEN_DAGSHUB" `
	conso-energie-app
```

Docker Hub est optionnel : il sert à publier cette image pour la déployer sur un autre cloud ou serveur. Streamlit Community Cloud peut utiliser directement GitHub sans Docker Hub.

Les métriques suivies sont RMSE, MAE et R2. Le modèle production utilise une validation temporelle et l'early stopping pour limiter l'overfitting.

### DagsHub

DagsHub peut héberger le tracking MLflow et un remote DVC pour ce MVP. Après avoir créé le dépôt `adem951/conso_energie_prediction` sur DagsHub :

```powershell
dvc remote add -d dagshub https://dagshub.com/adem951/conso_energie_prediction.dvc
$env:MLFLOW_TRACKING_URI = "https://dagshub.com/adem951/conso_energie_prediction.mlflow"
$env:MLFLOW_TRACKING_USERNAME = "adem951"
$env:MLFLOW_TRACKING_PASSWORD = "TON_TOKEN_DAGSHUB"
python -m src.train
dvc push
```

Le token ne doit jamais être écrit dans le code ou commité. Dans GitHub Actions, utiliser les secrets `DAGSHUB_USERNAME`, `DAGSHUB_TOKEN` et `MLFLOW_TRACKING_URI`.

## DVC

Le pipeline de préparation est défini dans `dvc.yaml` :

```powershell
dvc dag
dvc repro
```

Le stockage distant DVC doit être configuré séparément selon l'emplacement choisi pour les données volumineuses. Les données brutes et générées ne doivent pas être poussées directement dans GitHub.

## GitHub Actions

Le workflow `.github/workflows/ci.yml` se déclenche sur chaque `push` et `pull_request`. Il installe les dépendances, compile les sources Python, construit l'image Docker et vérifie le graphe DVC.

## Streamlit Community Cloud

Dans Streamlit Community Cloud, sélectionner ce dépôt GitHub et le fichier `app.py`. Ajouter dans **Secrets** :

```toml
MLFLOW_TRACKING_URI = "https://dagshub.com/adem951/conso_energie_prediction.mlflow"
MLFLOW_TRACKING_USERNAME = "adem951"
MLFLOW_TRACKING_PASSWORD = "TON_TOKEN_DAGSHUB"
```

L'application charge le dernier run tagué `stage=production` dans MLflow et accepte un CSV contenant `timestamp` et `consommation_mw`.

## Docker Hub

Docker est utile pour déployer l'application Streamlit sur une plateforme qui accepte des conteneurs. Le modèle n'est pas copié dans l'image : l'application le charge depuis MLflow DagsHub.

```powershell
docker build -t adem951/conso-energie-streamlit:latest .
docker push adem951/conso-energie-streamlit:latest
```

Les variables MLflow doivent être injectées comme secrets dans la plateforme de déploiement, jamais dans le Dockerfile.

## Limites actuelles

Ce MVP prévoit la consommation France entière. Les données régionales sont également exportées dans `data/eco2mix_regional.csv` pour une prochaine version de prévision par région. Les données météo et les jours fériés pourront ensuite améliorer le modèle, à condition de respecter leur disponibilité au moment de la prévision.