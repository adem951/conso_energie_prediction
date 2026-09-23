from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.features import FEATURE_COLUMNS, build_features


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "eco2mix.csv"
METRICS_PATH = ROOT / "metrics.json"


def calculate_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
    }


def configure_mlflow() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "conso_energie_day_ahead"))


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    data = build_features(data).dropna(subset=FEATURE_COLUMNS + ["consommation_mw"])

    test_start = data["timestamp"].max() - pd.DateOffset(months=3)
    train = data[data["timestamp"] < test_start].copy()
    test = data[data["timestamp"] >= test_start].copy()
    validation_start = train["timestamp"].max() - pd.DateOffset(months=1)
    fit = train[train["timestamp"] < validation_start].copy()
    validation = train[train["timestamp"] >= validation_start].copy()
    return fit, validation, test


def train_production_model() -> dict[str, float]:
    configure_mlflow()
    fit, validation, test = load_datasets()
    X_fit, y_fit = fit[FEATURE_COLUMNS], fit["consommation_mw"]
    X_validation, y_validation = validation[FEATURE_COLUMNS], validation["consommation_mw"]
    X_test, y_test = test[FEATURE_COLUMNS], test["consommation_mw"]

    parameters = {
        "learning_rate": 0.05,
        "n_estimators": 1000,
        "num_leaves": 31,
        "max_depth": -1,
        "colsample_bytree": 0.9,
        "random_state": 42,
        "verbosity": -1,
    }
    model = LGBMRegressor(**parameters)
    model.fit(
        X_fit,
        y_fit,
        eval_X=X_validation,
        eval_y=y_validation,
        eval_metric="rmse",
        callbacks=[early_stopping(50, verbose=False)],
    )

    # Référence naïve : demain ressemble à aujourd'hui (lag 24 h).
    baseline_scores = calculate_metrics(y_test, test["lag_48"])
    with mlflow.start_run(run_name="baseline_persistence_24h"):
        mlflow.set_tag("stage", "baseline")
        mlflow.log_metrics(baseline_scores)

    scores = calculate_metrics(y_test, model.predict(X_test))
    with mlflow.start_run(run_name="lightgbm_production_script"):
        mlflow.set_tag("stage", "production")
        mlflow.log_params(parameters)
        mlflow.log_param("best_iteration", model.best_iteration_)
        mlflow.log_metrics(scores)
        # Sauvegarde explicite pour garantir un artefact model/ compatible avec
        # le chargement `runs:/<run_id>/model` utilisé par Streamlit.
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_directory = Path(temporary_directory) / "model"
            mlflow.lightgbm.save_model(model, path=str(model_directory))
            mlflow.log_artifacts(str(model_directory), artifact_path="model")

    METRICS_PATH.write_text(json.dumps({"baseline": baseline_scores, "production": scores}, indent=2))
    print({"baseline": baseline_scores, "production": scores})
    return scores


if __name__ == "__main__":
    # MLflow affiche des emojis : évite une erreur d'encodage dans la console Windows.
    sys.stdout.reconfigure(encoding="utf-8")
    train_production_model()
