from __future__ import annotations

import os
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
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": 42,
        "verbosity": -1,
    }
    model = LGBMRegressor(**parameters)
    model.fit(
        X_fit,
        y_fit,
        eval_set=[(X_fit, y_fit), (X_validation, y_validation)],
        eval_names=["train", "validation"],
        eval_metric="rmse",
        callbacks=[early_stopping(50, verbose=False)],
    )

    scores = calculate_metrics(y_test, model.predict(X_test))
    with mlflow.start_run(run_name="lightgbm_production_script"):
        mlflow.set_tag("stage", "production")
        mlflow.log_params(parameters)
        mlflow.log_param("best_iteration", model.best_iteration_)
        mlflow.log_metrics(scores)
        mlflow.lightgbm.log_model(model, "model")

    print({"model": "lightgbm_production", **scores})
    return scores


if __name__ == "__main__":
    train_production_model()
