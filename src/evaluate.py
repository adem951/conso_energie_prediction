from __future__ import annotations

import json
import sys

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from src.config import MODEL_NAME, REPORTS_DIR, configure_mlflow, load_params, write_summary
from src.data import load_dataset, temporal_split
from src.features import FEATURE_COLUMNS
from src.models import score


def load_alias(alias: str):
    """Return (version, model) for an alias of the registry, or (None, None) if it does not exist."""
    try:
        version = mlflow.MlflowClient().get_model_version_by_alias(MODEL_NAME, alias)
    except mlflow.exceptions.MlflowException:
        return None, None
    return version, mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{alias}")


def forecast_figure(test: pd.DataFrame) -> plt.Figure:
    last_weeks = test[test["timestamp"] >= test["timestamp"].max() - pd.Timedelta(days=14)]
    figure, axis = plt.subplots(figsize=(12, 4))
    axis.plot(last_weeks["timestamp"], last_weeks["consommation_mw"], label="réel", color="black")
    axis.plot(last_weeks["timestamp"], last_weeks["prediction"], label="prévision @staging")
    axis.set(title="Réel vs prévision : 2 dernières semaines du test", ylabel="MW")
    axis.legend()
    return figure


def error_by_hour_figure(test: pd.DataFrame) -> plt.Figure:
    error = (test["prediction"] - test["consommation_mw"]).abs().groupby(test["hour"]).mean()
    figure, axis = plt.subplots(figsize=(8, 4))
    error.plot.bar(ax=axis, rot=0, title="Erreur absolue moyenne par heure (heure de Paris)")
    axis.set(xlabel="Heure", ylabel="MAE (MW)")
    return figure


def evaluate() -> dict:
    configure_mlflow()
    _, _, test = temporal_split(load_dataset(), load_params()["split"])
    test = test.copy()

    evaluation = {"test_start": str(test["timestamp"].min()), "test_end": str(test["timestamp"].max())}
    for alias in ["staging", "production"]:
        version, model = load_alias(alias)
        if model is None:
            continue
        predictions = model.predict(test[FEATURE_COLUMNS])
        evaluation[alias] = {"version": version.version, **score(test["consommation_mw"], predictions)}
        if alias == "staging":
            test["prediction"] = predictions
            staging_run_id = version.run_id

    # Figures du candidat, enregistrées dans son run MLflow et dans reports/.
    REPORTS_DIR.mkdir(exist_ok=True)
    figures = {"forecast_test.png": forecast_figure(test), "error_by_hour.png": error_by_hour_figure(test)}
    with mlflow.start_run(run_id=staging_run_id):
        for name, figure in figures.items():
            mlflow.log_figure(figure, name)
            figure.savefig(REPORTS_DIR / name, bbox_inches="tight")
    mlflow.MlflowClient().set_model_version_tag(
        MODEL_NAME, evaluation["staging"]["version"], "test_rmse", f"{evaluation['staging']['rmse']:.1f}"
    )
    (REPORTS_DIR / "evaluation.json").write_text(json.dumps(evaluation, indent=2))

    rows = [
        f"| @{alias} | v{evaluation[alias]['version']} | {evaluation[alias]['rmse']:.0f} | "
        f"{evaluation[alias]['mae']:.0f} | {evaluation[alias]['r2']:.3f} |"
        for alias in ["staging", "production"] if alias in evaluation
    ]
    write_summary(
        f"## Évaluation sur le test ({evaluation['test_start'][:10]} → {evaluation['test_end'][:10]})\n\n"
        "| Alias | Version | RMSE | MAE | R2 |\n|---|---|---|---|---|\n" + "\n".join(rows) + "\n"
    )
    return evaluation


if __name__ == "__main__":
    matplotlib.use("Agg")  # figures enregistrées en fichiers, sans affichage
    sys.stdout.reconfigure(encoding="utf-8")
    evaluate()
