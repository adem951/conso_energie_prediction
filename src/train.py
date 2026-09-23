from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import pandas as pd
from lightgbm import early_stopping
from mlflow.models import infer_signature

from src.config import (
    MODEL_NAME,
    RECENT_PATH,
    REPORTS_DIR,
    configure_mlflow,
    data_tags,
    git_commit,
    load_params,
    write_output,
    write_summary,
)
from src.data import load_dataset, temporal_split
from src.features import FEATURE_COLUMNS
from src.models import NAIVE_MODELS, lightgbm, linear_regression, score

TARGET = "consommation_mw"


def score_all_sets(sets: dict[str, pd.DataFrame], predict) -> dict[str, float]:
    """train / val / test metrics: the gap between them shows overfitting."""
    return {
        f"{set_name}_{metric}": value
        for set_name, frame in sets.items()
        for metric, value in score(frame[TARGET], predict(frame)).items()
    }


def log_model(model, flavor, example: pd.DataFrame) -> None:
    # save_model + log_artifacts : artefact `model/` lisible par le serveur MLflow de DagsHub.
    signature = infer_signature(example, model.predict(example))
    with tempfile.TemporaryDirectory() as temporary_directory:
        path = Path(temporary_directory) / "model"
        flavor.save_model(model, path=str(path), signature=signature)
        mlflow.log_artifacts(str(path), artifact_path="model")


def learning_curve_figure(model) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(8, 4))
    for name, curve in model.evals_result_.items():
        axis.plot(curve["rmse"], label=name)
    axis.axvline(model.best_iteration_, color="grey", linestyle="--", label="early stopping")
    axis.set(title="LightGBM : courbe d'apprentissage", xlabel="Nombre d'arbres", ylabel="RMSE (MW)")
    axis.legend()
    return figure


def importance_figure(model) -> plt.Figure:
    importance = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values()
    figure, axis = plt.subplots(figsize=(8, 4))
    importance.plot.barh(ax=axis, title="LightGBM : importance des variables (nombre de splits)")
    return figure


def comparison_figure(results: dict[str, dict[str, float]]) -> plt.Figure:
    table = pd.DataFrame(results).T[["train_rmse", "val_rmse", "test_rmse"]]
    figure, axis = plt.subplots(figsize=(9, 4))
    table.plot.bar(ax=axis, rot=0, title="RMSE par modèle et par jeu de données")
    axis.set_ylabel("RMSE (MW)")
    return figure


def train() -> str:
    params = load_params()
    configure_mlflow()
    fit, validation, test = temporal_split(load_dataset(), params["split"])
    sets = {"train": fit, "val": validation, "test": test}
    tags = {"git_commit": git_commit(), **data_tags()}

    results, run_ids = {}, {}
    with mlflow.start_run(run_name="training"):
        mlflow.set_tags({**tags, "stage": "training"})
        mlflow.log_params({
            **{f"split_{key}": value for key, value in params["split"].items()},
            **{f"{name}_start": str(frame["timestamp"].min()) for name, frame in sets.items()},
            "data_end": str(test["timestamp"].max()),
        })
        if RECENT_PATH.exists():
            mlflow.log_artifact(str(RECENT_PATH), artifact_path="data")  # instantané des données récentes

        for name, column in NAIVE_MODELS.items():
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.set_tags({**tags, "stage": "candidate", "model_type": name})
                results[name] = score_all_sets(sets, lambda frame, column=column: frame[column])
                mlflow.log_metrics(results[name])

        with mlflow.start_run(run_name="linear_regression", nested=True) as run:
            mlflow.set_tags({**tags, "stage": "candidate", "model_type": "linear_regression"})
            model = linear_regression().fit(fit[FEATURE_COLUMNS], fit[TARGET])
            results["linear_regression"] = score_all_sets(sets, lambda frame: model.predict(frame[FEATURE_COLUMNS]))
            mlflow.log_metrics(results["linear_regression"])
            log_model(model, mlflow.sklearn, fit[FEATURE_COLUMNS].head())
            run_ids["linear_regression"] = run.info.run_id

        with mlflow.start_run(run_name="lightgbm", nested=True) as run:
            mlflow.set_tags({**tags, "stage": "candidate", "model_type": "lightgbm"})
            model = lightgbm(params["lightgbm"])
            model.fit(
                fit[FEATURE_COLUMNS], fit[TARGET],
                eval_X=(fit[FEATURE_COLUMNS], validation[FEATURE_COLUMNS]),
                eval_y=(fit[TARGET], validation[TARGET]),
                eval_names=["train", "validation"],
                eval_metric="rmse",
                callbacks=[early_stopping(params["lightgbm"]["early_stopping_rounds"], verbose=False)],
            )
            results["lightgbm"] = score_all_sets(sets, lambda frame: model.predict(frame[FEATURE_COLUMNS]))
            mlflow.log_params({**params["lightgbm"], "best_iteration": model.best_iteration_})
            mlflow.log_metrics(results["lightgbm"])
            mlflow.log_figure(learning_curve_figure(model), "learning_curve.png")
            mlflow.log_figure(importance_figure(model), "feature_importance.png")
            log_model(model, mlflow.lightgbm, fit[FEATURE_COLUMNS].head())
            run_ids["lightgbm"] = run.info.run_id

        # Sélection sur la validation : le test reste réservé à l'évaluation finale.
        best = min(run_ids, key=lambda name: results[name]["val_rmse"])
        mlflow.log_param("selected_model", best)
        mlflow.log_figure(comparison_figure(results), "model_comparison.png")

    version = mlflow.register_model(f"runs:/{run_ids[best]}/model", MODEL_NAME).version
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(MODEL_NAME, "staging", version)
    for key, value in {**tags, "model_type": best, "val_rmse": round(results[best]["val_rmse"], 1)}.items():
        client.set_model_version_tag(MODEL_NAME, version, key, str(value))

    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "train_metrics.json").write_text(json.dumps(results, indent=2))
    comparison_figure(results).savefig(REPORTS_DIR / "model_comparison.png", bbox_inches="tight")

    table = pd.DataFrame(results).T[["train_rmse", "val_rmse", "test_rmse", "test_mae", "test_r2"]].round(3)
    write_summary(
        f"## Entraînement\n\n{table.to_markdown()}\n\n"
        f"Modèle retenu (meilleure RMSE de validation) : **{best}** → `{MODEL_NAME}` v{version} `@staging`\n"
    )
    write_output("model_version", version)
    return version


if __name__ == "__main__":
    matplotlib.use("Agg")  # figures enregistrées en fichiers, sans affichage
    # MLflow affiche des emojis : évite une erreur d'encodage dans la console Windows.
    sys.stdout.reconfigure(encoding="utf-8")
    train()
