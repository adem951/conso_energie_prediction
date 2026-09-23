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
from mlflow.entities import Metric
from mlflow.models import infer_signature

from src.config import (
    FIGURE_OPTIONS,
    MODEL_NAME,
    RECENT_PATH,
    REPORTS_DIR,
    ROOT,
    configure_mlflow,
    lineage,
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


def log_learning_curve(model) -> None:
    """RMSE per tree as step metrics: an interactive chart in the MLflow UI."""
    run_id = mlflow.active_run().info.run_id
    metrics = [
        Metric(f"courbe_rmse_{name}", value, timestamp=0, step=step)
        for name, curve in model.evals_result_.items()
        for step, value in enumerate(curve["rmse"])
    ]
    for start in range(0, len(metrics), 1000):  # 1 000 valeurs max par envoi
        mlflow.MlflowClient().log_batch(run_id, metrics=metrics[start:start + 1000])


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


def describe(best: str, version: str, results: dict, trace: dict, periods: dict) -> str:
    """Markdown shown at the top of the run and of the model version in MLflow."""
    rows = "\n".join(
        f"| {name} | {scores['train_rmse']:.0f} | {scores['val_rmse']:.0f} | {scores['test_rmse']:.0f} |"
        for name, scores in results.items()
    )
    return (
        f"**Modèle retenu : {best} → {MODEL_NAME} v{version} @staging** (meilleure RMSE de validation)\n\n"
        f"| Modèle | RMSE train | RMSE validation | RMSE test |\n|---|---|---|---|\n{rows}\n\n"
        f"**Données** : historique DVC `{trace['data_version']}` (voir `dvc.lock` dans les artefacts) "
        f"+ données RTE récentes `{trace['recent_data_md5']}` (artefact `data/recent.csv`)\n\n"
        f"**Périodes** : train depuis le {periods['train_start']}, validation depuis le {periods['val_start']}, "
        f"test du {periods['test_start']} au {periods['data_end']}\n\n"
        f"**Code** : commit `{trace['git_commit']}` · paramètres dans `params.yaml`"
    )


def train() -> str:
    params = load_params()
    configure_mlflow()
    fit, validation, test = temporal_split(load_dataset(), params["split"])
    sets = {"train": fit, "val": validation, "test": test}
    trace = lineage()
    periods = {
        **{f"{name}_start": f"{frame['timestamp'].min():%d/%m/%Y}" for name, frame in sets.items()},
        "data_end": f"{test['timestamp'].max():%d/%m/%Y}",
    }
    started = pd.Timestamp.now(tz="Europe/Paris")

    results, run_ids = {}, {}
    with mlflow.start_run(run_name=f"entrainement {started:%d/%m %H:%M}") as parent:
        mlflow.set_tag("stage", "training")
        # En paramètres (et pas seulement en tags) : visibles directement dans le tableau des runs.
        mlflow.log_params({**trace, **periods, **{f"split_{key}": value for key, value in params["split"].items()}})
        mlflow.log_artifact(str(ROOT / "dvc.lock"))
        if RECENT_PATH.exists():
            mlflow.log_artifact(str(RECENT_PATH), artifact_path="data")  # instantané des données récentes

        for name, column in NAIVE_MODELS.items():
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.set_tags({"stage": "candidate", "model_type": name})
                mlflow.log_params(trace)
                results[name] = score_all_sets(sets, lambda frame, column=column: frame[column])
                mlflow.log_metrics(results[name])

        with mlflow.start_run(run_name="linear_regression", nested=True) as run:
            mlflow.set_tags({"stage": "candidate", "model_type": "linear_regression"})
            mlflow.log_params(trace)
            model = linear_regression().fit(fit[FEATURE_COLUMNS], fit[TARGET])
            results["linear_regression"] = score_all_sets(sets, lambda frame: model.predict(frame[FEATURE_COLUMNS]))
            mlflow.log_metrics(results["linear_regression"])
            log_model(model, mlflow.sklearn, fit[FEATURE_COLUMNS].head())
            run_ids["linear_regression"] = run.info.run_id

        with mlflow.start_run(run_name="lightgbm", nested=True) as run:
            mlflow.set_tags({"stage": "candidate", "model_type": "lightgbm"})
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
            mlflow.log_params({**trace, **params["lightgbm"], "best_iteration": model.best_iteration_})
            mlflow.log_metrics(results["lightgbm"])
            log_learning_curve(model)
            mlflow.log_figure(learning_curve_figure(model), "learning_curve.png", save_kwargs=FIGURE_OPTIONS)
            mlflow.log_figure(importance_figure(model), "feature_importance.png", save_kwargs=FIGURE_OPTIONS)
            log_model(model, mlflow.lightgbm, fit[FEATURE_COLUMNS].head())
            run_ids["lightgbm"] = run.info.run_id

        # Sélection sur la validation : le test reste réservé à l'évaluation finale.
        best = min(run_ids, key=lambda name: results[name]["val_rmse"])
        mlflow.log_param("selected_model", best)
        mlflow.log_figure(comparison_figure(results), "model_comparison.png", save_kwargs=FIGURE_OPTIONS)

    version = mlflow.register_model(f"runs:/{run_ids[best]}/model", MODEL_NAME).version
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(MODEL_NAME, "staging", version)
    description = describe(best, version, results, trace, periods)
    client.update_model_version(MODEL_NAME, version, description=description)
    for key, value in {**trace, "model_type": best, "val_rmse": round(results[best]["val_rmse"], 1)}.items():
        client.set_model_version_tag(MODEL_NAME, version, key, str(value))

    # Relie les runs au registre : nom explicite, description et alias visibles depuis l'onglet Experiments.
    client.set_tag(parent.info.run_id, "mlflow.runName", f"entrainement {started:%d/%m %H:%M} → v{version} {best} @staging")
    for run_id in [parent.info.run_id, run_ids[best]]:
        client.set_tag(run_id, "mlflow.note.content", description)
    client.set_tag(run_ids[best], "registry", f"v{version} @staging")

    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "train_metrics.json").write_text(json.dumps(results, indent=2))
    comparison_figure(results).savefig(REPORTS_DIR / "model_comparison.png", **FIGURE_OPTIONS)

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
