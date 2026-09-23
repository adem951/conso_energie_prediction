from __future__ import annotations

import json
import sys

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.config import (
    FIGURE_OPTIONS,
    MODEL_NAME,
    MONITORING_EXPERIMENT,
    REPORTS_DIR,
    configure_mlflow,
    load_params,
    write_output,
    write_summary,
)
from src.data import load_dataset
from src.features import FEATURE_COLUMNS


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Population Stability Index: 0 when both distributions are identical, grows with the shift."""
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    reference_share = np.histogram(reference, edges)[0] / len(reference)
    current_share = np.histogram(current, edges)[0] / len(current)
    reference_share, current_share = np.clip(reference_share, 1e-6, None), np.clip(current_share, 1e-6, None)
    return float(np.sum((current_share - reference_share) * np.log(current_share / reference_share)))


def drift_status(drift: bool) -> str:
    return "🔴 dérive" if drift else "🟢 stable"


def distribution_figure(reference: pd.Series, current: pd.Series) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.hist(reference, bins=40, alpha=0.5, density=True, label="mêmes semaines l'an dernier")
    axis.hist(current, bins=40, alpha=0.5, density=True, label="semaines récentes")
    axis.set(title="Dérive des données : distribution de la consommation", xlabel="MW")
    axis.legend()
    return figure


def weekly_mae_figure(weekly_mae: pd.Series, threshold: float) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(8, 4))
    weekly_mae.plot.bar(ax=axis, rot=0, title="Dérive du modèle : MAE hebdomadaire de @production")
    axis.axhline(threshold, color="red", linestyle="--", label="seuil d'alerte")
    axis.set(xlabel="Semaine", ylabel="MAE (MW)")
    axis.legend()
    return figure


def monitor() -> bool:
    params = load_params()["monitoring"]
    configure_mlflow(MONITORING_EXPERIMENT)
    data = load_dataset()

    # La consommation est saisonnière : on compare aux mêmes semaines un an plus tôt,
    # sinon l'été serait toujours en « dérive » par rapport à l'hiver.
    end = data["timestamp"].max()
    start = end - pd.Timedelta(weeks=params["weeks"])
    current = data[data["timestamp"] > start].copy()
    last_year = data["timestamp"].between(start - pd.DateOffset(years=1), end - pd.DateOffset(years=1))
    reference = data[last_year]

    psi_value = psi(reference["consommation_mw"], current["consommation_mw"])
    ks_pvalue = float(ks_2samp(reference["consommation_mw"], current["consommation_mw"]).pvalue)

    # Dérive du modèle : erreur récente de @production comparée à son erreur de test.
    version = mlflow.MlflowClient().get_model_version_by_alias(MODEL_NAME, "production")
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@production")
    current["error"] = (model.predict(current[FEATURE_COLUMNS]) - current["consommation_mw"]).abs()
    weekly_mae = current.set_index("timestamp")["error"].resample("W-MON", label="left", closed="left").mean()
    weekly_mae = weekly_mae.rename("MAE hebdomadaire").set_axis(weekly_mae.index.strftime("sem. du %d/%m"))
    run_metrics = mlflow.get_run(version.run_id).data.metrics
    reference_mae = run_metrics.get("test_mae", run_metrics.get("mae"))
    threshold = reference_mae * (1 + params["mae_increase_threshold"])

    result = {
        "window": f"{current['timestamp'].min():%Y-%m-%d} → {end:%Y-%m-%d}",
        "model_version": version.version,
        "psi": psi_value,
        "ks_pvalue": ks_pvalue,
        "mae_recent": float(current["error"].mean()),
        "mae_last_week": float(weekly_mae.iloc[-1]),
        "mae_reference": reference_mae,
        "data_drift": psi_value > params["psi_threshold"],
        "model_drift": float(weekly_mae.iloc[-1]) > threshold,
    }
    retrain = result["data_drift"] or result["model_drift"]

    REPORTS_DIR.mkdir(exist_ok=True)
    figures = {
        "drift_distribution.png": distribution_figure(reference["consommation_mw"], current["consommation_mw"]),
        "drift_weekly_mae.png": weekly_mae_figure(weekly_mae, threshold),
    }
    data_status, model_status = drift_status(result["data_drift"]), drift_status(result["model_drift"])
    report = (
        f"## Monitoring de @production v{version.version} ({result['window']})\n\n"
        "| Contrôle | Valeur | Seuil | Statut |\n|---|---|---|---|\n"
        f"| PSI consommation (vs l'an dernier) | {psi_value:.3f} | {params['psi_threshold']} | {data_status} |\n"
        f"| KS p-value | {ks_pvalue:.3g} | info | |\n"
        f"| MAE dernière semaine | {result['mae_last_week']:.0f} MW | {threshold:.0f} MW | {model_status} |\n\n"
        f"Réentraînement nécessaire : **{'oui' if retrain else 'non'}**\n"
    )

    run_name = f"monitoring {end:%d/%m} · v{version.version} · {'dérive' if retrain else 'stable'}"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({"stage": "monitoring", "mlflow.note.content": report})
        mlflow.log_params({
            "model_version": version.version, "window": result["window"],
            "psi_threshold": params["psi_threshold"], "mae_threshold": round(threshold, 1),
        })
        mlflow.log_metrics({key: float(value) for key, value in result.items() if key not in ["window", "model_version"]})
        for name, figure in figures.items():
            mlflow.log_figure(figure, name, save_kwargs=FIGURE_OPTIONS)
            figure.savefig(REPORTS_DIR / name, **FIGURE_OPTIONS)
    (REPORTS_DIR / "monitoring.json").write_text(json.dumps(result, indent=2))
    write_summary(report)
    write_output("retrain", str(retrain).lower())
    return retrain


if __name__ == "__main__":
    matplotlib.use("Agg")  # figures enregistrées en fichiers, sans affichage
    sys.stdout.reconfigure(encoding="utf-8")
    monitor()
