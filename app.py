from __future__ import annotations

import os
from datetime import date, datetime, time, timedelta
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import streamlit as st

from src.features import FEATURE_COLUMNS, build_features


st.set_page_config(page_title="Prévision consommation", layout="wide")
st.title("Prévision électrique day-ahead")
st.caption("48 demi-heures prédites à partir de l'historique disponible.")

LOCAL_MLFLOW_DB = Path(__file__).resolve().parent / "mlflow.db"


def get_setting(name: str, default: str | None = None) -> str | None:
    """Read deployment settings from environment variables or Streamlit secrets."""
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets.get(name, default)
    except (FileNotFoundError, KeyError):
        return default


@st.cache_resource
def load_production_model():
    tracking_uri = get_setting("MLFLOW_TRACKING_URI")
    experiment_name = get_setting("MLFLOW_EXPERIMENT_NAME", "conso_energie_day_ahead")
    username = get_setting("MLFLOW_TRACKING_USERNAME")
    password = get_setting("MLFLOW_TRACKING_PASSWORD")
    if not tracking_uri and LOCAL_MLFLOW_DB.exists():
        tracking_uri = f"sqlite:///{LOCAL_MLFLOW_DB.as_posix()}"
    if not tracking_uri:
        return None, "Configurez MLFLOW_TRACKING_URI dans les secrets Streamlit."

    if username:
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
    if password:
        os.environ["MLFLOW_TRACKING_PASSWORD"] = password

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None, f"Expérience MLflow introuvable : {experiment_name}"

    client = mlflow.MlflowClient()
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.stage = 'production'",
        order_by=["attributes.start_time DESC"],
        max_results=20,
    )
    if runs.empty:
        return None, "Aucun run production trouvé dans MLflow."

    run_id = None
    for run in runs:
        artifacts = client.list_artifacts(run.info.run_id)
        if any(artifact.path == "model" for artifact in artifacts):
            run_id = run.info.run_id
            break
    if run_id is None:
        return None, "Les runs production existent, mais aucun ne contient l'artefact model. Relancez src.train avec MLflow DagsHub configuré."

    try:
        model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
    except Exception as error:
        return None, f"Impossible de charger le modèle distant : {error}"
    return model, None


def forecast_next_day(model, last_timestamp: pd.Timestamp, seed_values: dict[str, float]) -> pd.DataFrame:
    """Predict 48 half-hours from the four lag values supplied by the user."""
    extended_values = {
        1: seed_values["lag_1"],
        48: seed_values["lag_48"],
        96: seed_values["lag_96"],
        336: seed_values["lag_336"],
    }
    future_index = pd.date_range(start=last_timestamp + pd.Timedelta(minutes=30), periods=48, freq="30min")
    predictions = []

    for timestamp in future_index:
        row = pd.DataFrame({"timestamp": [timestamp]})
        row["hour"] = timestamp.hour
        row["minute"] = timestamp.minute
        row["dayofweek"] = timestamp.dayofweek
        row["month"] = timestamp.month
        row["dayofyear"] = timestamp.dayofyear
        for lag in [1, 48, 96, 336]:
            # Les valeurs futures deviennent disponibles au fur et à mesure.
            # Avant cela, on utilise les valeurs de référence saisies par l'utilisateur.
            row[f"lag_{lag}"] = predictions[-lag] if len(predictions) >= lag else extended_values[lag]

        prediction = float(model.predict(row[FEATURE_COLUMNS])[0])
        predictions.append(prediction)

    return pd.DataFrame({"timestamp": future_index, "prediction_mw": predictions})


def calculate_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Population Stability Index for a simple operational drift signal."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    reference_distribution, _ = np.histogram(reference, bins=edges)
    current_distribution, _ = np.histogram(current, bins=edges)
    reference_distribution = np.clip(reference_distribution / len(reference), 1e-6, None)
    current_distribution = np.clip(current_distribution / len(current), 1e-6, None)
    return float(np.sum((current_distribution - reference_distribution) * np.log(current_distribution / reference_distribution)))


def load_mlflow_runs() -> pd.DataFrame:
    tracking_uri = get_setting("MLFLOW_TRACKING_URI")
    experiment_name = get_setting("MLFLOW_EXPERIMENT_NAME", "conso_energie_day_ahead")
    if not tracking_uri and LOCAL_MLFLOW_DB.exists():
        tracking_uri = f"sqlite:///{LOCAL_MLFLOW_DB.as_posix()}"
    if not tracking_uri:
        return pd.DataFrame()
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return pd.DataFrame()
    return mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"])


model, model_error = load_production_model()
if model_error:
    st.error(model_error)
    st.stop()

tab_forecast, tab_performance, tab_monitoring = st.tabs(
    ["Prévoir demain", "Performance modèle", "Monitoring drift"]
)

with tab_forecast:
    st.write("Sélectionnez le dernier timestamp connu et les valeurs historiques utilisées par le modèle.")
    selected_date = st.date_input("Date de la dernière observation", value=date.today())
    selected_time = st.time_input("Heure de la dernière observation", value=time(0, 0), step=1800)
    last_timestamp = pd.Timestamp(datetime.combine(selected_date, selected_time), tz="UTC")

    col1, col2 = st.columns(2)
    with col1:
        lag_1 = st.number_input("Consommation il y a 30 min (MW)", min_value=0.0, value=55000.0, step=100.0)
        lag_48 = st.number_input("Consommation il y a 24 h (MW)", min_value=0.0, value=55000.0, step=100.0)
    with col2:
        lag_96 = st.number_input("Consommation il y a 48 h (MW)", min_value=0.0, value=55000.0, step=100.0)
        lag_336 = st.number_input("Consommation il y a 7 jours (MW)", min_value=0.0, value=55000.0, step=100.0)

    st.caption("Les variables calendrier sont calculées automatiquement à partir de la date et de l'heure choisies.")
    if st.button("Prévoir la consommation de demain", type="primary"):
        try:
            forecast = forecast_next_day(
                model,
                last_timestamp,
                {"lag_1": lag_1, "lag_48": lag_48, "lag_96": lag_96, "lag_336": lag_336},
            )
            st.session_state["forecast"] = forecast
        except Exception as error:
            st.error(f"Prévision impossible : {error}")

    forecast = st.session_state.get("forecast")
    if forecast is not None:
        chart_forecast = forecast.rename(columns={"prediction_mw": "Prévision (MW)"})
        st.line_chart(chart_forecast.set_index("timestamp"))
        st.dataframe(forecast, use_container_width=True, hide_index=True)
        st.download_button("Télécharger les prévisions", forecast.to_csv(index=False), "previsions_24h.csv")

with tab_performance:
    runs = load_mlflow_runs()
    if runs.empty:
        st.warning("Aucune métrique MLflow disponible.")
    else:
        metrics = runs[["tags.stage", "metrics.rmse", "metrics.mae", "metrics.r2", "start_time"]].copy()
        metrics.columns = ["stage", "RMSE", "MAE", "R2", "date"]
        st.dataframe(metrics, use_container_width=True, hide_index=True)
        st.bar_chart(metrics.set_index("stage")[["RMSE", "MAE"]])

with tab_monitoring:
    st.subheader("Monitoring des variables envoyées au modèle")
    seed = pd.Series({"lag_1": lag_1, "lag_48": lag_48, "lag_96": lag_96, "lag_336": lag_336})
    st.dataframe(seed.rename("valeur MW").to_frame(), use_container_width=True)
    st.info("Sans historique distant ou mesures réelles récentes, un drift statistique complet ne peut pas être calculé. Les valeurs saisies sont affichées pour contrôle opérationnel.")
    st.caption("Pour activer Evidently/PSI en production, branchez une source de mesures récente et comparez-la à une référence versionnée dans DVC.")
