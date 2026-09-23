from __future__ import annotations

import os
from datetime import date, datetime, time, timedelta
from pathlib import Path

import mlflow
import pandas as pd
import streamlit as st

from src.features import FEATURE_COLUMNS, build_features


st.set_page_config(page_title="Prévision consommation", layout="wide")
st.title("Prévision électrique day-ahead")
st.caption("48 demi-heures prédites à partir des 7 derniers jours de consommation.")

ROOT = Path(__file__).resolve().parent
LOCAL_MLFLOW_DB = ROOT / "mlflow.db"
EXAMPLE_PATH = ROOT / "exemples" / "historique_7_jours.csv"


def get_setting(name: str, default: str | None = None) -> str | None:
    """Read deployment settings from environment variables or Streamlit secrets."""
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets.get(name, default)
    except (FileNotFoundError, KeyError):
        return default


def connect_mlflow() -> str | None:
    """Connect to MLflow (DagsHub, or local mlflow.db) and return the experiment id."""
    tracking_uri = get_setting("MLFLOW_TRACKING_URI")
    if not tracking_uri and LOCAL_MLFLOW_DB.exists():
        tracking_uri = f"sqlite:///{LOCAL_MLFLOW_DB.as_posix()}"
    if not tracking_uri:
        return None

    for name in ["MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"]:
        value = get_setting(name)
        if value:
            os.environ[name] = value

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(get_setting("MLFLOW_EXPERIMENT_NAME", "conso_energie_day_ahead"))
    return experiment.experiment_id if experiment else None


@st.cache_resource
def load_production_model():
    experiment_id = connect_mlflow()
    if experiment_id is None:
        return None, "MLflow non configuré : vérifiez MLFLOW_TRACKING_URI et MLFLOW_EXPERIMENT_NAME dans les secrets Streamlit."

    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string="tags.stage = 'production'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        return None, "Aucun run production trouvé dans MLflow. Lancez python -m src.train."

    try:
        model = mlflow.pyfunc.load_model(f"runs:/{runs.loc[0, 'run_id']}/model")
    except Exception as error:
        return None, f"Impossible de charger le modèle : {error}"
    return model, None


@st.cache_data(ttl=600)
def load_mlflow_runs() -> pd.DataFrame:
    experiment_id = connect_mlflow()
    if experiment_id is None:
        return pd.DataFrame()
    return mlflow.search_runs(experiment_ids=[experiment_id], order_by=["attributes.start_time DESC"])


def forecast_next_day(model, history: pd.DataFrame) -> pd.DataFrame:
    """Predict the 48 half-hours following the last timestamp of the history."""
    history = history[["timestamp", "consommation_mw"]].copy()
    history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True)
    future_index = pd.date_range(history["timestamp"].max(), periods=49, freq="30min")[1:]

    frame = pd.concat([history, pd.DataFrame({"timestamp": future_index})], ignore_index=True)
    future = build_features(frame).tail(48)
    if future[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("l'historique doit couvrir les 7 derniers jours sans trou (336 demi-heures).")

    return pd.DataFrame({
        "timestamp": future["timestamp"].dt.tz_convert("Europe/Paris"),
        "prediction_mw": model.predict(future[FEATURE_COLUMNS]),
    })


def forecast_one_half_hour(model, moment: datetime, lag_48: float, lag_96: float, lag_336: float) -> float:
    """Predict a single half-hour (Paris time) from three past values."""
    local = pd.Timestamp(moment).tz_localize("Europe/Paris", nonexistent="shift_forward", ambiguous=False)
    row = pd.DataFrame([{
        "hour": local.hour,
        "minute": local.minute,
        "dayofweek": local.dayofweek,
        "month": local.month,
        "dayofyear": local.dayofyear,
        "lag_48": lag_48,
        "lag_96": lag_96,
        "lag_336": lag_336,
    }])
    return float(model.predict(row[FEATURE_COLUMNS])[0])


model, model_error = load_production_model()
if model_error:
    st.error(model_error)
    st.stop()

tab_quick, tab_forecast, tab_performance = st.tabs(
    ["Prévision rapide", "Prévoir demain (48 demi-heures)", "Performance modèle"]
)

with tab_quick:
    st.write("Choisissez une demi-heure et réglez la consommation observée au même moment les jours précédents.")
    col_date, col_time = st.columns(2)
    selected_date = col_date.date_input("Date (heure de Paris)", value=date.today() + timedelta(days=1))
    selected_time = col_time.time_input("Heure", value=time(19, 0), step=1800)

    lag_48 = st.slider("Consommation 24 h avant (MW)", 25000, 100000, 55000, step=500)
    lag_96 = st.slider("Consommation 48 h avant (MW)", 25000, 100000, 55000, step=500)
    lag_336 = st.slider("Consommation 7 jours avant (MW)", 25000, 100000, 55000, step=500)

    prediction = forecast_one_half_hour(
        model, datetime.combine(selected_date, selected_time), lag_48, lag_96, lag_336
    )
    st.metric(f"Prévision le {selected_date:%d/%m/%Y} à {selected_time:%H:%M}", f"{prediction:,.0f} MW".replace(",", " "))

with tab_forecast:
    uploaded = st.file_uploader("Historique CSV (colonnes timestamp, consommation_mw)", type="csv")
    if uploaded is None:
        st.info("Aucun fichier chargé : utilisation de l'historique d'exemple.")
        history = pd.read_csv(EXAMPLE_PATH)
    else:
        history = pd.read_csv(uploaded)

    try:
        forecast = forecast_next_day(model, history)
    except Exception as error:
        st.error(f"Prévision impossible : {error}")
    else:
        chart = pd.concat([
            history.assign(timestamp=pd.to_datetime(history["timestamp"], utc=True).dt.tz_convert("Europe/Paris"))
                   .set_index("timestamp")["consommation_mw"].rename("Historique (MW)"),
            forecast.set_index("timestamp")["prediction_mw"].rename("Prévision (MW)"),
        ], axis=1)
        st.line_chart(chart)
        st.dataframe(forecast, width="stretch", hide_index=True)
        st.download_button("Télécharger les prévisions", forecast.to_csv(index=False), "previsions_24h.csv")

with tab_performance:
    runs = load_mlflow_runs()
    if runs.empty:
        st.warning("Aucune métrique MLflow disponible.")
    else:
        metrics = runs.reindex(columns=["tags.stage", "metrics.rmse", "metrics.mae", "metrics.r2", "start_time"])
        metrics.columns = ["stage", "RMSE", "MAE", "R2", "date"]
        st.dataframe(metrics, width="stretch", hide_index=True)
        st.bar_chart(metrics.dropna(subset=["RMSE"]).drop_duplicates("stage").set_index("stage")[["RMSE", "MAE"]])
