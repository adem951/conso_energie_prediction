from __future__ import annotations

import os
from datetime import date, datetime, time, timedelta
from pathlib import Path

import mlflow
import pandas as pd
import streamlit as st

from src.config import EXPERIMENT_NAME, MODEL_NAME, MONITORING_EXPERIMENT
from src.data import fetch_rte
from src.features import FEATURE_COLUMNS, build_features

st.set_page_config(page_title="Prévision consommation", layout="wide")
st.title("Prévision électrique day-ahead")
st.caption("Consommation France à pas de 30 min · modèle servi depuis le registre MLflow (DagsHub).")

ROOT = Path(__file__).resolve().parent
LOCAL_MLFLOW_DB = ROOT / "mlflow.db"
EXAMPLE_PATH = ROOT / "exemples" / "historique_7_jours.csv"


# ---------- Connexion MLflow ----------

def get_setting(name: str, default: str | None = None) -> str | None:
    """Read deployment settings from environment variables or Streamlit secrets."""
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets.get(name, default)
    except (FileNotFoundError, KeyError):
        return default


def connect_mlflow() -> bool:
    """Connect to MLflow (DagsHub, or local mlflow.db)."""
    tracking_uri = get_setting("MLFLOW_TRACKING_URI")
    if not tracking_uri and LOCAL_MLFLOW_DB.exists():
        tracking_uri = f"sqlite:///{LOCAL_MLFLOW_DB.as_posix()}"
    if not tracking_uri:
        return False

    for name in ["MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"]:
        value = get_setting(name)
        if value:
            os.environ[name] = value

    mlflow.set_tracking_uri(tracking_uri)
    return True


@st.cache_resource(ttl=3600)  # une promotion en production atteint l'app en moins d'une heure
def load_production_model():
    if not connect_mlflow():
        return None, None, "MLflow non configuré : vérifiez MLFLOW_TRACKING_URI dans les secrets Streamlit."
    try:
        version = mlflow.MlflowClient().get_model_version_by_alias(MODEL_NAME, "production")
        model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@production")
    except Exception as error:
        return None, None, f"Impossible de charger {MODEL_NAME}@production : {error}"
    return model, version, None


@st.cache_data(ttl=600)
def search_runs(experiment_name: str, filter_string: str = "", max_results: int = 100) -> pd.DataFrame:
    experiment = mlflow.get_experiment_by_name(experiment_name) if connect_mlflow() else None
    if experiment is None:
        return pd.DataFrame()
    return mlflow.search_runs(
        experiment_ids=[experiment.experiment_id], filter_string=filter_string,
        order_by=["attributes.start_time DESC"], max_results=max_results,
    )


@st.cache_data(ttl=600)
def model_versions() -> pd.DataFrame:
    connect_mlflow()
    client = mlflow.MlflowClient()
    aliases = client.get_registered_model(MODEL_NAME).aliases  # {"production": "3", "staging": "4"}
    versions = client.search_model_versions(f"name = '{MODEL_NAME}'")
    return pd.DataFrame([
        {
            "version": int(version.version),
            "alias": ", ".join(f"@{alias}" for alias, number in aliases.items() if str(number) == str(version.version)),
            "modèle": version.tags.get("model_type", ""),
            "RMSE validation": version.tags.get("val_rmse", ""),
            "RMSE test": version.tags.get("test_rmse", ""),
            "décision": version.tags.get("decision", ""),
            "données": version.tags.get("data_version", "")[:8],
            "commit": version.tags.get("git_commit", ""),
            "créée le": pd.to_datetime(version.creation_timestamp, unit="ms"),
        }
        for version in versions
    ]).sort_values("version", ascending=False)


@st.cache_data(ttl=600)
def download_figures(run_id: str, names: list[str]) -> list[str]:
    """Download the figures that exist in the run (a missing file makes MLflow retry for minutes)."""
    existing = {artifact.path for artifact in mlflow.MlflowClient().list_artifacts(run_id)}
    return [mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=name) for name in names if name in existing]


def show_figures(run_id: str, names: list[str]) -> None:
    paths = download_figures(run_id, names)
    if not paths:
        st.info("Pas de graphiques pour ce run (créé avant le pipeline actuel).")
    columns = st.columns(2)
    for index, path in enumerate(paths):
        columns[index % 2].image(path)


# ---------- Prévisions ----------

@st.cache_data(ttl=1800)
def recent_history() -> pd.DataFrame:
    """Last 8 days published by RTE (real-time API)."""
    return fetch_rte(pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=8))


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
    # Même calcul du calendrier qu'à l'entraînement.
    row = build_features(pd.DataFrame({"timestamp": [local], "consommation_mw": [float("nan")]}))
    row[["lag_48", "lag_96", "lag_336"]] = [float(lag_48), float(lag_96), float(lag_336)]
    return float(model.predict(row[FEATURE_COLUMNS])[0])


# ---------- Interface ----------

model, production_version, model_error = load_production_model()
if model_error:
    st.error(model_error)
    st.stop()
st.caption(f"Modèle en service : **{MODEL_NAME} v{production_version.version}** "
           f"({production_version.tags.get('model_type', 'n/a')})")

tab_quick, tab_forecast, tab_performance, tab_monitoring = st.tabs(
    ["Prévision rapide", "Prévoir demain (48 demi-heures)", "Performance modèle", "Monitoring drift"]
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
    source = st.radio(
        "Historique utilisé", ["Données RTE récentes (API)", "Fichier CSV", "Exemple"], horizontal=True
    )
    history = None
    if source == "Données RTE récentes (API)":
        try:
            history = recent_history()
        except Exception as error:
            st.warning(f"API RTE indisponible ({error}) : utilisation de l'exemple.")
            history = pd.read_csv(EXAMPLE_PATH)
    elif source == "Fichier CSV":
        uploaded = st.file_uploader("Historique CSV (colonnes timestamp, consommation_mw)", type="csv")
        if uploaded is not None:
            history = pd.read_csv(uploaded)
    else:
        history = pd.read_csv(EXAMPLE_PATH)

    if history is not None:
        try:
            forecast = forecast_next_day(model, history)
        except Exception as error:
            st.error(f"Prévision impossible : {error}")
        else:
            history_paris = history.assign(
                timestamp=pd.to_datetime(history["timestamp"], utc=True).dt.tz_convert("Europe/Paris")
            )
            chart = pd.concat([
                history_paris.set_index("timestamp")["consommation_mw"].rename("Historique (MW)"),
                forecast.set_index("timestamp")["prediction_mw"].rename("Prévision (MW)"),
            ], axis=1)
            st.line_chart(chart)
            st.dataframe(forecast, width="stretch", hide_index=True)
            st.download_button("Télécharger les prévisions", forecast.to_csv(index=False), "previsions_24h.csv")

with tab_performance:
    st.subheader("Registre de modèles")
    st.dataframe(model_versions(), width="stretch", hide_index=True)

    trainings = search_runs(EXPERIMENT_NAME, "tags.stage = 'training'", max_results=1)
    if not trainings.empty:
        training = trainings.iloc[0]
        st.subheader(f"Dernier entraînement ({training['start_time']:%d/%m/%Y %H:%M})")
        st.caption("Écart train / validation / test : un grand écart signale du sur-apprentissage.")
        candidates = search_runs(EXPERIMENT_NAME, f"tags.mlflow.parentRunId = '{training['run_id']}'")
        rmse = candidates.set_index("tags.model_type")[["metrics.train_rmse", "metrics.val_rmse", "metrics.test_rmse"]]
        rmse.columns = ["train", "validation", "test"]
        st.bar_chart(rmse, stack=False)
        st.dataframe(rmse.round(0), width="stretch")

    st.subheader("Diagnostics d'une version")
    versions = model_versions()
    labels = [f"v{row.version} {row.alias}".strip() for row in versions.itertuples()]
    selected = st.selectbox("Version", labels, index=labels.index(next(label for label in labels if "@production" in label)))
    selected_version = mlflow.MlflowClient().get_model_version(MODEL_NAME, selected.split()[0][1:])
    show_figures(
        selected_version.run_id,
        ["learning_curve.png", "feature_importance.png", "forecast_test.png", "error_by_hour.png"],
    )

with tab_monitoring:
    monitorings = search_runs(MONITORING_EXPERIMENT, "tags.stage = 'monitoring'", max_results=20)
    if monitorings.empty:
        st.warning("Aucun monitoring enregistré : lancez python -m src.monitor.")
    else:
        last = monitorings.iloc[0]
        st.subheader(f"Dernier contrôle : {last['params.window']} (modèle v{last['params.model_version']})")
        col_psi, col_mae, col_ref = st.columns(3)
        col_psi.metric("PSI consommation (vs l'an dernier)", f"{last['metrics.psi']:.3f}")
        col_mae.metric("MAE dernière semaine", f"{last['metrics.mae_last_week']:.0f} MW")
        col_ref.metric("MAE de référence (test)", f"{last['metrics.mae_reference']:.0f} MW")
        for label, drift in [("Données", last["metrics.data_drift"]), ("Modèle", last["metrics.model_drift"])]:
            (st.error if drift else st.success)(f"{label} : {'dérive détectée' if drift else 'stable'}")
        show_figures(last["run_id"], ["drift_distribution.png", "drift_weekly_mae.png"])

        st.subheader("Historique des contrôles")
        checks = monitorings.set_index("start_time")[["metrics.mae_last_week", "metrics.mae_reference"]]
        st.line_chart(checks.rename(columns={"metrics.mae_last_week": "MAE semaine", "metrics.mae_reference": "MAE référence"}))
