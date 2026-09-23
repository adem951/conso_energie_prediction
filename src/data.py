from __future__ import annotations

import sys
from io import StringIO

import pandas as pd
import requests

from src.config import HISTORY_PATH, RECENT_PATH
from src.features import FEATURE_COLUMNS, build_features

RTE_API_URL = "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/eco2mix-regional-tr/exports/csv"
REGION_COUNT = 12


def fetch_rte(start: pd.Timestamp, attempts: int = 3) -> pd.DataFrame:
    """National consumption since `start`, retried because the API can return a truncated export."""
    for _ in range(attempts):
        data = fetch_rte_once(start)
        # Une réponse complète va jusqu'aux dernières heures publiées.
        if not data.empty and data["timestamp"].max() > pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1):
            return data
    raise RuntimeError("Réponse de l'API RTE incomplète : données récentes manquantes.")


def fetch_rte_once(start: pd.Timestamp) -> pd.DataFrame:
    """National consumption since `start` = sum of the 12 regions from the RTE real-time API."""
    response = requests.get(
        RTE_API_URL,
        params={
            "select": "date_heure, sum(consommation) as consommation_mw, count(consommation) as regions",
            "group_by": "date_heure",
            "where": f"date_heure >= '{start.isoformat()}'",
            "order_by": "date_heure",
        },
        timeout=60,
    )
    response.raise_for_status()
    data = pd.read_csv(StringIO(response.content.decode("utf-8-sig")), sep=";")
    data["timestamp"] = pd.to_datetime(data["date_heure"], utc=True)

    # Même format que l'historique : pas de 30 min et les 12 régions présentes.
    complete = (data["regions"] == REGION_COUNT) & data["timestamp"].dt.minute.isin([0, 30])
    return data.loc[complete, ["timestamp", "consommation_mw"]].reset_index(drop=True)


def fetch_recent() -> None:
    """Download everything published after the consolidated history."""
    history_end = pd.read_csv(HISTORY_PATH, usecols=["timestamp"])["timestamp"].max()
    recent = fetch_rte(pd.Timestamp(history_end) + pd.Timedelta(minutes=30))
    recent.to_csv(RECENT_PATH, index=False)
    print(f"Données récentes : {RECENT_PATH} ({len(recent):,} lignes jusqu'au {recent['timestamp'].max()})")


def load_series() -> pd.DataFrame:
    """Consolidated history followed by the recent real-time data (if downloaded)."""
    parts = [pd.read_csv(HISTORY_PATH)]
    if RECENT_PATH.exists():
        parts.append(pd.read_csv(RECENT_PATH))
    series = pd.concat(parts, ignore_index=True)
    series["timestamp"] = pd.to_datetime(series["timestamp"], utc=True)
    return series


def load_dataset() -> pd.DataFrame:
    data = build_features(load_series())
    return data.dropna(subset=FEATURE_COLUMNS + ["consommation_mw"]).reset_index(drop=True)


def temporal_split(data: pd.DataFrame, split: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """fit < validation < test, in chronological order, without shuffling."""
    test_start = data["timestamp"].max() - pd.DateOffset(months=split["test_months"])
    validation_start = test_start - pd.DateOffset(months=split["validation_months"])
    fit = data[data["timestamp"] < validation_start]
    validation = data[(data["timestamp"] >= validation_start) & (data["timestamp"] < test_start)]
    test = data[data["timestamp"] >= test_start]
    return fit, validation, test


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    fetch_recent()
