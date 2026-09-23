from __future__ import annotations

import pandas as pd


# Tous les lags font au moins 24 h : les 48 demi-heures de demain
# se prédisent d'un coup, uniquement à partir de valeurs déjà connues.
LAGS = [48, 96, 336]
FEATURE_COLUMNS = [
    "hour", "minute", "dayofweek", "month", "dayofyear",
    *[f"lag_{lag}" for lag in LAGS],
]


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create calendar features (Paris time) and 30-minute historical lags."""
    frame = data.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp")
    frame = frame.set_index("timestamp").asfreq("30min").rename_axis("timestamp").reset_index()

    # La consommation suit l'heure locale (changements d'heure inclus).
    local_time = frame["timestamp"].dt.tz_convert("Europe/Paris")
    frame["hour"] = local_time.dt.hour
    frame["minute"] = local_time.dt.minute
    frame["dayofweek"] = local_time.dt.dayofweek
    frame["month"] = local_time.dt.month
    frame["dayofyear"] = local_time.dt.dayofyear

    for lag in LAGS:
        frame[f"lag_{lag}"] = frame["consommation_mw"].shift(lag)

    return frame
