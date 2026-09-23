from __future__ import annotations

import pandas as pd


FEATURE_COLUMNS = [
    "hour", "minute", "dayofweek", "month", "dayofyear",
    "lag_1", "lag_48", "lag_96", "lag_336",
]


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create calendar features and 30-minute historical lags."""
    frame = data.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp")
    frame = frame.set_index("timestamp").asfreq("30min").rename_axis("timestamp").reset_index()

    frame["hour"] = frame["timestamp"].dt.hour
    frame["minute"] = frame["timestamp"].dt.minute
    frame["dayofweek"] = frame["timestamp"].dt.dayofweek
    frame["month"] = frame["timestamp"].dt.month
    frame["dayofyear"] = frame["timestamp"].dt.dayofyear

    for lag in [1, 48, 96, 336]:
        frame[f"lag_{lag}"] = frame["consommation_mw"].shift(lag)

    return frame
