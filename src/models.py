from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import LAGS

# Références naïves : pas d'apprentissage, le modèle doit faire mieux.
NAIVE_MODELS = {
    "naive_24h": "lag_48",   # demain = aujourd'hui
    "naive_7d": "lag_336",   # demain = même jour la semaine dernière
}


def linear_regression() -> Pipeline:
    """Linear model: calendar one-hot encoded (hour 7h is not 'half' of 14h), scaled lags."""
    preprocessing = ColumnTransformer([
        ("calendar", OneHotEncoder(handle_unknown="ignore"), ["hour", "minute", "dayofweek", "month"]),
        ("lags", StandardScaler(), [f"lag_{lag}" for lag in LAGS]),
    ])
    return Pipeline([("preprocessing", preprocessing), ("model", LinearRegression())])


def lightgbm(params: dict) -> LGBMRegressor:
    model_params = {key: value for key, value in params.items() if key != "early_stopping_rounds"}
    return LGBMRegressor(**model_params, random_state=42, verbosity=-1)


def score(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
    }
