import numpy as np
import pandas as pd

from src.data import temporal_split
from src.features import FEATURE_COLUMNS, build_features
from src.models import linear_regression, score
from src.monitor import psi


def make_dataset(days: int = 400) -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01", periods=days * 48, freq="30min", tz="UTC")
    consumption = 50000 + 10000 * np.sin(np.arange(len(timestamps)) * 2 * np.pi / 48)
    data = build_features(pd.DataFrame({"timestamp": timestamps, "consommation_mw": consumption}))
    return data.dropna().reset_index(drop=True)


def test_temporal_split_has_no_overlap():
    fit, validation, test = temporal_split(make_dataset(), {"test_months": 3, "validation_months": 1})
    assert fit["timestamp"].max() < validation["timestamp"].min()
    assert validation["timestamp"].max() < test["timestamp"].min()
    assert len(fit) + len(validation) + len(test) == len(make_dataset())


def test_psi_detects_a_shift_only_when_there_is_one():
    rng = np.random.default_rng(0)
    reference = pd.Series(rng.normal(50000, 5000, 5000))
    assert psi(reference, pd.Series(rng.normal(50000, 5000, 5000))) < 0.1
    assert psi(reference, pd.Series(rng.normal(56000, 5000, 5000))) > 0.2


def test_linear_regression_learns_the_daily_profile():
    data = make_dataset(60)
    model = linear_regression().fit(data[FEATURE_COLUMNS], data["consommation_mw"])
    assert score(data["consommation_mw"], model.predict(data[FEATURE_COLUMNS]))["r2"] > 0.99
