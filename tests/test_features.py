import pandas as pd

from src.features import FEATURE_COLUMNS, build_features


def make_history(periods: int) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=periods, freq="30min", tz="UTC")
    return pd.DataFrame({"timestamp": timestamps, "consommation_mw": range(periods)})


def test_lags_look_only_at_the_past():
    features = build_features(make_history(400))
    last = features.iloc[-1]
    assert last["lag_48"] == last["consommation_mw"] - 48
    assert last["lag_336"] == last["consommation_mw"] - 336


def test_calendar_uses_paris_time():
    features = build_features(make_history(1))
    # 00:00 UTC en janvier = 01:00 à Paris.
    assert features.loc[0, "hour"] == 1


def test_seven_days_are_enough_to_forecast_tomorrow():
    history = make_history(336)
    future = pd.DataFrame({"timestamp": pd.date_range(history["timestamp"].max(), periods=49, freq="30min")[1:]})
    features = build_features(pd.concat([history, future], ignore_index=True)).tail(48)
    assert not features[FEATURE_COLUMNS].isna().any().any()
