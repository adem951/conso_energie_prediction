from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "data" / "eco2mix-regional-cons-def.csv"
NATIONAL_PATH = ROOT / "data" / "eco2mix.csv"
REGIONAL_PATH = ROOT / "data" / "eco2mix_regional.csv"


def prepare_data() -> None:
    raw = pd.read_csv(
        SOURCE_PATH,
        sep=";",
        na_values=["ND", "nd", "", " "],
        low_memory=False,
        encoding="utf-8",
    )
    raw["Date - Heure"] = pd.to_datetime(raw["Date - Heure"], errors="coerce", utc=True)
    raw["Consommation (MW)"] = pd.to_numeric(raw["Consommation (MW)"], errors="coerce")
    raw = raw.dropna(subset=["Date - Heure"])

    regional = raw[["Région", "Date - Heure", "Consommation (MW)"]].dropna(
        subset=["Région", "Date - Heure", "Consommation (MW)"]
    )
    regional = regional.rename(
        columns={"Date - Heure": "timestamp", "Consommation (MW)": "consommation_mw"}
    )
    regional = regional.drop_duplicates(["Région", "timestamp"])
    regional = regional.sort_values(["Région", "timestamp"])
    regional.to_csv(REGIONAL_PATH, index=False)

    national = (
        regional.groupby("timestamp", as_index=False)["consommation_mw"]
        .sum()
        .sort_values("timestamp")
    )
    national.to_csv(NATIONAL_PATH, index=False)
    print(f"National: {NATIONAL_PATH} ({len(national):,} lignes)")
    print(f"Regional: {REGIONAL_PATH} ({len(regional):,} lignes)")


if __name__ == "__main__":
    prepare_data()
