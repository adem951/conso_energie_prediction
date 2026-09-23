from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import mlflow
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
HISTORY_PATH = DATA_DIR / "eco2mix.csv"   # données consolidées, versionnées par DVC
RECENT_PATH = DATA_DIR / "recent.csv"     # données temps réel récupérées sur l'API RTE
REPORTS_DIR = ROOT / "reports"

MODEL_NAME = "conso_energie_day_ahead"
# Une expérience MLflow par usage : les graphiques de comparaison restent lisibles.
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "conso_energie_day_ahead")  # entraînements
TUNING_EXPERIMENT = "conso_energie_tuning"
MONITORING_EXPERIMENT = "conso_energie_monitoring"
FIGURE_OPTIONS = {"dpi": 150, "bbox_inches": "tight"}


def load_params() -> dict:
    return yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))


def configure_mlflow(experiment: str = EXPERIMENT_NAME) -> None:
    """Use DagsHub when MLFLOW_TRACKING_URI is set, otherwise a local mlflow.db."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"))
    mlflow.set_experiment(experiment)


def git_commit() -> str:
    if os.getenv("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"][:7]
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def lineage() -> dict[str, str]:
    """What a model was built from: code commit, DVC hash of the history, hash of the recent snapshot."""
    lock = yaml.safe_load((ROOT / "dvc.lock").read_text(encoding="utf-8"))
    history = next(out["md5"] for out in lock["stages"]["prepare_data"]["outs"] if out["path"] == "data/eco2mix.csv")
    recent = hashlib.md5(RECENT_PATH.read_bytes()).hexdigest() if RECENT_PATH.exists() else "none"
    return {"git_commit": git_commit(), "data_version": history, "recent_data_md5": recent}


def write_summary(markdown: str) -> None:
    """Print a Markdown summary, also shown on the GitHub Actions run page."""
    print(markdown)
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as summary:
            summary.write(markdown + "\n")


def write_output(name: str, value: str) -> None:
    """Expose a value to the next GitHub Actions jobs."""
    output_path = os.getenv("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a", encoding="utf-8") as output:
            output.write(f"{name}={value}\n")
