from __future__ import annotations

import json
import sys
from datetime import UTC, datetime

import mlflow

from src.config import MODEL_NAME, REPORTS_DIR, configure_mlflow, write_summary


def record(version: str, decision: str) -> None:
    """Write the decision on the model version and on its run (visible in the Experiments tab)."""
    client = mlflow.MlflowClient()
    client.set_model_version_tag(MODEL_NAME, version, "decision", decision)
    run = client.get_run(client.get_model_version(MODEL_NAME, version).run_id)
    client.set_tag(run.info.run_id, "registry", f"v{version} {decision}")
    parent_id = run.data.tags.get("mlflow.parentRunId")
    if parent_id:  # « entrainement 24/09 10:00 → v4 lightgbm (décision) »
        name = client.get_run(parent_id).data.tags["mlflow.runName"].split(" → ")[0]
        client.set_tag(parent_id, "mlflow.runName", f"{name} → v{version} {run.data.tags.get('model_type', '')} ({decision})")


def promote() -> bool:
    """Champion / challenger: @staging becomes @production only if its test RMSE is lower."""
    configure_mlflow()
    evaluation = json.loads((REPORTS_DIR / "evaluation.json").read_text())
    staging, production = evaluation["staging"], evaluation.get("production")

    if production and staging["version"] == production["version"]:
        write_summary(f"## Promotion\n\nv{staging['version']} est déjà `@production`.\n")
        return False
    if production and staging["rmse"] >= production["rmse"]:
        record(staging["version"], f"refusé : RMSE {staging['rmse']:.0f} ≥ production v{production['version']}")
        write_summary(
            f"## Promotion refusée\n\nv{staging['version']} (RMSE {staging['rmse']:.0f}) ne bat pas "
            f"la production v{production['version']} (RMSE {production['rmse']:.0f}).\n"
        )
        return False

    mlflow.MlflowClient().set_registered_model_alias(MODEL_NAME, "production", staging["version"])
    record(staging["version"], f"@production depuis le {datetime.now(UTC):%d/%m/%Y}")
    if production:
        record(production["version"], f"ancienne production, remplacée par v{staging['version']}")
    previous = f"v{production['version']} (RMSE {production['rmse']:.0f})" if production else "aucune"
    write_summary(
        f"## Promotion acceptée\n\nv{staging['version']} (RMSE {staging['rmse']:.0f}) passe `@production`. "
        f"Production précédente : {previous}.\n"
    )
    return True


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    promote()
