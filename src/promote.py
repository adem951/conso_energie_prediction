from __future__ import annotations

import json
import sys
from datetime import UTC, datetime

import mlflow

from src.config import MODEL_NAME, REPORTS_DIR, configure_mlflow, write_summary


def promote() -> bool:
    """Champion / challenger: @staging becomes @production only if its test RMSE is lower."""
    configure_mlflow()
    evaluation = json.loads((REPORTS_DIR / "evaluation.json").read_text())
    staging, production = evaluation["staging"], evaluation.get("production")

    if production and staging["version"] == production["version"]:
        write_summary(f"## Promotion\n\nv{staging['version']} est déjà `@production`.\n")
        return False
    if production and staging["rmse"] >= production["rmse"]:
        write_summary(
            f"## Promotion refusée\n\nv{staging['version']} (RMSE {staging['rmse']:.0f}) ne bat pas "
            f"la production v{production['version']} (RMSE {production['rmse']:.0f}).\n"
        )
        return False

    client = mlflow.MlflowClient()
    client.set_registered_model_alias(MODEL_NAME, "production", staging["version"])
    client.set_model_version_tag(MODEL_NAME, staging["version"], "promoted_at", datetime.now(UTC).isoformat())
    previous = f"v{production['version']} (RMSE {production['rmse']:.0f})" if production else "aucune"
    write_summary(
        f"## Promotion acceptée\n\nv{staging['version']} (RMSE {staging['rmse']:.0f}) passe `@production`. "
        f"Production précédente : {previous}.\n"
    )
    return True


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    promote()
