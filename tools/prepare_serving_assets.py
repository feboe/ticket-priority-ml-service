"""Prepare fixed serving assets from selected MLflow runs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib

EXPERIMENT_ID = "889656232015600132"
SELECTED_RUNS = {
    "queue": "08767b25d399410bb4d9d819c0d8fceb",
    "priority": "d7faa4f8e38c4345b8abc295be5db18a",
}

ROOT = Path(__file__).resolve().parents[1]
MLRUNS_ROOT = ROOT / "mlruns" / EXPERIMENT_ID
SERVING_ROOT = ROOT / "serving_assets"
MODELS_ROOT = SERVING_ROOT / "models"
CONFIGS_ROOT = SERVING_ROOT / "configs"


def _normalize_run_config(
    run_config: dict[str, object], trainer: object
) -> dict[str, object]:
    normalized = dict(run_config)
    preprocessing = dict(normalized.get("preprocessing", {}))
    preprocessing.setdefault(
        "analyzer",
        getattr(trainer.preprocessor.pipeline.feature_extractor, "analyzer", "word"),
    )
    normalized["preprocessing"] = preprocessing

    if "feature_matrix" not in normalized:
        feature_families = ["tfidf"]
        if preprocessing.get("length_feature_enabled"):
            feature_families.append("length")
        normalized["feature_matrix"] = {
            "rows": normalized["dataset"]["row_count"],
            "columns": len(trainer.feature_names_),
            "feature_families": feature_families,
        }

    return normalized


def main() -> None:
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIGS_ROOT.mkdir(parents=True, exist_ok=True)

    serving_config = {
        "app": {
            "title": "Ticket Triage Demo",
        },
        "models": {},
    }

    for task_name, run_id in SELECTED_RUNS.items():
        artifact_root = MLRUNS_ROOT / run_id / "artifacts"
        model_source = artifact_root / "trained_model.joblib"
        run_config_source = artifact_root / "run_config.json"

        trainer = joblib.load(model_source)
        run_config = json.loads(run_config_source.read_text(encoding="utf-8"))
        normalized_run_config = _normalize_run_config(run_config, trainer)

        model_destination = MODELS_ROOT / f"{task_name}_model.joblib"
        run_config_destination = CONFIGS_ROOT / f"{task_name}_run_config.json"

        shutil.copy2(model_source, model_destination)
        run_config_destination.write_text(
            json.dumps(normalized_run_config, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        serving_config["models"][task_name] = {
            "run_id": run_id,
            "model_path": str(model_destination.relative_to(SERVING_ROOT)).replace(
                "\\", "/"
            ),
            "run_config_path": str(
                run_config_destination.relative_to(SERVING_ROOT)
            ).replace("\\", "/"),
        }

    (SERVING_ROOT / "serving_config.json").write_text(
        json.dumps(serving_config, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
