"""MLflow tracking helpers for training runs."""

from __future__ import annotations

import json
import re
import tempfile
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import joblib
import mlflow
import pandas as pd


@contextmanager
def start_run(run_name: str, nested: bool = False) -> Iterator[Any]:
    """Start an MLflow run."""
    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        yield run


def configure_tracking(tracking_uri: str, experiment_name: str) -> None:
    """Configure MLflow tracking and experiment selection."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def build_base_run_name(
    *,
    run_group: str,
    dataset_id: str,
    cv_folds: int,
    seed: int,
    run_name: str | None = None,
) -> str:
    """Create a readable base run name for the task runs."""
    if run_name:
        return run_name
    safe_run_group = _slugify(run_group)
    safe_dataset_id = _slugify(dataset_id)
    return f"{safe_run_group}-{safe_dataset_id}-cv{cv_folds}-seed{seed}"


def build_dataset_metadata(df: pd.DataFrame, data_path: Path) -> dict[str, Any]:
    return {
        "dataset_id": data_path.stem,
        "dataset_path": str(data_path.resolve()),
        "dataset_row_count": int(len(df)),
    }


def build_shared_tracking_payload(
    *,
    run_group: str,
    algorithm: str,
    cv_folds: int,
    random_state: int,
    stratify_columns: list[str],
    dataset_metadata: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    params = {
        "run_group": run_group,
        "dataset_id": dataset_metadata["dataset_id"],
        "dataset_row_count": dataset_metadata["dataset_row_count"],
        "cv_folds": cv_folds,
        "random_state": random_state,
        "stratify_columns": stratify_columns,
        "algorithm": algorithm,
    }
    tags = {
        "run_group": run_group,
    }
    return params, tags


def build_task_tracking_payload(
    *,
    task_name: str,
    task_results: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    task_config = task_results["task_config"]
    per_class_metrics = task_results["per_class_metrics"]

    params = {
        "task_name": task_name,
        "target_column": task_config["target_column"],
        **task_config["model"],
        **task_config["preprocessing"],
        "num_classes": int(len(per_class_metrics)),
    }
    tags = {
        "task_name": task_name,
    }
    return params, tags


def build_run_config(
    *,
    run_group: str,
    cv_folds: int,
    random_state: int,
    task_name: str,
    task_results: dict[str, Any],
    dataset_metadata: dict[str, Any],
    final_trainer: Any,
    run_name: str,
    stratify_columns: list[str],
) -> dict[str, Any]:
    task_config = task_results["task_config"]
    per_class_metrics = task_results["per_class_metrics"]
    feature_summary = final_trainer.get_feature_summary()

    return {
        "run": {
            "group": run_group,
            "name": run_name,
        },
        "task": {
            "name": task_name,
            "target_column": task_config["target_column"],
            "labels": per_class_metrics["label"].tolist(),
        },
        "dataset": {
            "id": dataset_metadata["dataset_id"],
            "path": dataset_metadata["dataset_path"],
            "row_count": dataset_metadata["dataset_row_count"],
        },
        "training": {
            "cv_folds": cv_folds,
            "random_state": random_state,
            "stratify_columns": stratify_columns,
        },
        "model": task_config["model"],
        "preprocessing": task_config["preprocessing"],
        "feature_matrix": {
            "rows": dataset_metadata["dataset_row_count"],
            "columns": feature_summary["feature_count"],
            "feature_families": feature_summary["feature_families"],
        },
        "artifacts": {
            "trained_model": "trained_model.joblib",
        },
    }


def log_run_metadata(
    *,
    params: Mapping[str, Any] | None = None,
    tags: Mapping[str, Any] | None = None,
    metrics: Mapping[str, float] | None = None,
) -> None:
    """Log a batch of params, tags, and metrics to the active run."""
    if params:
        for key, value in params.items():
            if value is None:
                continue
            mlflow.log_param(key, _stringify(value))

    if tags:
        normalized_tags = {
            key: _stringify(value) for key, value in tags.items() if value is not None
        }
        if normalized_tags:
            mlflow.set_tags(normalized_tags)

    if metrics:
        normalized_metrics = {
            key: float(value) for key, value in metrics.items() if value is not None
        }
        if normalized_metrics:
            mlflow.log_metrics(normalized_metrics)


def log_dataframe_artifact(frame: pd.DataFrame, artifact_file: str) -> None:
    """Persist a dataframe artifact to the active run."""
    with tempfile.TemporaryDirectory() as temp_dir:
        artifact_path = Path(temp_dir) / artifact_file
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(artifact_path, index=False)
        _log_artifact(artifact_path, Path(temp_dir))


def log_json_artifact(payload: Mapping[str, Any], artifact_file: str) -> None:
    """Persist a JSON artifact to the active run."""
    with tempfile.TemporaryDirectory() as temp_dir:
        artifact_path = Path(temp_dir) / artifact_file
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        _log_artifact(artifact_path, Path(temp_dir))


def log_model_artifact(
    model_object: Any, artifact_file: str = "trained_model.joblib"
) -> None:
    """Persist a fitted model object to the active run."""
    with tempfile.TemporaryDirectory() as temp_dir:
        artifact_path = Path(temp_dir) / artifact_file
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_object, artifact_path)
        _log_artifact(artifact_path, Path(temp_dir))


def log_holdout_evaluation_runs(
    *,
    evaluations: Mapping[str, Any],
    dataset_metadata: Mapping[str, Any],
    model_metadata: Mapping[str, Any],
    run_group: str,
) -> dict[str, str]:
    """Log one traceable MLflow holdout-evaluation run per task."""
    run_ids: dict[str, str] = {}

    for task_name, result in evaluations.items():
        task_model_metadata = dict(model_metadata.get(task_name, {}))
        source_model_run_id = task_model_metadata.get("run_id")
        if not source_model_run_id:
            raise ValueError(
                f"Missing source model run ID for holdout task '{task_name}'."
            )

        metrics = {
            "holdout_accuracy": float(result.fold_metrics["accuracy"]),
            "holdout_macro_f1": float(result.fold_metrics["macro_f1"]),
            **_flatten_holdout_language_metrics(result.language_metrics),
        }
        params = {
            "run_group": run_group,
            "evaluation_type": "frozen_holdout",
            "task_name": task_name,
            "dataset_file": dataset_metadata["file"],
            "dataset_sha256": dataset_metadata["sha256"],
            "dataset_source_row_count": dataset_metadata["source_row_count"],
            "dataset_evaluated_row_count": dataset_metadata["evaluated_row_count"],
            "languages": dataset_metadata["languages"],
            "source_model_run_id": source_model_run_id,
            "source_training_dataset_id": task_model_metadata.get("dataset_id"),
        }
        tags = {
            "run_type": "holdout_evaluation",
            "task_name": task_name,
            "source_model_run_id": source_model_run_id,
            "dataset_sha256": dataset_metadata["sha256"],
        }
        artifact_names = {
            "language_metrics": "language_metrics.csv",
            "per_class_metrics": "per_class_metrics.csv",
            "confusion_matrix": "confusion_matrix.csv",
            "per_class_confusion": "per_class_confusion.csv",
            "run_config": "holdout_run_config.json",
        }
        run_config = {
            "evaluation": {
                "type": "frozen_holdout",
                "run_group": run_group,
                "task_name": task_name,
            },
            "dataset": dict(dataset_metadata),
            "source_model": task_model_metadata,
            "metrics": metrics,
            "artifacts": artifact_names,
        }
        run_name = (
            f"{_slugify(run_group)}::{task_name}::"
            f"{str(dataset_metadata['sha256'])[:12]}"
        )

        with start_run(run_name) as run:
            log_run_metadata(params=params, tags=tags, metrics=metrics)
            log_dataframe_artifact(
                result.language_metrics, artifact_names["language_metrics"]
            )
            log_dataframe_artifact(
                result.per_class_metrics, artifact_names["per_class_metrics"]
            )
            log_dataframe_artifact(
                result.confusion_matrix, artifact_names["confusion_matrix"]
            )
            log_dataframe_artifact(
                result.per_class_confusion,
                artifact_names["per_class_confusion"],
            )
            log_json_artifact(run_config, artifact_names["run_config"])
            run_ids[task_name] = run.info.run_id

    return run_ids


def _log_artifact(artifact_path: Path, temp_root: Path) -> None:
    relative_parent = artifact_path.parent.relative_to(temp_root)
    artifact_subdir = None if str(relative_parent) == "." else str(relative_parent)
    mlflow.log_artifact(str(artifact_path), artifact_path=artifact_subdir)


def _stringify(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (list, tuple, set, dict)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _flatten_holdout_language_metrics(
    language_metrics: pd.DataFrame,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for row in language_metrics.itertuples(index=False):
        language_slug = _slugify(str(row.language))
        metrics[f"holdout_accuracy__lang_{language_slug}"] = float(row.accuracy)
        metrics[f"holdout_macro_f1__lang_{language_slug}"] = float(row.macro_f1)
        metrics[f"holdout_sample_count__lang_{language_slug}"] = float(
            row.sample_count
        )
    return metrics


def _slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "run"
