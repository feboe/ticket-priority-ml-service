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


def _slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "run"
