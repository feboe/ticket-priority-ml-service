"""MLflow tracking helpers for training runs."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import mlflow


@contextmanager
def start_run(run_name: str, nested: bool = False) -> Iterator[Any]:
    """Start an MLflow run."""
    mlflow_module = _require_mlflow()
    with mlflow_module.start_run(run_name=run_name, nested=nested) as run:
        yield run


def configure_tracking(tracking_uri: str, experiment_name: str) -> None:
    """Configure MLflow tracking and experiment selection."""
    mlflow_module = _require_mlflow()
    mlflow_module.set_tracking_uri(tracking_uri)
    mlflow_module.set_experiment(experiment_name)


def build_base_run_name(
    *,
    run_group: str,
    dataset_id: str,
    cv_folds: int,
    seed: int,
    git_sha: str,
    run_name: str | None = None,
) -> str:
    """Create a readable base run name for the task runs."""
    if run_name:
        return run_name
    safe_run_group = _slugify(run_group)
    safe_dataset_id = _slugify(dataset_id)
    safe_git_sha = _slugify(git_sha or "unknown")
    return f"{safe_run_group}-{safe_dataset_id}-cv{cv_folds}-seed{seed}-{safe_git_sha}"


def collect_git_metadata(workdir: Path | None = None) -> dict[str, str]:
    """Collect lightweight git metadata for logging."""
    location = workdir or Path.cwd()
    branch = _run_git_command(["git", "branch", "--show-current"], location)
    commit = _run_git_command(["git", "rev-parse", "--short", "HEAD"], location)
    status_output = _run_git_command(["git", "status", "--short"], location)
    return {
        "git_branch": branch or "unknown",
        "git_commit": commit or "unknown",
        "git_dirty": str(bool(status_output)).lower(),
    }


def log_run_metadata(
    *,
    params: Mapping[str, Any] | None = None,
    tags: Mapping[str, Any] | None = None,
    metrics: Mapping[str, float] | None = None,
) -> None:
    """Log a batch of params, tags, and metrics to the active run."""
    mlflow_module = _require_mlflow()

    if params:
        for key, value in params.items():
            if value is None:
                continue
            mlflow_module.log_param(key, _stringify(value))

    if tags:
        normalized_tags = {
            key: _stringify(value) for key, value in tags.items() if value is not None
        }
        if normalized_tags:
            mlflow_module.set_tags(normalized_tags)

    if metrics:
        normalized_metrics = {
            key: float(value) for key, value in metrics.items() if value is not None
        }
        if normalized_metrics:
            mlflow_module.log_metrics(normalized_metrics)


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


def build_runtime_metadata(command_args: list[str]) -> dict[str, str]:
    """Collect process-level runtime metadata."""
    return {
        "command": " ".join(command_args),
        "python_version": sys.version.split()[0],
    }


def _log_artifact(artifact_path: Path, temp_root: Path) -> None:
    mlflow_module = _require_mlflow()
    relative_parent = artifact_path.parent.relative_to(temp_root)
    artifact_subdir = None if str(relative_parent) == "." else str(relative_parent)
    mlflow_module.log_artifact(str(artifact_path), artifact_path=artifact_subdir)


def _require_mlflow() -> Any:
    if mlflow is None:
        raise ImportError(
            "mlflow is required for training tracking. Install the dependencies from requirements.txt."
        ) from _MLFLOW_IMPORT_ERROR
    return mlflow


def _run_git_command(command: list[str], workdir: Path) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=workdir,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _stringify(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (list, tuple, set, dict)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "run"
