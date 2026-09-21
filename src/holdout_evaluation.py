"""Evaluation helpers for the frozen external holdout dataset."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from .evaluation import FoldEvaluation, evaluate_fitted_trainer

HOLDOUT_SHA256 = "9aae7120cf459fc27561febe29c7757c6d222bfebff50e8baa868991e57b87d1"
HOLDOUT_LANGUAGES = ("en", "de")
REQUIRED_HOLDOUT_COLUMNS = {"subject", "body", "language", "queue", "priority"}
REQUIRED_TASKS = ("queue", "priority")


def compute_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: Path, expected_sha256: str) -> str:
    """Verify a file against its frozen digest and return the actual digest."""
    actual_sha256 = compute_sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"SHA-256 mismatch for '{path}': expected {expected_sha256}, "
            f"found {actual_sha256}."
        )
    return actual_sha256


def prepare_holdout_frame(
    frame: pd.DataFrame,
    languages: Sequence[str] = HOLDOUT_LANGUAGES,
) -> pd.DataFrame:
    """Validate the holdout schema and return its supported-language view."""
    missing_columns = sorted(REQUIRED_HOLDOUT_COLUMNS.difference(frame.columns))
    if missing_columns:
        raise KeyError("Missing holdout columns: " + ", ".join(missing_columns))

    selected_languages = tuple(languages)
    if not selected_languages:
        raise ValueError("At least one holdout language is required.")

    language_values = frame["language"].fillna("").astype(str).str.lower().str.strip()
    filtered = frame[language_values.isin(selected_languages)].copy()
    filtered["language"] = language_values[language_values.isin(selected_languages)]
    filtered = filtered.reset_index(drop=True)
    if filtered.empty:
        raise ValueError("No holdout rows remain after language filtering.")
    return filtered


def evaluate_loaded_models(
    models: Mapping[str, Any],
    frame: pd.DataFrame,
) -> dict[str, FoldEvaluation]:
    """Evaluate the fitted trainers held by the serving model wrappers."""
    missing_tasks = [task_name for task_name in REQUIRED_TASKS if task_name not in models]
    if missing_tasks:
        raise KeyError("Missing promoted holdout tasks: " + ", ".join(missing_tasks))

    return {
        task_name: evaluate_fitted_trainer(
            trainer=models[task_name].trainer,
            frame=frame,
        )
        for task_name in REQUIRED_TASKS
    }


def write_holdout_artifacts(
    *,
    evaluations: Mapping[str, FoldEvaluation],
    output_dir: Path,
    dataset_metadata: Mapping[str, Any],
    model_metadata: Mapping[str, Any],
) -> Path:
    """Write a compact JSON summary and detailed CSV evaluation artifacts."""
    missing_tasks = [
        task_name for task_name in REQUIRED_TASKS if task_name not in evaluations
    ]
    if missing_tasks:
        raise KeyError("Missing holdout evaluations: " + ", ".join(missing_tasks))

    output_dir.mkdir(parents=True, exist_ok=True)
    task_summaries: dict[str, Any] = {}

    for task_name in REQUIRED_TASKS:
        result = evaluations[task_name]
        artifact_names = {
            "language_metrics": f"{task_name}_language_metrics.csv",
            "per_class_metrics": f"{task_name}_per_class_metrics.csv",
            "confusion_matrix": f"{task_name}_confusion_matrix.csv",
            "per_class_confusion": f"{task_name}_per_class_confusion.csv",
        }
        result.language_metrics.to_csv(
            output_dir / artifact_names["language_metrics"], index=False
        )
        result.per_class_metrics.to_csv(
            output_dir / artifact_names["per_class_metrics"], index=False
        )
        result.confusion_matrix.to_csv(
            output_dir / artifact_names["confusion_matrix"], index=False
        )
        result.per_class_confusion.to_csv(
            output_dir / artifact_names["per_class_confusion"], index=False
        )
        task_summaries[task_name] = {
            "accuracy": float(result.fold_metrics["accuracy"]),
            "macro_f1": float(result.fold_metrics["macro_f1"]),
            "model": dict(model_metadata.get(task_name, {})),
            "artifacts": artifact_names,
        }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "dataset": dict(dataset_metadata),
                "tasks": task_summaries,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return summary_path
