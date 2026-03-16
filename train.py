"""Top-level entrypoint for stratified cross-validation training."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.classification import ClassificationTrainer
from src.evaluation import evaluate_fold, summarize_cv_results
from src.tracking import (
    build_base_run_name,
    build_runtime_metadata,
    collect_git_metadata,
    configure_tracking,
    log_dataframe_artifact,
    log_json_artifact,
    log_run_metadata,
    start_run,
)
from src.training_utils import make_stratified_folds


TASK_NAMES = ("queue", "priority")
STRATIFY_TARGET_COLUMNS = ("queue", "priority")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run shared stratified cross-validation for queue and priority models."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data") / "aa_dataset-tickets-multi-lang-5-2-50-version.csv",
        help="Path to the training CSV file.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of stratified folds to evaluate.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used when shuffling folds.",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default="file:./mlruns",
        help="MLflow tracking URI.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="ticket-priority-cv",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--run-group",
        type=str,
        default="baseline",
        help="Logical group name for related experiment runs.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional override for the base run name.",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default=None,
        help="Optional free-text note stored with the run.",
    )
    return parser.parse_args()


def evaluate_task(
    task_name: str,
    folds: list,
    random_state: int,
) -> dict[str, Any]:
    fold_evaluations = []
    trainer_config: dict[str, Any] | None = None

    for fold_index, split in enumerate(folds, start=1):
        trainer = ClassificationTrainer(task_name=task_name, random_state=random_state)
        trainer.fit_on_split(split)

        fold_evaluations.append(
            evaluate_fold(
                fold_index=fold_index,
                y_true=trainer.get_test_truth(),
                y_pred=trainer.get_test_predictions(),
                label_ids=trainer.get_label_order(),
                label_names=trainer.get_label_names(),
            )
        )

        if trainer_config is None:
            trainer_config = {
                "target_column": trainer.get_target_column(),
                "model": trainer.get_model_config(),
                "preprocessing": trainer.get_preprocessing_config(),
            }

    task_results = summarize_cv_results(fold_evaluations)
    task_results["task_config"] = trainer_config or {}
    return task_results


def print_task_results(task_name: str, task_results: dict[str, Any]) -> None:
    metrics = task_results["overall_metrics"]
    print(task_name)
    print(f"  cv_accuracy_mean: {metrics['cv_accuracy_mean']:.4f}")
    print(f"  cv_accuracy_std: {metrics['cv_accuracy_std']:.4f}")
    print(f"  cv_micro_f1_mean: {metrics['cv_micro_f1_mean']:.4f}")
    print(f"  cv_micro_f1_std: {metrics['cv_micro_f1_std']:.4f}")
    print(f"  cv_macro_f1_mean: {metrics['cv_macro_f1_mean']:.4f}")
    print(f"  cv_macro_f1_std: {metrics['cv_macro_f1_std']:.4f}")


def build_dataset_metadata(df: pd.DataFrame, data_path: Path) -> dict[str, Any]:
    version_values = []
    if "version" in df.columns:
        version_values = sorted(df["version"].dropna().astype(str).unique().tolist())

    return {
        "dataset_path": str(data_path.resolve()),
        "dataset_name": data_path.name,
        "dataset_id": data_path.stem,
        "dataset_row_count": int(len(df)),
        "dataset_version_values": version_values,
    }


def build_shared_tracking_payload(
    *,
    args: argparse.Namespace,
    dataset_metadata: dict[str, Any],
    git_metadata: dict[str, str],
    runtime_metadata: dict[str, str],
    resolved_base_run_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    timestamp = datetime.now(timezone.utc).isoformat()
    params = {
        "run_group": args.run_group,
        "dataset_name": dataset_metadata["dataset_name"],
        "dataset_id": dataset_metadata["dataset_id"],
        "dataset_row_count": dataset_metadata["dataset_row_count"],
        "cv_folds": args.cv_folds,
        "random_state": args.random_state,
        "stratify_columns": list(STRATIFY_TARGET_COLUMNS),
    }
    tags = {
        "run_group": args.run_group,
        "run_base_name": resolved_base_run_name,
        "timestamp": timestamp,
        "dataset_path": dataset_metadata["dataset_path"],
        "dataset_name": dataset_metadata["dataset_name"],
        "dataset_id": dataset_metadata["dataset_id"],
        "dataset_version_values": dataset_metadata["dataset_version_values"],
        "git_commit": git_metadata["git_commit"],
        "git_branch": git_metadata["git_branch"],
        "git_dirty": git_metadata["git_dirty"],
        "command": runtime_metadata["command"],
        "python_version": runtime_metadata["python_version"],
        "tracking_uri": args.tracking_uri,
        "experiment_name": args.experiment_name,
        "notes": args.notes,
        "mlflow.note.content": args.notes,
    }
    return params, tags


def build_task_tracking_payload(
    *,
    task_name: str,
    base_run_name: str,
    task_results: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
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
        "run_name": f"{base_run_name}::{task_name}",
        "class_labels": per_class_metrics["label"].tolist(),
    }
    run_config = {
        "task_name": task_name,
        "target_column": task_config["target_column"],
        "labels": per_class_metrics[["label_id", "label", "label_slug"]].to_dict(
            orient="records"
        ),
        "model": task_config["model"],
        "preprocessing": task_config["preprocessing"],
    }
    return params, tags, run_config


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.data)
    folds = make_stratified_folds(
        frame=df,
        target_columns=STRATIFY_TARGET_COLUMNS,
        n_splits=args.cv_folds,
        random_state=args.random_state,
    )

    dataset_metadata = build_dataset_metadata(df, args.data)
    git_metadata = collect_git_metadata(Path.cwd())
    runtime_metadata = build_runtime_metadata(sys.argv)
    resolved_base_run_name = build_base_run_name(
        run_group=args.run_group,
        dataset_id=dataset_metadata["dataset_id"],
        cv_folds=args.cv_folds,
        seed=args.random_state,
        git_sha=git_metadata["git_commit"],
        run_name=args.run_name,
    )
    shared_params, shared_tags = build_shared_tracking_payload(
        args=args,
        dataset_metadata=dataset_metadata,
        git_metadata=git_metadata,
        runtime_metadata=runtime_metadata,
        resolved_base_run_name=resolved_base_run_name,
    )

    configure_tracking(args.tracking_uri, args.experiment_name)
    for task_name in TASK_NAMES:
        task_results = evaluate_task(task_name, folds, args.random_state)
        print_task_results(task_name, task_results)

        run_name = f"{resolved_base_run_name}::{task_name}"
        task_params, task_tags, run_config = build_task_tracking_payload(
            task_name=task_name,
            base_run_name=resolved_base_run_name,
            task_results=task_results,
        )
        task_tags = {**shared_tags, **task_tags}
        task_params = {**shared_params, **task_params}
        run_config = {
            **run_config,
            "tracking": {
                "tracking_uri": args.tracking_uri,
                "experiment_name": args.experiment_name,
            },
            "shared_metadata": {
                **dataset_metadata,
                **git_metadata,
                **runtime_metadata,
                "run_group": args.run_group,
                "run_name": run_name,
                "notes": args.notes,
                "cv_folds": args.cv_folds,
                "random_state": args.random_state,
                "stratify_columns": list(STRATIFY_TARGET_COLUMNS),
            },
        }

        with start_run(run_name):
            log_run_metadata(
                params=task_params,
                tags=task_tags,
                metrics=task_results["mlflow_metrics"],
            )
            log_dataframe_artifact(
                task_results["per_class_metrics"], "per_class_metrics.csv"
            )
            log_dataframe_artifact(
                task_results["per_class_confusion"], "per_class_confusion.csv"
            )
            log_dataframe_artifact(
                task_results["confusion_matrix_mean"], "confusion_matrix_mean.csv"
            )
            log_dataframe_artifact(
                task_results["confusion_matrix_std"], "confusion_matrix_std.csv"
            )
            log_json_artifact(run_config, "run_config.json")


if __name__ == "__main__":
    main()
