"""Top-level entrypoint for stratified cross-validation training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.classification import (
    SUPPORTED_ALGORITHMS,
    evaluate_task,
    fit_final_model,
)
from src.tracking import (
    build_base_run_name,
    build_dataset_metadata,
    build_run_config,
    build_shared_tracking_payload,
    build_task_tracking_payload,
    configure_tracking,
    log_dataframe_artifact,
    log_json_artifact,
    log_model_artifact,
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
        "--algorithm",
        type=str,
        choices=SUPPORTED_ALGORITHMS,
        default="logreg",
        help="Classifier algorithm used for both tasks.",
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
    return parser.parse_args()


def print_task_results(task_name: str, task_results: dict[str, Any]) -> None:
    metrics = task_results["overall_metrics"]
    print(task_name)
    print(f"  cv_accuracy_mean: {metrics['cv_accuracy_mean']:.4f}")
    print(f"  cv_accuracy_std: {metrics['cv_accuracy_std']:.4f}")
    print(f"  cv_macro_f1_mean: {metrics['cv_macro_f1_mean']:.4f}")
    print(f"  cv_macro_f1_std: {metrics['cv_macro_f1_std']:.4f}")


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
    resolved_base_run_name = build_base_run_name(
        run_group=args.run_group,
        dataset_id=dataset_metadata["dataset_id"],
        cv_folds=args.cv_folds,
        seed=args.random_state,
        run_name=args.run_name,
    )
    shared_params, shared_tags = build_shared_tracking_payload(
        run_group=args.run_group,
        algorithm=args.algorithm,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
        stratify_columns=list(STRATIFY_TARGET_COLUMNS),
        dataset_metadata=dataset_metadata,
    )

    configure_tracking(args.tracking_uri, args.experiment_name)
    for task_name in TASK_NAMES:
        task_results = evaluate_task(
            task_name=task_name,
            folds=folds,
            random_state=args.random_state,
            algorithm=args.algorithm,
        )
        final_trainer = fit_final_model(
            task_name=task_name,
            df=df,
            random_state=args.random_state,
            algorithm=args.algorithm,
        )
        print_task_results(task_name, task_results)

        run_name = f"{resolved_base_run_name}::{args.algorithm}::{task_name}"
        task_params, task_tags = build_task_tracking_payload(
            task_name=task_name,
            task_results=task_results,
        )
        task_tags = {**shared_tags, **task_tags}
        task_params = {**shared_params, **task_params}
        run_config = build_run_config(
            run_group=args.run_group,
            cv_folds=args.cv_folds,
            random_state=args.random_state,
            task_name=task_name,
            task_results=task_results,
            dataset_metadata=dataset_metadata,
            final_trainer=final_trainer,
            run_name=run_name,
            stratify_columns=list(STRATIFY_TARGET_COLUMNS),
        )

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
            log_model_artifact(final_trainer, "trained_model.joblib")


if __name__ == "__main__":
    main()
