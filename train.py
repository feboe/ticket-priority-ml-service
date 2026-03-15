"""Top-level entrypoint for stratified cross-validation training."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.classification import ClassificationTrainer
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
    return parser.parse_args()


def evaluate_task(
    task_name: str,
    folds: list,
    random_state: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for fold_index, split in enumerate(folds, start=1):
        trainer = ClassificationTrainer(task_name=task_name, random_state=random_state)
        metrics = trainer.evaluate_split(split)
        rows.append(
            {
                "fold": fold_index,
                "train_rows": int(trainer.split_metadata_["train_rows"]),
                "test_rows": int(trainer.split_metadata_["test_rows"]),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
            }
        )
    return pd.DataFrame(rows)


def print_task_results(task_name: str, fold_metrics: pd.DataFrame) -> None:
    print(task_name)
    for row in fold_metrics.itertuples(index=False):
        print(
            f"  fold_{row.fold}: train_rows={row.train_rows} test_rows={row.test_rows} "
            f"accuracy={row.accuracy:.4f} macro_f1={row.macro_f1:.4f}"
        )
    print(f"  mean_accuracy: {fold_metrics['accuracy'].mean():.4f}")
    print(f"  std_accuracy: {fold_metrics['accuracy'].std(ddof=0):.4f}")
    print(f"  mean_macro_f1: {fold_metrics['macro_f1'].mean():.4f}")
    print(f"  std_macro_f1: {fold_metrics['macro_f1'].std(ddof=0):.4f}")


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.data)
    folds = make_stratified_folds(
        frame=df,
        target_columns=STRATIFY_TARGET_COLUMNS,
        n_splits=args.cv_folds,
        random_state=args.random_state,
    )

    for task_name in TASK_NAMES:
        fold_metrics = evaluate_task(task_name, folds, args.random_state)
        print_task_results(task_name, fold_metrics)


if __name__ == "__main__":
    main()
