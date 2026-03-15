"""Top-level entrypoint for queue and priority baseline training."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.classification import ClassificationTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train queue and priority baseline models.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data") / "aa_dataset-tickets-multi-lang-5-2-50-version.csv",
        help="Path to the training CSV file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.data)

    queue_trainer = ClassificationTrainer(task_name="queue")
    priority_trainer = ClassificationTrainer(task_name="priority")

    metrics = {
        "queue": queue_trainer.evaluate(df),
        "priority": priority_trainer.evaluate(df),
    }

    for task_name, task_metrics in metrics.items():
        print(task_name)
        for metric_name, value in task_metrics.items():
            print(f"  {metric_name}: {value:.4f}")


if __name__ == "__main__":
    main()
