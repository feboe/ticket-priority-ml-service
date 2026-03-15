"""Top-level entrypoint for baseline model training."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.classification import ClassificationTrainer
from src.regression import RegressionTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ticket service baseline models.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data") / "synthetic_it_support_tickets.csv",
        help="Path to the training CSV file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.data)

    product_area_trainer = ClassificationTrainer(task_name="product_area")
    priority_trainer = ClassificationTrainer(task_name="priority")
    regression_trainer = RegressionTrainer()

    metrics = {
        "product_area": product_area_trainer.evaluate(df),
        "priority": priority_trainer.evaluate(df),
        "resolution_time_hours": regression_trainer.evaluate(df),
    }

    for task_name, task_metrics in metrics.items():
        print(task_name)
        for metric_name, value in task_metrics.items():
            print(f"  {metric_name}: {value:.4f}")


if __name__ == "__main__":
    main()
