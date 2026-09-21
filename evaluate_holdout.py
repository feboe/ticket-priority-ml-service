"""Evaluate the fixed promoted models on the frozen EN/DE holdout."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from app.service import DEFAULT_SERVING_CONFIG_PATH, TicketRoutingService
from src.holdout_evaluation import (
    HOLDOUT_LANGUAGES,
    HOLDOUT_SHA256,
    evaluate_loaded_models,
    prepare_holdout_frame,
    verify_sha256,
    write_holdout_artifacts,
)

DEFAULT_HOLDOUT_PATH = Path("data") / "dataset-tickets-multi-lang3-4k.csv"
DEFAULT_RESULTS_DIR = Path("results") / "holdout"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate promoted models on the frozen English/German holdout."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_HOLDOUT_PATH,
        help="Path to the frozen holdout CSV.",
    )
    parser.add_argument(
        "--serving-config",
        type=Path,
        default=DEFAULT_SERVING_CONFIG_PATH,
        help="Path to the promoted serving configuration.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory for JSON and CSV evaluation artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    actual_sha256 = verify_sha256(args.data, HOLDOUT_SHA256)
    raw_frame = pd.read_csv(args.data)
    holdout_frame = prepare_holdout_frame(raw_frame)

    service = TicketRoutingService.from_config(args.serving_config)
    evaluations = evaluate_loaded_models(service.models, holdout_frame)
    summary_path = write_holdout_artifacts(
        evaluations=evaluations,
        output_dir=args.output_dir,
        dataset_metadata={
            "file": args.data.name,
            "sha256": actual_sha256,
            "source_row_count": int(len(raw_frame)),
            "evaluated_row_count": int(len(holdout_frame)),
            "languages": list(HOLDOUT_LANGUAGES),
        },
        model_metadata=service.describe_models(),
    )

    print(f"Holdout rows: {len(holdout_frame)}")
    for task_name, result in evaluations.items():
        print(
            f"{task_name}: accuracy={result.fold_metrics['accuracy']:.4f}, "
            f"macro_f1={result.fold_metrics['macro_f1']:.4f}"
        )
    print(f"Artifacts: {summary_path}")


if __name__ == "__main__":
    main()
