"""Command-line entrypoint for evaluating promoted models on the holdout."""

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
from src.tracking import (
    configure_tracking,
    log_holdout_evaluation_runs,
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
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default="file:./mlruns",
        help="MLflow tracking URI.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="ticket-priority-holdout",
        help="MLflow experiment for frozen holdout evaluation runs.",
    )
    parser.add_argument(
        "--run-group",
        type=str,
        default="frozen-holdout",
        help="Logical group name shared by the task evaluation runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    actual_sha256 = verify_sha256(args.data, HOLDOUT_SHA256)
    raw_frame = pd.read_csv(args.data)
    holdout_frame = prepare_holdout_frame(raw_frame)

    dataset_metadata = {
        "file": args.data.name,
        "sha256": actual_sha256,
        "source_row_count": int(len(raw_frame)),
        "evaluated_row_count": int(len(holdout_frame)),
        "languages": list(HOLDOUT_LANGUAGES),
    }
    service = TicketRoutingService.from_config(args.serving_config)
    model_metadata = service.describe_models()
    evaluations = evaluate_loaded_models(service.models, holdout_frame)

    configure_tracking(args.tracking_uri, args.experiment_name)
    evaluation_run_ids = log_holdout_evaluation_runs(
        evaluations=evaluations,
        dataset_metadata=dataset_metadata,
        model_metadata=model_metadata,
        run_group=args.run_group,
    )
    summary_path = write_holdout_artifacts(
        evaluations=evaluations,
        output_dir=args.output_dir,
        dataset_metadata=dataset_metadata,
        model_metadata=model_metadata,
        tracking_metadata={
            "uri": args.tracking_uri,
            "experiment_name": args.experiment_name,
            "run_group": args.run_group,
            "run_ids": evaluation_run_ids,
        },
    )

    print(f"Holdout rows: {len(holdout_frame)}")
    for task_name, result in evaluations.items():
        print(
            f"{task_name}: accuracy={result.fold_metrics['accuracy']:.4f}, "
            f"macro_f1={result.fold_metrics['macro_f1']:.4f}, "
            f"mlflow_run_id={evaluation_run_ids[task_name]}"
        )
    print(f"Artifacts: {summary_path}")


if __name__ == "__main__":
    main()
