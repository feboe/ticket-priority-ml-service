from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import mlflow

from src.evaluation import evaluate_fold
from src.tracking import configure_tracking, log_holdout_evaluation_runs
from tests.helpers import artifact_root_from_uri

class HoldoutTrackingTests(unittest.TestCase):
    def test_logs_one_traceable_run_per_task(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            tracking_uri = f"file:{(temp_root / 'mlruns').resolve().as_posix()}"
            experiment_name = "holdout-tracking-test"
            evaluation = evaluate_fold(
                fold_index=1,
                y_true=[0, 0, 1, 1],
                y_pred=[0, 1, 1, 1],
                label_ids=[0, 1],
                label_names=["first", "second"],
                languages=["en", "en", "de", "de"],
            )

            configure_tracking(tracking_uri, experiment_name)
            run_ids = log_holdout_evaluation_runs(
                evaluations={"queue": evaluation, "priority": evaluation},
                dataset_metadata={
                    "file": "holdout.csv",
                    "sha256": "a" * 64,
                    "source_row_count": 6,
                    "evaluated_row_count": 4,
                    "languages": ["en", "de"],
                },
                model_metadata={
                    "queue": {
                        "run_id": "queue-source-run",
                        "dataset_id": "training-data",
                    },
                    "priority": {
                        "run_id": "priority-source-run",
                        "dataset_id": "training-data",
                    },
                },
                run_group="holdout-test",
            )

            self.assertEqual(set(run_ids), {"queue", "priority"})
            experiment = mlflow.get_experiment_by_name(experiment_name)
            self.assertIsNotNone(experiment)
            runs = mlflow.search_runs(
                [experiment.experiment_id], output_format="pandas"
            )
            self.assertEqual(len(runs), 2)
            self.assertEqual(
                set(runs["tags.run_type"]), {"holdout_evaluation"}
            )
            self.assertEqual(set(runs["params.evaluation_type"]), {"frozen_holdout"})
            self.assertEqual(
                set(runs["params.dataset_sha256"]), {"a" * 64}
            )
            self.assertEqual(
                set(runs["tags.source_model_run_id"]),
                {"queue-source-run", "priority-source-run"},
            )
            self.assertIn("metrics.holdout_accuracy", runs.columns)
            self.assertIn("metrics.holdout_macro_f1", runs.columns)
            self.assertIn("metrics.holdout_accuracy__lang_en", runs.columns)
            self.assertIn("metrics.holdout_macro_f1__lang_de", runs.columns)

            for run_id in run_ids.values():
                run = mlflow.get_run(run_id)
                artifact_root = artifact_root_from_uri(run.info.artifact_uri)
                for artifact_name in (
                    "language_metrics.csv",
                    "per_class_metrics.csv",
                    "confusion_matrix.csv",
                    "per_class_confusion.csv",
                    "holdout_run_config.json",
                ):
                    self.assertTrue((artifact_root / artifact_name).exists())

                run_config = json.loads(
                    (artifact_root / "holdout_run_config.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(run_config["evaluation"]["type"], "frozen_holdout")
                self.assertEqual(run_config["dataset"]["sha256"], "a" * 64)
                self.assertEqual(
                    run_config["source_model"]["run_id"],
                    run.data.tags["source_model_run_id"],
                )

if __name__ == "__main__":
    unittest.main()
