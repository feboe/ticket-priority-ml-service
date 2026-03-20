from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from tests.helpers import build_smoke_dataset, run_training_smoke


MLFLOW_AVAILABLE = importlib.util.find_spec("mlflow") is not None
JOBLIB_AVAILABLE = importlib.util.find_spec("joblib") is not None

if MLFLOW_AVAILABLE:
    import mlflow

if JOBLIB_AVAILABLE:
    import joblib


@unittest.skipUnless(MLFLOW_AVAILABLE, "mlflow is not installed")
class TrainingTrackingSmokeTests(unittest.TestCase):
    def test_training_creates_two_top_level_task_runs_with_minimal_artifacts(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        for algorithm, expected_model_family in [
            ("logreg", "LogisticRegression"),
            ("linear_svc", "LinearSVC"),
        ]:
            with self.subTest(algorithm=algorithm):
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_root = Path(temp_dir)
                    data_path = temp_root / "smoke_dataset.csv"
                    tracking_uri = f"file:{(temp_root / 'mlruns').resolve().as_posix()}"
                    experiment_name = f"ticket-priority-smoke-{algorithm}"

                    build_smoke_dataset(data_path)
                    run_training_smoke(
                        repo_root,
                        data_path,
                        tracking_uri,
                        experiment_name,
                        algorithm=algorithm,
                    )

                    mlflow.set_tracking_uri(tracking_uri)
                    experiment = mlflow.get_experiment_by_name(experiment_name)
                    self.assertIsNotNone(experiment)

                    runs = mlflow.search_runs(
                        [experiment.experiment_id], output_format="pandas"
                    )
                    self.assertEqual(len(runs), 2)
                    self.assertEqual(
                        set(runs["tags.task_name"].tolist()), {"queue", "priority"}
                    )
                    self.assertEqual(
                        set(runs["params.dataset_id"].tolist()), {"smoke_dataset"}
                    )
                    self.assertEqual(
                        set(runs["params.algorithm"].tolist()), {algorithm}
                    )
                    self.assertEqual(set(runs["params.analyzer"].tolist()), {"word"})
                    self.assertEqual(
                        set(runs["params.model_family"].tolist()),
                        {expected_model_family},
                    )
                    self.assertEqual(
                        set(runs["params.length_feature_enabled"].tolist()), {"false"}
                    )

                    self.assertIn("metrics.cv_accuracy_mean", runs.columns)
                    self.assertIn("metrics.cv_macro_f1_mean", runs.columns)
                    self.assertTrue(
                        any(
                            column.startswith("metrics.cv_precision_mean__")
                            for column in runs.columns
                        )
                    )
                    self.assertTrue(
                        any(
                            column.startswith("metrics.cv_tp_mean__")
                            for column in runs.columns
                        )
                    )

                    for run_id in runs["run_id"].tolist():
                        run = mlflow.get_run(run_id)
                        artifact_root = _artifact_root_from_uri(run.info.artifact_uri)
                        model_path = artifact_root / "trained_model.joblib"

                        self.assertTrue((artifact_root / "per_class_metrics.csv").exists())
                        self.assertTrue(
                            (artifact_root / "per_class_confusion.csv").exists()
                        )
                        self.assertTrue(
                            (artifact_root / "confusion_matrix_mean.csv").exists()
                        )
                        self.assertTrue(
                            (artifact_root / "confusion_matrix_std.csv").exists()
                        )
                        self.assertTrue((artifact_root / "run_config.json").exists())
                        self.assertTrue(model_path.exists())

                        run_config = json.loads(
                            (artifact_root / "run_config.json").read_text(
                                encoding="utf-8"
                            )
                        )
                        self.assertEqual(
                            set(run_config),
                            {
                                "run",
                                "task",
                                "dataset",
                                "training",
                                "model",
                                "preprocessing",
                                "feature_matrix",
                                "artifacts",
                            },
                        )
                        self.assertEqual(run_config["dataset"]["id"], "smoke_dataset")
                        self.assertEqual(
                            run_config["task"]["name"], run.data.tags["task_name"]
                        )
                        self.assertEqual(run_config["model"]["algorithm"], algorithm)
                        self.assertEqual(run_config["preprocessing"]["analyzer"], "word")
                        self.assertFalse(
                            run_config["preprocessing"]["length_feature_enabled"]
                        )
                        self.assertEqual(
                            run_config["artifacts"]["trained_model"],
                            "trained_model.joblib",
                        )
                        self.assertEqual(
                            run_config["feature_matrix"]["rows"],
                            run_config["dataset"]["row_count"],
                        )
                        self.assertGreater(run_config["feature_matrix"]["columns"], 0)
                        self.assertEqual(
                            run_config["feature_matrix"]["feature_families"], ["tfidf"]
                        )
                        self.assertNotIn("shared_metadata", run_config)
                        self.assertNotIn("tracking", run_config)

                        if JOBLIB_AVAILABLE:
                            trainer = joblib.load(model_path)
                            self.assertEqual(trainer.algorithm, algorithm)
                            self.assertIn(trainer.task_name, {"queue", "priority"})
                            self.assertEqual(
                                run_config["feature_matrix"]["columns"],
                                len(trainer.feature_names_),
                            )



def _artifact_root_from_uri(artifact_uri: str) -> Path:
    parsed = urlparse(artifact_uri)
    if parsed.scheme == "file":
        return Path(url2pathname(unquote(parsed.path)))
    return Path(artifact_uri)


if __name__ == "__main__":
    unittest.main()

