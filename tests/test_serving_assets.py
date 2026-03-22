from __future__ import annotations

import json
import unittest
from pathlib import Path


class PromotedModelSummaryTests(unittest.TestCase):
    def test_summary_exists_and_matches_serving_config(self) -> None:
        summary_path = Path("serving_assets/promoted_models.json")
        serving_config_path = Path("serving_assets/serving_config.json")

        self.assertTrue(summary_path.exists())
        self.assertTrue(serving_config_path.exists())

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        serving_config = json.loads(serving_config_path.read_text(encoding="utf-8"))

        self.assertEqual(
            set(summary),
            {
                "priority",
                "queue",
                "serving_config_path",
                "tracking_experiment_id",
            },
        )
        self.assertEqual(summary["serving_config_path"], "serving_config.json")

        expected_metric_keys = {
            "accuracy_mean",
            "accuracy_std",
            "macro_f1_mean",
            "macro_f1_std",
        }
        expected_task_keys = {
            "dataset",
            "docs_artifacts",
            "feature_matrix",
            "headline_metrics",
            "model",
            "preprocessing",
            "run_id",
            "serving_artifacts",
            "training",
        }
        for task_name, spec in serving_config["models"].items():
            task_summary = summary[task_name]
            self.assertEqual(set(task_summary), expected_task_keys)
            self.assertEqual(task_summary["run_id"], spec["run_id"])
            self.assertEqual(
                task_summary["serving_artifacts"]["model_path"],
                spec["model_path"],
            )
            self.assertEqual(
                task_summary["serving_artifacts"]["run_config_path"],
                spec["run_config_path"],
            )
            confusion_matrix_path = (
                Path("docs") / task_summary["docs_artifacts"]["confusion_matrix_path"]
            )
            self.assertTrue(confusion_matrix_path.exists())
            self.assertEqual(
                set(task_summary["headline_metrics"]),
                expected_metric_keys,
            )
            for metric_value in task_summary["headline_metrics"].values():
                self.assertIsInstance(metric_value, float)
                self.assertGreater(metric_value, 0.0)


if __name__ == "__main__":
    unittest.main()
