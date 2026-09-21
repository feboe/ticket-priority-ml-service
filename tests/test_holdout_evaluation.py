from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from src.evaluation import evaluate_fold
from src.holdout_evaluation import (
    compute_sha256,
    evaluate_loaded_models,
    prepare_holdout_frame,
    verify_sha256,
    write_holdout_artifacts,
)


class HoldoutEvaluationTests(unittest.TestCase):
    def test_verify_sha256_accepts_expected_digest_and_rejects_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "holdout.csv"
            path.write_bytes(b"holdout fixture\n")
            expected = compute_sha256(path)

            self.assertEqual(verify_sha256(path, expected), expected)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                verify_sha256(path, "0" * 64)

    def test_prepare_holdout_frame_filters_and_normalizes_supported_languages(
        self,
    ) -> None:
        frame = pd.DataFrame(
            {
                "subject": ["one", "two", "three"],
                "body": ["body", "body", "body"],
                "language": [" EN ", "de", "es"],
                "queue": ["A", "A", "A"],
                "priority": ["low", "medium", "high"],
            }
        )

        result = prepare_holdout_frame(frame)

        self.assertEqual(result["language"].tolist(), ["en", "de"])
        self.assertEqual(result["subject"].tolist(), ["one", "two"])

    def test_prepare_holdout_frame_rejects_missing_schema(self) -> None:
        with self.assertRaisesRegex(KeyError, "Missing holdout columns: priority"):
            prepare_holdout_frame(
                pd.DataFrame(
                    {
                        "subject": ["one"],
                        "body": ["body"],
                        "language": ["en"],
                        "queue": ["A"],
                    }
                )
            )

    @patch("src.holdout_evaluation.evaluate_fitted_trainer")
    def test_evaluate_loaded_models_uses_promoted_trainers(self, evaluate_mock) -> None:
        queue_result = _evaluation_result()
        priority_result = _evaluation_result()
        evaluate_mock.side_effect = [queue_result, priority_result]
        queue_trainer = object()
        priority_trainer = object()
        models = {
            "queue": SimpleNamespace(trainer=queue_trainer),
            "priority": SimpleNamespace(trainer=priority_trainer),
        }
        frame = pd.DataFrame({"language": ["en"]})

        results = evaluate_loaded_models(models, frame)

        self.assertIs(results["queue"], queue_result)
        self.assertIs(results["priority"], priority_result)
        self.assertIs(evaluate_mock.call_args_list[0].kwargs["trainer"], queue_trainer)
        self.assertIs(evaluate_mock.call_args_list[1].kwargs["trainer"], priority_trainer)

    def test_write_holdout_artifacts_writes_summary_and_detailed_csvs(self) -> None:
        evaluations = {
            "queue": _evaluation_result(),
            "priority": _evaluation_result(),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            summary_path = write_holdout_artifacts(
                evaluations=evaluations,
                output_dir=output_dir,
                dataset_metadata={"sha256": "abc", "evaluated_row_count": 4},
                model_metadata={
                    "queue": {"run_id": "queue-run"},
                    "priority": {"run_id": "priority-run"},
                },
                tracking_metadata={
                    "experiment_name": "holdout-test",
                    "run_ids": {
                        "queue": "queue-evaluation-run",
                        "priority": "priority-evaluation-run",
                    },
                },
            )

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["dataset"]["evaluated_row_count"], 4)
            self.assertEqual(
                summary["tracking"]["run_ids"]["queue"],
                "queue-evaluation-run",
            )
            self.assertEqual(summary["tasks"]["queue"]["model"]["run_id"], "queue-run")
            self.assertAlmostEqual(summary["tasks"]["priority"]["accuracy"], 0.75)
            for task_name in ("queue", "priority"):
                artifact_names = summary["tasks"][task_name]["artifacts"].values()
                for artifact_name in artifact_names:
                    self.assertTrue((output_dir / artifact_name).exists())


def _evaluation_result():
    return evaluate_fold(
        fold_index=1,
        y_true=[0, 0, 1, 1],
        y_pred=[0, 1, 1, 1],
        label_ids=[0, 1],
        label_names=["first", "second"],
        languages=["en", "en", "de", "de"],
    )


if __name__ == "__main__":
    unittest.main()
