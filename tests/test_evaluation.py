from __future__ import annotations

import unittest

import pandas as pd

from src.evaluation import (
    evaluate_fitted_trainer,
    evaluate_fold,
    summarize_cv_results,
)


def _two_fold_evaluations(
    *, label_ids: list[int], label_names: list[str], with_languages: bool = False
):
    language_rows = (
        (["en", "en", "de", "de"], ["en", "de", "en", "de"])
        if with_languages
        else (None, None)
    )
    return [
        evaluate_fold(
            fold_index=1,
            y_true=[0, 0, 1, 1],
            y_pred=[0, 1, 1, 1],
            label_ids=label_ids,
            label_names=label_names,
            languages=language_rows[0],
        ),
        evaluate_fold(
            fold_index=2,
            y_true=[0, 1, 2, 2],
            y_pred=[0, 2, 2, 1],
            label_ids=label_ids,
            label_names=label_names,
            languages=language_rows[1],
        ),
    ]


class EvaluationModuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.label_ids = [0, 1, 2]
        self.label_names = ["low", "medium", "high"]

    def test_evaluate_fold_preserves_label_order_and_zero_support_rows(self) -> None:
        result = evaluate_fold(
            fold_index=1,
            y_true=[0, 0, 1, 1],
            y_pred=[0, 1, 1, 1],
            label_ids=self.label_ids,
            label_names=self.label_names,
        )

        self.assertEqual(result.per_class_metrics["label"].tolist(), self.label_names)
        self.assertNotIn("label_slug", result.per_class_metrics.columns)

        high_row = result.per_class_metrics[
            result.per_class_metrics["label"] == "high"
        ].iloc[0]
        self.assertEqual(int(high_row["support"]), 0)
        self.assertEqual(float(high_row["precision"]), 0.0)
        self.assertEqual(float(high_row["recall"]), 0.0)

    def test_evaluate_fold_builds_language_metrics_when_languages_are_provided(
        self,
    ) -> None:
        result = evaluate_fold(
            fold_index=1,
            y_true=[0, 0, 1, 1],
            y_pred=[0, 1, 1, 1],
            label_ids=self.label_ids,
            label_names=self.label_names,
            languages=["en", "en", "de", "de"],
        )

        self.assertEqual(result.language_metrics["language"].tolist(), ["en", "de"])
        self.assertEqual(result.language_metrics["sample_count"].tolist(), [2, 2])
        self.assertAlmostEqual(float(result.language_metrics.iloc[0]["accuracy"]), 0.5)
        self.assertAlmostEqual(float(result.language_metrics.iloc[1]["accuracy"]), 1.0)

    def test_evaluate_fitted_trainer_reuses_fold_metric_path(self) -> None:
        trainer = _FakeTrainer()
        frame = pd.DataFrame(
            {
                "priority": ["low", "medium", "high"],
                "language": ["en", "de", "en"],
                "prediction_id": [0, 2, 2],
            }
        )

        result = evaluate_fitted_trainer(trainer=trainer, frame=frame)

        self.assertAlmostEqual(result.fold_metrics["accuracy"], 2 / 3)
        self.assertEqual(result.per_class_metrics["label"].tolist(), self.label_names)
        self.assertEqual(result.language_metrics["language"].tolist(), ["en", "de"])

    def test_summarize_cv_results_computes_overall_mean_std_and_per_class_metrics(
        self,
    ) -> None:
        fold_evaluations = _two_fold_evaluations(
            label_ids=self.label_ids,
            label_names=self.label_names,
        )

        summary = summarize_cv_results(fold_evaluations)

        self.assertAlmostEqual(summary["overall_metrics"]["cv_accuracy_mean"], 0.625)
        self.assertAlmostEqual(summary["overall_metrics"]["cv_accuracy_std"], 0.125)
        self.assertAlmostEqual(
            summary["overall_metrics"]["cv_macro_f1_mean"], 0.49444444444444446
        )
        self.assertAlmostEqual(
            summary["overall_metrics"]["cv_macro_f1_std"], 0.005555555555555536
        )
        self.assertNotIn("label_slug", summary["per_class_metrics"].columns)

        high_row = summary["per_class_metrics"][
            summary["per_class_metrics"]["label"] == "high"
        ].iloc[0]
        self.assertAlmostEqual(float(high_row["support_mean"]), 1.0)
        self.assertAlmostEqual(float(high_row["support_std"]), 1.0)
        self.assertIn("cv_precision_mean__high", summary["mlflow_metrics"])

    def test_summarize_cv_results_builds_confusion_matrix_mean_std(self) -> None:
        fold_evaluations = _two_fold_evaluations(
            label_ids=self.label_ids,
            label_names=self.label_names,
        )

        summary = summarize_cv_results(fold_evaluations)
        confusion_mean = summary["confusion_matrix_mean"].set_index("actual_label")
        confusion_std = summary["confusion_matrix_std"].set_index("actual_label")

        self.assertAlmostEqual(float(confusion_mean.loc["low", "low"]), 1.0)
        self.assertAlmostEqual(float(confusion_mean.loc["low", "medium"]), 0.5)
        self.assertAlmostEqual(float(confusion_std.loc["low", "medium"]), 0.5)

    def test_summarize_cv_results_aggregates_language_metrics_and_flattens_mlflow_keys(
        self,
    ) -> None:
        fold_evaluations = _two_fold_evaluations(
            label_ids=self.label_ids,
            label_names=self.label_names,
            with_languages=True,
        )

        summary = summarize_cv_results(fold_evaluations)
        language_metrics = summary["language_metrics"].set_index("language")

        self.assertEqual(language_metrics.index.tolist(), ["en", "de"])
        self.assertAlmostEqual(float(language_metrics.loc["en", "accuracy_mean"]), 0.75)
        self.assertAlmostEqual(float(language_metrics.loc["en", "accuracy_std"]), 0.25)
        self.assertAlmostEqual(float(language_metrics.loc["de", "accuracy_mean"]), 0.5)
        self.assertAlmostEqual(float(language_metrics.loc["de", "accuracy_std"]), 0.5)
        self.assertAlmostEqual(
            float(language_metrics.loc["en", "sample_count_mean"]), 2.0
        )
        self.assertIn("cv_accuracy_mean__lang_en", summary["mlflow_metrics"])
        self.assertIn("cv_macro_f1_mean__lang_de", summary["mlflow_metrics"])


class _FakeTargetEncoder:
    def transform(self, target: pd.Series) -> pd.Series:
        mapping = {"low": 0, "medium": 1, "high": 2}
        return target.map(mapping)


class _FakePipeline:
    target_encoder = _FakeTargetEncoder()


class _FakePreprocessor:
    pipeline = _FakePipeline()

    def transform(self, frame: pd.DataFrame):
        return frame[["prediction_id"]].to_numpy()


class _FakeModel:
    def predict(self, features):
        return features[:, 0]


class _FakeTrainer:
    task_name = "priority"
    preprocessor = _FakePreprocessor()
    model = _FakeModel()

    @staticmethod
    def get_target_column() -> str:
        return "priority"

    @staticmethod
    def get_label_order() -> list[int]:
        return [0, 1, 2]

    @staticmethod
    def get_label_names() -> list[str]:
        return ["low", "medium", "high"]


if __name__ == "__main__":
    unittest.main()
