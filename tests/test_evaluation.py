from __future__ import annotations

import unittest

from src.evaluation import evaluate_fold, summarize_cv_results


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
        self.assertNotIn("label_slug", result.per_class_confusion.columns)

        high_row = result.per_class_metrics[result.per_class_metrics["label"] == "high"].iloc[0]
        self.assertEqual(int(high_row["support"]), 0)
        self.assertEqual(float(high_row["precision"]), 0.0)
        self.assertEqual(float(high_row["recall"]), 0.0)

        high_confusion = result.per_class_confusion[
            result.per_class_confusion["label"] == "high"
        ].iloc[0]
        self.assertEqual(int(high_confusion["tp"]), 0)
        self.assertEqual(int(high_confusion["fp"]), 0)
        self.assertEqual(int(high_confusion["fn"]), 0)
        self.assertEqual(int(high_confusion["tn"]), 4)

    def test_evaluate_fold_builds_language_metrics_when_languages_are_provided(self) -> None:
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

    def test_summarize_cv_results_computes_overall_mean_std_and_per_class_metrics(self) -> None:
        fold_one = evaluate_fold(
            fold_index=1,
            y_true=[0, 0, 1, 1],
            y_pred=[0, 1, 1, 1],
            label_ids=self.label_ids,
            label_names=self.label_names,
        )
        fold_two = evaluate_fold(
            fold_index=2,
            y_true=[0, 1, 2, 2],
            y_pred=[0, 2, 2, 1],
            label_ids=self.label_ids,
            label_names=self.label_names,
        )

        summary = summarize_cv_results([fold_one, fold_two])

        self.assertAlmostEqual(summary["overall_metrics"]["cv_accuracy_mean"], 0.625)
        self.assertAlmostEqual(summary["overall_metrics"]["cv_accuracy_std"], 0.125)
        self.assertAlmostEqual(summary["overall_metrics"]["cv_macro_f1_mean"], 0.49444444444444446)
        self.assertAlmostEqual(summary["overall_metrics"]["cv_macro_f1_std"], 0.005555555555555536)
        self.assertNotIn("label_slug", summary["per_class_metrics"].columns)
        self.assertNotIn("label_slug", summary["per_class_confusion"].columns)

        high_row = summary["per_class_metrics"][summary["per_class_metrics"]["label"] == "high"].iloc[0]
        self.assertAlmostEqual(float(high_row["support_mean"]), 1.0)
        self.assertAlmostEqual(float(high_row["support_std"]), 1.0)
        self.assertIn("cv_precision_mean__high", summary["mlflow_metrics"])
        self.assertIn("cv_tp_mean__high", summary["mlflow_metrics"])

    def test_summarize_cv_results_builds_confusion_matrix_mean_std(self) -> None:
        fold_one = evaluate_fold(
            fold_index=1,
            y_true=[0, 0, 1, 1],
            y_pred=[0, 1, 1, 1],
            label_ids=self.label_ids,
            label_names=self.label_names,
        )
        fold_two = evaluate_fold(
            fold_index=2,
            y_true=[0, 1, 2, 2],
            y_pred=[0, 2, 2, 1],
            label_ids=self.label_ids,
            label_names=self.label_names,
        )

        summary = summarize_cv_results([fold_one, fold_two])
        confusion_mean = summary["confusion_matrix_mean"].set_index("actual_label")
        confusion_std = summary["confusion_matrix_std"].set_index("actual_label")
        per_class_confusion = summary["per_class_confusion"]

        self.assertAlmostEqual(float(confusion_mean.loc["low", "low"]), 1.0)
        self.assertAlmostEqual(float(confusion_mean.loc["low", "medium"]), 0.5)
        self.assertAlmostEqual(float(confusion_std.loc["low", "medium"]), 0.5)

        high_row = per_class_confusion[per_class_confusion["label"] == "high"].iloc[0]
        self.assertAlmostEqual(float(high_row["tp_mean"]), 0.5)
        self.assertAlmostEqual(float(high_row["tp_std"]), 0.5)
        self.assertAlmostEqual(float(high_row["tn_mean"]), 2.5)
        self.assertAlmostEqual(float(high_row["tn_std"]), 1.5)

    def test_summarize_cv_results_aggregates_language_metrics_and_flattens_mlflow_keys(self) -> None:
        fold_one = evaluate_fold(
            fold_index=1,
            y_true=[0, 0, 1, 1],
            y_pred=[0, 1, 1, 1],
            label_ids=self.label_ids,
            label_names=self.label_names,
            languages=["en", "en", "de", "de"],
        )
        fold_two = evaluate_fold(
            fold_index=2,
            y_true=[0, 1, 2, 2],
            y_pred=[0, 2, 2, 1],
            label_ids=self.label_ids,
            label_names=self.label_names,
            languages=["en", "de", "en", "de"],
        )

        summary = summarize_cv_results([fold_one, fold_two])
        language_metrics = summary["language_metrics"].set_index("language")

        self.assertEqual(language_metrics.index.tolist(), ["en", "de"])
        self.assertAlmostEqual(float(language_metrics.loc["en", "accuracy_mean"]), 0.75)
        self.assertAlmostEqual(float(language_metrics.loc["en", "accuracy_std"]), 0.25)
        self.assertAlmostEqual(float(language_metrics.loc["de", "accuracy_mean"]), 0.5)
        self.assertAlmostEqual(float(language_metrics.loc["de", "accuracy_std"]), 0.5)
        self.assertAlmostEqual(float(language_metrics.loc["en", "sample_count_mean"]), 2.0)
        self.assertIn("cv_accuracy_mean__lang_en", summary["mlflow_metrics"])
        self.assertIn("cv_macro_f1_mean__lang_de", summary["mlflow_metrics"])


if __name__ == "__main__":
    unittest.main()
