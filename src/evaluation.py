"""Evaluation helpers for fold and cross-validation reporting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


@dataclass
class FoldEvaluation:
    """Structured evaluation outputs for a single held-out fold."""

    fold_metrics: dict[str, int | float]
    language_metrics: pd.DataFrame
    per_class_metrics: pd.DataFrame
    confusion_matrix: pd.DataFrame
    per_class_confusion: pd.DataFrame


def evaluate_fold(
    *,
    fold_index: int,
    y_true: pd.Series,
    y_pred: pd.Series,
    label_ids: Sequence[int],
    label_names: Sequence[str],
    languages: pd.Series | Sequence[str] | None = None,
) -> FoldEvaluation:
    """Compute scalar, per-class, and confusion metrics for one fold."""
    labels = list(label_ids)
    names = list(label_names)
    truth = pd.Series(y_true).reset_index(drop=True)
    predictions = pd.Series(y_pred).reset_index(drop=True)

    fold_metrics = {
        "fold": fold_index,
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(
            f1_score(truth, predictions, labels=labels, average="macro", zero_division=0)
        ),
    }
    language_metrics = _build_language_metrics(
        fold_index=fold_index,
        truth=truth,
        predictions=predictions,
        labels=labels,
        languages=languages,
    )

    precision, recall, f1_values, support = precision_recall_fscore_support(
        truth,
        predictions,
        labels=labels,
        zero_division=0,
    )
    per_class_metrics = pd.DataFrame(
        {
            "fold": fold_index,
            "label_id": labels,
            "label": names,
            "precision": precision,
            "recall": recall,
            "f1": f1_values,
            "support": support.astype(int),
        }
    )

    matrix = confusion_matrix(truth, predictions, labels=labels)
    confusion_rows: list[dict[str, int | str]] = []
    per_class_confusion_rows: list[dict[str, int | str]] = []
    total = int(matrix.sum())

    for actual_position, actual_label_id in enumerate(labels):
        for predicted_position, predicted_label_id in enumerate(labels):
            confusion_rows.append(
                {
                    "fold": fold_index,
                    "actual_label_id": actual_label_id,
                    "actual_label": names[actual_position],
                    "predicted_label_id": predicted_label_id,
                    "predicted_label": names[predicted_position],
                    "count": int(matrix[actual_position, predicted_position]),
                }
            )

    for label_position, label_id in enumerate(labels):
        tp = int(matrix[label_position, label_position])
        fn = int(matrix[label_position, :].sum() - tp)
        fp = int(matrix[:, label_position].sum() - tp)
        tn = int(total - tp - fn - fp)
        per_class_confusion_rows.append(
            {
                "fold": fold_index,
                "label_id": label_id,
                "label": names[label_position],
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )

    return FoldEvaluation(
        fold_metrics=fold_metrics,
        language_metrics=language_metrics,
        per_class_metrics=per_class_metrics,
        confusion_matrix=pd.DataFrame(confusion_rows),
        per_class_confusion=pd.DataFrame(per_class_confusion_rows),
    )


def summarize_cv_results(fold_evaluations: Sequence[FoldEvaluation]) -> dict[str, object]:
    """Aggregate fold evaluations into compact CV summaries for MLflow."""
    if not fold_evaluations:
        raise ValueError("At least one fold evaluation is required.")

    fold_metrics = (
        pd.DataFrame([evaluation.fold_metrics for evaluation in fold_evaluations])
        .sort_values("fold")
        .reset_index(drop=True)
    )
    language_metrics = pd.concat(
        [evaluation.language_metrics for evaluation in fold_evaluations],
        ignore_index=True,
    )
    per_class_metrics = pd.concat(
        [evaluation.per_class_metrics for evaluation in fold_evaluations],
        ignore_index=True,
    )
    confusion_matrix_rows = pd.concat(
        [evaluation.confusion_matrix for evaluation in fold_evaluations],
        ignore_index=True,
    )
    per_class_confusion = pd.concat(
        [evaluation.per_class_confusion for evaluation in fold_evaluations],
        ignore_index=True,
    )

    overall_metrics = {
        "cv_accuracy_mean": float(fold_metrics["accuracy"].mean()),
        "cv_accuracy_std": float(fold_metrics["accuracy"].std(ddof=0)),
        "cv_macro_f1_mean": float(fold_metrics["macro_f1"].mean()),
        "cv_macro_f1_std": float(fold_metrics["macro_f1"].std(ddof=0)),
    }
    language_summary = _summarize_language_metrics(language_metrics)

    per_class_summary = (
        per_class_metrics.groupby(["label_id", "label"], sort=False)
        .agg(
            precision_mean=("precision", "mean"),
            precision_std=("precision", lambda values: float(values.std(ddof=0))),
            recall_mean=("recall", "mean"),
            recall_std=("recall", lambda values: float(values.std(ddof=0))),
            f1_mean=("f1", "mean"),
            f1_std=("f1", lambda values: float(values.std(ddof=0))),
            support_mean=("support", "mean"),
            support_std=("support", lambda values: float(values.std(ddof=0))),
        )
        .reset_index()
        .sort_values("label_id")
        .reset_index(drop=True)
    )

    per_class_confusion_summary = (
        per_class_confusion.groupby(["label_id", "label"], sort=False)
        .agg(
            tp_mean=("tp", "mean"),
            tp_std=("tp", lambda values: float(values.std(ddof=0))),
            fp_mean=("fp", "mean"),
            fp_std=("fp", lambda values: float(values.std(ddof=0))),
            fn_mean=("fn", "mean"),
            fn_std=("fn", lambda values: float(values.std(ddof=0))),
            tn_mean=("tn", "mean"),
            tn_std=("tn", lambda values: float(values.std(ddof=0))),
        )
        .reset_index()
        .sort_values("label_id")
        .reset_index(drop=True)
    )

    confusion_summary = (
        confusion_matrix_rows.groupby(
            [
                "actual_label_id",
                "actual_label",
                "predicted_label_id",
                "predicted_label",
            ],
            sort=False,
        )
        .agg(
            count_mean=("count", "mean"),
            count_std=("count", lambda values: float(values.std(ddof=0))),
        )
        .reset_index()
    )
    confusion_matrix_mean = _pivot_confusion_summary(confusion_summary, "count_mean")
    confusion_matrix_std = _pivot_confusion_summary(confusion_summary, "count_std")

    mlflow_metrics = {
        **overall_metrics,
        **_flatten_language_metrics(language_summary),
        **_flatten_per_class_metrics(per_class_summary),
        **_flatten_per_class_confusion_metrics(per_class_confusion_summary),
    }

    return {
        "overall_metrics": overall_metrics,
        "mlflow_metrics": mlflow_metrics,
        "language_metrics": language_summary,
        "per_class_metrics": per_class_summary,
        "per_class_confusion": per_class_confusion_summary,
        "confusion_matrix_mean": confusion_matrix_mean,
        "confusion_matrix_std": confusion_matrix_std,
    }


def _flatten_per_class_metrics(per_class_summary: pd.DataFrame) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for row in per_class_summary.itertuples(index=False):
        label_slug = _slugify(row.label)
        metrics[f"cv_precision_mean__{label_slug}"] = float(row.precision_mean)
        metrics[f"cv_precision_std__{label_slug}"] = float(row.precision_std)
        metrics[f"cv_recall_mean__{label_slug}"] = float(row.recall_mean)
        metrics[f"cv_recall_std__{label_slug}"] = float(row.recall_std)
        metrics[f"cv_f1_mean__{label_slug}"] = float(row.f1_mean)
        metrics[f"cv_f1_std__{label_slug}"] = float(row.f1_std)
        metrics[f"cv_support_mean__{label_slug}"] = float(row.support_mean)
        metrics[f"cv_support_std__{label_slug}"] = float(row.support_std)
    return metrics


def _flatten_language_metrics(language_summary: pd.DataFrame) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for row in language_summary.itertuples(index=False):
        language_slug = _slugify(row.language)
        metrics[f"cv_accuracy_mean__lang_{language_slug}"] = float(row.accuracy_mean)
        metrics[f"cv_accuracy_std__lang_{language_slug}"] = float(row.accuracy_std)
        metrics[f"cv_macro_f1_mean__lang_{language_slug}"] = float(row.macro_f1_mean)
        metrics[f"cv_macro_f1_std__lang_{language_slug}"] = float(row.macro_f1_std)
        metrics[f"cv_sample_count_mean__lang_{language_slug}"] = float(row.sample_count_mean)
        metrics[f"cv_sample_count_std__lang_{language_slug}"] = float(row.sample_count_std)
    return metrics


def _flatten_per_class_confusion_metrics(
    per_class_confusion_summary: pd.DataFrame,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for row in per_class_confusion_summary.itertuples(index=False):
        label_slug = _slugify(row.label)
        metrics[f"cv_tp_mean__{label_slug}"] = float(row.tp_mean)
        metrics[f"cv_tp_std__{label_slug}"] = float(row.tp_std)
        metrics[f"cv_fp_mean__{label_slug}"] = float(row.fp_mean)
        metrics[f"cv_fp_std__{label_slug}"] = float(row.fp_std)
        metrics[f"cv_fn_mean__{label_slug}"] = float(row.fn_mean)
        metrics[f"cv_fn_std__{label_slug}"] = float(row.fn_std)
        metrics[f"cv_tn_mean__{label_slug}"] = float(row.tn_mean)
        metrics[f"cv_tn_std__{label_slug}"] = float(row.tn_std)
    return metrics


def _pivot_confusion_summary(
    confusion_summary: pd.DataFrame, value_column: str
) -> pd.DataFrame:
    matrix = (
        confusion_summary.pivot(
            index="actual_label",
            columns="predicted_label",
            values=value_column,
        )
        .fillna(0.0)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    matrix.index.name = "actual_label"
    return matrix.reset_index()


def _build_language_metrics(
    *,
    fold_index: int,
    truth: pd.Series,
    predictions: pd.Series,
    labels: Sequence[int],
    languages: pd.Series | Sequence[str] | None,
) -> pd.DataFrame:
    if languages is None:
        return pd.DataFrame(
            columns=["fold", "language", "accuracy", "macro_f1", "sample_count"]
        )

    language_series = (
        pd.Series(languages)
        .reset_index(drop=True)
        .fillna("unknown")
        .astype(str)
        .str.lower()
        .str.strip()
    )
    language_series = language_series.mask(language_series.eq(""), "unknown")
    if len(language_series) != len(truth):
        raise ValueError("Language values must match the number of truth labels.")

    metrics_rows: list[dict[str, int | float | str]] = []
    for language in language_series.drop_duplicates().tolist():
        mask = language_series == language
        truth_subset = truth[mask]
        prediction_subset = predictions[mask]
        metrics_rows.append(
            {
                "fold": fold_index,
                "language": language,
                "accuracy": float(accuracy_score(truth_subset, prediction_subset)),
                "macro_f1": float(
                    f1_score(
                        truth_subset,
                        prediction_subset,
                        labels=list(labels),
                        average="macro",
                        zero_division=0,
                    )
                ),
                "sample_count": int(mask.sum()),
            }
        )
    return pd.DataFrame(metrics_rows)


def _summarize_language_metrics(language_metrics: pd.DataFrame) -> pd.DataFrame:
    if language_metrics.empty:
        return pd.DataFrame(
            columns=[
                "language",
                "accuracy_mean",
                "accuracy_std",
                "macro_f1_mean",
                "macro_f1_std",
                "sample_count_mean",
                "sample_count_std",
            ]
        )

    return (
        language_metrics.groupby("language", sort=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", lambda values: float(values.std(ddof=0))),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", lambda values: float(values.std(ddof=0))),
            sample_count_mean=("sample_count", "mean"),
            sample_count_std=("sample_count", lambda values: float(values.std(ddof=0))),
        )
        .reset_index()
    )


def _slugify(value: str) -> str:
    return "_".join(value.lower().split())
