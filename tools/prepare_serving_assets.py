"""Internal helper for promoting selected MLflow run artifacts into serving assets."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import joblib
import matplotlib
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPERIMENT_ID = "167858914656602414"
SELECTED_RUNS = {
    "queue": "7db335f24d524827aa3f27bba2aa8d8d",
    "priority": "5b3febadbb894da28b181a253e99ef56",
}

ROOT = Path(__file__).resolve().parents[1]
MLRUNS_ROOT = ROOT / "mlruns" / EXPERIMENT_ID
SERVING_ROOT = ROOT / "serving_assets"
MODELS_ROOT = SERVING_ROOT / "models"
CONFIGS_ROOT = SERVING_ROOT / "configs"
DOCS_ROOT = ROOT / "docs"
DOCS_ASSETS_ROOT = DOCS_ROOT / "assets"
SUMMARY_PATH = SERVING_ROOT / "promoted_models.json"

HEADLINE_METRICS = {
    "accuracy_mean": "cv_accuracy_mean",
    "accuracy_std": "cv_accuracy_std",
    "macro_f1_mean": "cv_macro_f1_mean",
    "macro_f1_std": "cv_macro_f1_std",
}


def _normalize_run_config(
    run_config: dict[str, object], trainer: object
) -> dict[str, object]:
    normalized = dict(run_config)
    dataset = dict(normalized.get("dataset", {}))
    dataset.pop("path", None)
    normalized["dataset"] = dataset

    preprocessing = dict(normalized.get("preprocessing", {}))
    preprocessing.setdefault(
        "analyzer",
        getattr(trainer.preprocessor.pipeline.feature_extractor, "analyzer", "word"),
    )
    normalized["preprocessing"] = preprocessing

    if "feature_matrix" not in normalized:
        feature_families = ["tfidf"]
        if preprocessing.get("length_feature_enabled"):
            feature_families.append("length")
        normalized["feature_matrix"] = {
            "rows": normalized["dataset"]["row_count"],
            "columns": len(trainer.feature_names_),
            "feature_families": feature_families,
        }

    return normalized


def _read_metric(run_id: str, metric_name: str) -> float:
    metric_path = MLRUNS_ROOT / run_id / "metrics" / metric_name
    raw_line = metric_path.read_text(encoding="utf-8").strip()
    _, metric_value, _ = raw_line.split(maxsplit=2)
    return float(metric_value)


def _build_headline_metrics(run_id: str) -> dict[str, float]:
    return {
        public_name: _read_metric(run_id, metric_name)
        for public_name, metric_name in HEADLINE_METRICS.items()
    }


def _load_confusion_matrix(run_id: str) -> tuple[list[str], list[list[float]]]:
    matrix_path = MLRUNS_ROOT / run_id / "artifacts" / "confusion_matrix_mean.csv"
    with matrix_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    labels = rows[0][1:]
    matrix = np.array([[float(value) for value in row[1:]] for row in rows[1:]])
    return labels, matrix


def _draw_confusion_matrix_image(
    *,
    task_name: str,
    labels: list[str],
    matrix: list[list[float]],
    output_path: Path,
) -> None:
    figure_size = (12, 10) if len(labels) > 3 else (7, 6)
    fig, ax = plt.subplots(figsize=figure_size)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    display.plot(ax=ax, cmap="Blues", values_format=".1f", colorbar=False)

    ax.set_title(f"{task_name.title()} Mean CV Confusion Matrix", pad=16)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")

    if len(labels) > 3:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax.tick_params(axis="x", labelsize=9)
        ax.tick_params(axis="y", labelsize=9)
    else:
        ax.tick_params(axis="both", labelsize=12)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIGS_ROOT.mkdir(parents=True, exist_ok=True)
    DOCS_ROOT.mkdir(parents=True, exist_ok=True)
    DOCS_ASSETS_ROOT.mkdir(parents=True, exist_ok=True)

    serving_config = {
        "app": {
            "title": "Ticket Triage Demo",
        },
        "models": {},
    }
    promoted_summary = {
        "tracking_experiment_id": EXPERIMENT_ID,
        "serving_config_path": "serving_config.json",
    }

    for task_name, run_id in SELECTED_RUNS.items():
        artifact_root = MLRUNS_ROOT / run_id / "artifacts"
        model_source = artifact_root / "trained_model.joblib"
        run_config_source = artifact_root / "run_config.json"

        trainer = joblib.load(model_source)
        run_config = json.loads(run_config_source.read_text(encoding="utf-8"))
        normalized_run_config = _normalize_run_config(run_config, trainer)

        model_destination = MODELS_ROOT / f"{task_name}_model.joblib"
        run_config_destination = CONFIGS_ROOT / f"{task_name}_run_config.json"

        shutil.copy2(model_source, model_destination)
        run_config_destination.write_text(
            json.dumps(normalized_run_config, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        serving_config["models"][task_name] = {
            "run_id": run_id,
            "model_path": str(model_destination.relative_to(SERVING_ROOT)).replace(
                "\\", "/"
            ),
            "run_config_path": str(
                run_config_destination.relative_to(SERVING_ROOT)
            ).replace("\\", "/"),
        }
        promoted_summary[task_name] = {
            "run_id": run_id,
            "dataset": normalized_run_config["dataset"],
            "training": normalized_run_config["training"],
            "headline_metrics": _build_headline_metrics(run_id),
            "model": normalized_run_config["model"],
            "preprocessing": normalized_run_config["preprocessing"],
            "feature_matrix": normalized_run_config["feature_matrix"],
            "serving_artifacts": {
                "model_path": str(model_destination.relative_to(SERVING_ROOT)).replace(
                    "\\", "/"
                ),
                "run_config_path": str(
                    run_config_destination.relative_to(SERVING_ROOT)
                ).replace("\\", "/"),
            },
            "docs_artifacts": {
                "confusion_matrix_path": f"assets/{task_name}-confusion-matrix.png",
            },
        }

        labels, matrix = _load_confusion_matrix(run_id)
        _draw_confusion_matrix_image(
            task_name=task_name,
            labels=labels,
            matrix=matrix,
            output_path=DOCS_ASSETS_ROOT / f"{task_name}-confusion-matrix.png",
        )

    (SERVING_ROOT / "serving_config.json").write_text(
        json.dumps(serving_config, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    SUMMARY_PATH.write_text(
        json.dumps(promoted_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
