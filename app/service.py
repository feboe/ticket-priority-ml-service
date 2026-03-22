"""Inference service for the ticket triage MVP."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

DEFAULT_SERVING_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "serving_assets" / "serving_config.json"
)


@dataclass
class LoadedTaskModel:
    task_name: str
    run_id: str
    algorithm: str
    model_family: str
    c: float
    feature_summary: str
    dataset_id: str
    cv_macro_f1_mean: float
    cv_accuracy_mean: float
    trainer: Any

    def predict(self, frame: pd.DataFrame) -> dict[str, Any]:
        features = self.trainer.preprocessor.transform(frame)
        scores = np.asarray(self.trainer.model.decision_function(features))
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])

        score_row = scores[0]
        class_ids = np.asarray(self.trainer.model.classes_)
        ranked_indices = np.argsort(score_row)
        best_index = int(ranked_indices[-1])
        runner_up_index = (
            int(ranked_indices[-2]) if ranked_indices.size >= 2 else best_index
        )

        predicted_id = int(class_ids[best_index])
        runner_up_id = int(class_ids[runner_up_index])
        predicted_label = self.trainer.target_mapping_[predicted_id]
        runner_up_label = self.trainer.target_mapping_[runner_up_id]
        gap = float(score_row[best_index] - score_row[runner_up_index])

        return {
            "label": predicted_label,
            "runner_up_label": runner_up_label,
            "margin_gap": round(gap, 6),
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "algorithm": self.algorithm,
            "model_family": self.model_family,
            "c": self.c,
            "feature_summary": self.feature_summary,
            "dataset_id": self.dataset_id,
            "cv_macro_f1_mean": self.cv_macro_f1_mean,
            "cv_accuracy_mean": self.cv_accuracy_mean,
        }


def _format_ngram_range(ngram_min: Any, ngram_max: Any) -> str:
    if ngram_min == ngram_max:
        return f"{ngram_min}-gram"
    return f"{ngram_min}-{ngram_max} grams"


def _build_feature_summary(
    *, preprocessing: dict[str, Any], feature_matrix: dict[str, Any]
) -> str:
    analyzer = str(preprocessing.get("analyzer", "word"))
    ngram_min = preprocessing.get("ngram_min", 1)
    ngram_max = preprocessing.get("ngram_max", 1)
    base_summary = f"TF-IDF {analyzer} {_format_ngram_range(ngram_min, ngram_max)}"

    length_enabled = bool(
        preprocessing.get("length_feature_enabled")
        or "length" in feature_matrix.get("feature_families", [])
    )
    if length_enabled:
        return f"{base_summary} + length"
    return base_summary


class TicketRoutingService:
    """Load fixed task models and provide prediction helpers."""

    def __init__(self, *, title: str, models: dict[str, LoadedTaskModel]) -> None:
        self.title = title
        self.models = models

    @classmethod
    def from_config(
        cls, config_path: Path | str = DEFAULT_SERVING_CONFIG_PATH
    ) -> "TicketRoutingService":
        config_path = Path(config_path)
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        base_dir = config_path.parent
        promoted_models = json.loads(
            (base_dir / "promoted_models.json").read_text(encoding="utf-8")
        )
        models: dict[str, LoadedTaskModel] = {}

        for task_name, spec in payload["models"].items():
            trainer = joblib.load(base_dir / spec["model_path"])
            run_config = json.loads(
                (base_dir / spec["run_config_path"]).read_text(encoding="utf-8")
            )
            promoted_spec = promoted_models[task_name]
            feature_matrix = promoted_spec.get(
                "feature_matrix", run_config.get("feature_matrix", {})
            )
            preprocessing = promoted_spec.get(
                "preprocessing", run_config.get("preprocessing", {})
            )
            model_config = promoted_spec.get("model", run_config.get("model", {}))
            headline_metrics = promoted_spec.get("headline_metrics", {})
            dataset = promoted_spec.get("dataset", {})

            models[task_name] = LoadedTaskModel(
                task_name=task_name,
                run_id=promoted_spec.get("run_id", spec["run_id"]),
                algorithm=str(model_config["algorithm"]),
                model_family=str(model_config["model_family"]),
                c=float(model_config["C"]),
                feature_summary=_build_feature_summary(
                    preprocessing=preprocessing,
                    feature_matrix=feature_matrix,
                ),
                dataset_id=str(dataset["id"]),
                cv_macro_f1_mean=float(headline_metrics["macro_f1_mean"]),
                cv_accuracy_mean=float(headline_metrics["accuracy_mean"]),
                trainer=trainer,
            )

        return cls(title=payload["app"]["title"], models=models)

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "tasks": list(self.models),
            "models": self.describe_models(),
        }

    def describe_models(self) -> dict[str, Any]:
        return {
            task_name: task_model.metadata()
            for task_name, task_model in self.models.items()
        }

    def predict_ticket(
        self,
        *,
        subject: str,
        body: str,
        language: str | None = None,
    ) -> dict[str, Any]:
        row: dict[str, str] = {"subject": subject, "body": body}
        if language:
            row["language"] = language
        frame = pd.DataFrame([row])
        predictions = {
            task_name: task_model.predict(frame)
            for task_name, task_model in self.models.items()
        }
        input_payload: dict[str, str | None] = {"subject": subject, "body": body}
        if language:
            input_payload["language"] = language
        return {
            "input": input_payload,
            "predictions": predictions,
            "models": self.describe_models(),
        }


@lru_cache(maxsize=1)
def get_default_service() -> TicketRoutingService:
    return TicketRoutingService.from_config(DEFAULT_SERVING_CONFIG_PATH)
