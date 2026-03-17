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
    run_name: str
    algorithm: str
    model_family: str
    analyzer: str
    feature_families: list[str]
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
            "run_name": self.run_name,
            "algorithm": self.algorithm,
            "model_family": self.model_family,
            "analyzer": self.analyzer,
            "feature_families": self.feature_families,
        }


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
        models: dict[str, LoadedTaskModel] = {}

        for task_name, spec in payload["models"].items():
            trainer = joblib.load(base_dir / spec["model_path"])
            run_config = json.loads(
                (base_dir / spec["run_config_path"]).read_text(encoding="utf-8")
            )
            preprocessing = run_config.get("preprocessing", {})
            analyzer = preprocessing.get(
                "analyzer",
                getattr(
                    trainer.preprocessor.pipeline.feature_extractor, "analyzer", "word"
                ),
            )
            feature_matrix = run_config.get("feature_matrix", {})
            feature_families = feature_matrix.get("feature_families")
            if not feature_families:
                feature_families = ["tfidf"]
                if run_config.get("preprocessing", {}).get("length_feature_enabled"):
                    feature_families.append("length")
            models[task_name] = LoadedTaskModel(
                task_name=task_name,
                run_id=spec["run_id"],
                run_name=run_config["run"]["name"],
                algorithm=run_config["model"]["algorithm"],
                model_family=run_config["model"]["model_family"],
                analyzer=analyzer,
                feature_families=feature_families,
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
    ) -> dict[str, Any]:
        frame = pd.DataFrame(
            [
                {
                    "subject": subject,
                    "body": body,
                }
            ]
        )
        predictions = {
            task_name: task_model.predict(frame)
            for task_name, task_model in self.models.items()
        }
        return {
            "input": {
                "subject": subject,
                "body": body,
            },
            "predictions": predictions,
            "models": self.describe_models(),
        }


@lru_cache(maxsize=1)
def get_default_service() -> TicketRoutingService:
    return TicketRoutingService.from_config(DEFAULT_SERVING_CONFIG_PATH)
