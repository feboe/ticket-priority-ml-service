"""Project source package."""

from .classification import (
    ClassificationTrainer,
    evaluate_task,
    fit_final_model,
)
from .evaluation import FoldEvaluation, evaluate_fold, summarize_cv_results
from .preprocessing import (
    PriorityPreprocessor,
    QueuePreprocessor,
    TargetEncoder,
    TextPreparationPipeline,
    TfidfFeatureExtractor,
    TfidfTargetPreprocessor,
    VectorizedDataset,
)
from .tracking import (
    build_dataset_metadata,
    build_run_config,
    build_shared_tracking_payload,
    build_task_tracking_payload,
)

__all__ = [
    "ClassificationTrainer",
    "FoldEvaluation",
    "PriorityPreprocessor",
    "QueuePreprocessor",
    "TargetEncoder",
    "TextPreparationPipeline",
    "TfidfFeatureExtractor",
    "TfidfTargetPreprocessor",
    "VectorizedDataset",
    "build_dataset_metadata",
    "build_run_config",
    "build_shared_tracking_payload",
    "build_task_tracking_payload",
    "evaluate_fold",
    "evaluate_task",
    "fit_final_model",
    "summarize_cv_results",
]
