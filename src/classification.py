"""Classification training workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from .evaluation import FoldEvaluation, evaluate_fold, summarize_cv_results
from .preprocessing import PriorityPreprocessor, QueuePreprocessor
from .training_utils import HoldoutSplit


PREPROCESSOR_FACTORY = {
    "queue": QueuePreprocessor,
    "priority": PriorityPreprocessor,
}

SUPPORTED_ALGORITHMS = ("logreg", "linear_svc")

CLASS_WEIGHT_BY_TASK = {
    "queue": "balanced",
    "priority": "balanced",
}

C_BY_ALGORITHM_AND_TASK = {
    "logreg": {
        "queue": 2.0,
        "priority": 2.0,
    },
    "linear_svc": {
        "queue": 16.0,
        "priority": 12.0,
    },
}

MAX_ITER_BY_ALGORITHM = {
    "logreg": 5000,
    "linear_svc": 5000,
}


@dataclass
class ClassificationTrainer:
    """Reusable trainer for queue and priority classification."""

    task_name: str
    algorithm: str = "logreg"
    random_state: int = 42
    model: LogisticRegression | LinearSVC = field(init=False)
    preprocessor: QueuePreprocessor | PriorityPreprocessor = field(init=False)
    feature_names_: list[str] = field(default_factory=list, init=False)
    target_mapping_: dict[int, str] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.task_name not in PREPROCESSOR_FACTORY:
            raise ValueError(
                f"Unsupported classification task '{self.task_name}'. "
                f"Choose one of: {', '.join(PREPROCESSOR_FACTORY)}."
            )
        if self.algorithm not in SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm '{self.algorithm}'. "
                f"Choose one of: {', '.join(SUPPORTED_ALGORITHMS)}."
            )

        self.model = self._build_model()
        self.preprocessor = PREPROCESSOR_FACTORY[self.task_name]()

    def fit_train(self, df: pd.DataFrame) -> ClassificationTrainer:
        train_data = self.preprocessor.fit_transform(df)
        self.model.fit(train_data.X, train_data.y)
        self.feature_names_ = train_data.feature_names
        self.target_mapping_ = train_data.target_mapping or {}
        return self

    def fit_full(self, df: pd.DataFrame) -> ClassificationTrainer:
        return self.fit_train(df)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if not self.target_mapping_:
            raise ValueError("Target mapping is unavailable. Fit the trainer first.")
        X = self.preprocessor.transform(df)
        predictions = self.model.predict(X)
        decoded = pd.Series(predictions).map(
            lambda value: self.target_mapping_[int(value)]
        )
        return pd.Series(decoded, name=self.task_name)

    def get_label_order(self) -> list[int]:
        return sorted(self.target_mapping_)

    def get_label_names(self) -> list[str]:
        return [self.target_mapping_[label_id] for label_id in self.get_label_order()]

    def get_target_column(self) -> str:
        return self.preprocessor.pipeline.target_column

    def get_model_config(self) -> dict[str, Any]:
        config = {
            "algorithm": self.algorithm,
            "model_family": type(self.model).__name__,
            "C": self.model.C,
            "class_weight": self.model.class_weight,
            "max_iter": self.model.max_iter,
        }
        if hasattr(self.model, "solver"):
            config["solver"] = self.model.solver
        return config

    def get_preprocessing_config(self) -> dict[str, Any]:
        feature_extractor = self.preprocessor.pipeline.feature_extractor
        return {
            "min_df": feature_extractor.min_df,
            "max_df": feature_extractor.max_df,
            "ngram_min": feature_extractor.ngram_range[0],
            "ngram_max": feature_extractor.ngram_range[1],
            "analyzer": feature_extractor.analyzer,
            "sublinear_tf": feature_extractor.sublinear_tf,
            "length_feature_enabled": self.preprocessor.pipeline.length_feature_enabled,
        }

    def get_feature_summary(self) -> dict[str, Any]:
        feature_families = ["tfidf"]
        if self.preprocessor.pipeline.length_feature_enabled:
            feature_families.append("length")
        return {
            "feature_count": len(self.feature_names_),
            "feature_families": feature_families,
        }

    def _build_model(self) -> LogisticRegression | LinearSVC:
        c_value = C_BY_ALGORITHM_AND_TASK[self.algorithm][self.task_name]
        class_weight = CLASS_WEIGHT_BY_TASK[self.task_name]
        max_iter = MAX_ITER_BY_ALGORITHM[self.algorithm]

        if self.algorithm == "logreg":
            return LogisticRegression(
                C=c_value,
                max_iter=max_iter,
                random_state=self.random_state,
                class_weight=class_weight,
            )
        if self.algorithm == "linear_svc":
            return LinearSVC(
                C=c_value,
                class_weight=class_weight,
                max_iter=max_iter,
                random_state=self.random_state,
            )
        raise ValueError(f"Unsupported algorithm '{self.algorithm}'.")


def _evaluate_split(
    *,
    fold_index: int,
    split: HoldoutSplit,
    task_name: str,
    algorithm: str,
    random_state: int,
) -> tuple[FoldEvaluation, ClassificationTrainer]:
    trainer = ClassificationTrainer(
        task_name=task_name,
        algorithm=algorithm,
        random_state=random_state,
    )
    trainer.fit_train(split.train_df)

    target_column = trainer.get_target_column()
    X_test = trainer.preprocessor.transform(split.test_df)
    y_true = trainer.preprocessor.pipeline.target_encoder.transform(
        split.test_df[target_column].reset_index(drop=True)
    )
    y_pred = pd.Series(
        trainer.model.predict(X_test),
        name=f"{task_name}_prediction",
    )
    fold_evaluation = evaluate_fold(
        fold_index=fold_index,
        y_true=y_true,
        y_pred=y_pred,
        label_ids=trainer.get_label_order(),
        label_names=trainer.get_label_names(),
        languages=split.test_df.get("language"),
    )
    return fold_evaluation, trainer


def evaluate_task(
    task_name: str,
    folds: list[HoldoutSplit],
    random_state: int,
    algorithm: str,
) -> dict[str, Any]:
    fold_evaluations = []
    trainer_config: dict[str, Any] | None = None

    for fold_index, split in enumerate(folds, start=1):
        fold_evaluation, trainer = _evaluate_split(
            fold_index=fold_index,
            split=split,
            task_name=task_name,
            algorithm=algorithm,
            random_state=random_state,
        )
        fold_evaluations.append(fold_evaluation)

        if trainer_config is None:
            trainer_config = {
                "target_column": trainer.get_target_column(),
                "model": trainer.get_model_config(),
                "preprocessing": trainer.get_preprocessing_config(),
            }

    task_results = summarize_cv_results(fold_evaluations)
    task_results["task_config"] = trainer_config or {}
    return task_results


def fit_final_model(
    task_name: str,
    df: pd.DataFrame,
    random_state: int,
    algorithm: str,
) -> ClassificationTrainer:
    trainer = ClassificationTrainer(
        task_name=task_name,
        algorithm=algorithm,
        random_state=random_state,
    )
    trainer.fit_full(df)
    return trainer
