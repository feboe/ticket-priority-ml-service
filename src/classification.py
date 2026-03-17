"""Classification training workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from .evaluation import evaluate_fold, summarize_cv_results
from .preprocessing import PriorityPreprocessor, QueuePreprocessor
from .training_utils import HoldoutSplit, make_holdout_split


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
        "queue": 8.0,
        "priority": 4.0,
    },
}

MAX_ITER_BY_ALGORITHM = {
    "logreg": 2000,
    "linear_svc": 5000,
}


@dataclass
class ClassificationTrainer:
    """Reusable trainer for queue and priority classification."""

    task_name: str
    algorithm: str = "logreg"
    random_state: int = 42
    test_size: float = 0.2
    length_feature_enabled: bool = False
    model: LogisticRegression | LinearSVC = field(init=False)
    preprocessor: QueuePreprocessor | PriorityPreprocessor = field(init=False)
    feature_names_: list[str] = field(default_factory=list, init=False)
    target_mapping_: dict[int, str] = field(default_factory=dict, init=False)
    _test_X: csr_matrix | None = field(default=None, init=False, repr=False)
    _test_y: pd.Series | None = field(default=None, init=False, repr=False)
    _test_predictions: pd.Series | None = field(default=None, init=False, repr=False)

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
        self.preprocessor = PREPROCESSOR_FACTORY[self.task_name](
            length_feature_enabled=self.length_feature_enabled
        )

    def fit(self, df: pd.DataFrame) -> ClassificationTrainer:
        target_column = self.preprocessor.pipeline.target_column
        split = make_holdout_split(
            frame=df,
            target_column=target_column,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=True,
        )
        return self.fit_on_split(split)

    def fit_on_split(self, split: HoldoutSplit) -> ClassificationTrainer:
        target_column = self.preprocessor.pipeline.target_column
        train_data = self.preprocessor.fit_transform(split.train_df)
        self.model.fit(train_data.X, train_data.y)

        self._test_X = self.preprocessor.transform(split.test_df)
        self._test_y = self.preprocessor.pipeline.target_encoder.transform(
            split.test_df[target_column].reset_index(drop=True)
        )
        self._test_predictions = None
        self.feature_names_ = train_data.feature_names
        self.target_mapping_ = train_data.target_mapping or {}
        return self

    def fit_full(self, df: pd.DataFrame) -> ClassificationTrainer:
        train_data = self.preprocessor.fit_transform(df)
        self.model.fit(train_data.X, train_data.y)
        self.feature_names_ = train_data.feature_names
        self.target_mapping_ = train_data.target_mapping or {}
        self._test_X = None
        self._test_y = None
        self._test_predictions = None
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if not self.target_mapping_:
            raise ValueError("Target mapping is unavailable. Fit the trainer first.")
        X = self.preprocessor.transform(df)
        predictions = self.model.predict(X)
        decoded = pd.Series(predictions).map(
            lambda value: self.target_mapping_[int(value)]
        )
        return pd.Series(decoded, name=self.task_name)

    def get_test_truth(self) -> pd.Series:
        return self._require_test_y().copy()

    def get_test_predictions(self) -> pd.Series:
        return self._predict_current_split().copy()

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

    def _predict_current_split(self) -> pd.Series:
        if self._test_predictions is None:
            self._test_predictions = pd.Series(
                self.model.predict(self._require_test_X()),
                name=f"{self.task_name}_prediction",
            )
        return self._test_predictions

    def _require_test_X(self) -> csr_matrix:
        if self._test_X is None:
            raise ValueError("Test features are unavailable. Fit the trainer first.")
        return self._test_X

    def _require_test_y(self) -> pd.Series:
        if self._test_y is None:
            raise ValueError("Test targets are unavailable. Fit the trainer first.")
        return self._test_y


def evaluate_task(
    task_name: str,
    folds: list[HoldoutSplit],
    random_state: int,
    algorithm: str,
    length_feature_enabled: bool = True,
) -> dict[str, Any]:
    fold_evaluations = []
    trainer_config: dict[str, Any] | None = None

    for fold_index, split in enumerate(folds, start=1):
        trainer = ClassificationTrainer(
            task_name=task_name,
            algorithm=algorithm,
            random_state=random_state,
            length_feature_enabled=length_feature_enabled,
        )
        trainer.fit_on_split(split)

        fold_evaluations.append(
            evaluate_fold(
                fold_index=fold_index,
                y_true=trainer.get_test_truth(),
                y_pred=trainer.get_test_predictions(),
                label_ids=trainer.get_label_order(),
                label_names=trainer.get_label_names(),
            )
        )

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
    length_feature_enabled: bool = True,
) -> ClassificationTrainer:
    trainer = ClassificationTrainer(
        task_name=task_name,
        algorithm=algorithm,
        random_state=random_state,
        length_feature_enabled=length_feature_enabled,
    )
    trainer.fit_full(df)
    return trainer
