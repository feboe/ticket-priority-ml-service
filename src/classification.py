"""Classification training workflows."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from .preprocessing import PriorityPreprocessor, ProductAreaPreprocessor
from .training_utils import make_holdout_split


PREPROCESSOR_FACTORY = {
    "product_area": ProductAreaPreprocessor,
    "priority": PriorityPreprocessor,
}

CLASS_WEIGHT_BY_TASK = {
    "product_area": None,
    "priority": "balanced",
}


@dataclass
class ClassificationTrainer:
    """Reusable trainer for text classification tasks."""

    task_name: str
    random_state: int = 42
    test_size: float = 0.2
    model: LogisticRegression = field(init=False)
    preprocessor: ProductAreaPreprocessor | PriorityPreprocessor = field(init=False)
    feature_names_: list[str] = field(default_factory=list, init=False)
    target_mapping_: dict[int, str] = field(default_factory=dict, init=False)
    split_metadata_: dict[str, int | float | str] = field(
        default_factory=dict, init=False
    )
    _test_X: csr_matrix | None = field(default=None, init=False, repr=False)
    _test_y: pd.Series | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.task_name not in PREPROCESSOR_FACTORY:
            raise ValueError(
                f"Unsupported classification task '{self.task_name}'. "
                f"Choose one of: {', '.join(PREPROCESSOR_FACTORY)}."
            )

        self.model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=self.random_state,
            class_weight=CLASS_WEIGHT_BY_TASK[self.task_name],
        )
        self.preprocessor = PREPROCESSOR_FACTORY[self.task_name]()

    def fit(self, df: pd.DataFrame) -> ClassificationTrainer:
        split = make_holdout_split(
            frame=df,
            target_column=self.task_name,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=True,
        )
        train_data = self.preprocessor.fit_transform(split.train_df)
        self.model.fit(train_data.X, train_data.y)

        self._test_X = self.preprocessor.transform(split.test_df)
        self._test_y = self.preprocessor.pipeline.target_encoder.transform(
            split.test_df[self.task_name].reset_index(drop=True)
        )
        self.feature_names_ = train_data.feature_names
        self.target_mapping_ = train_data.target_mapping or {}
        self.split_metadata_ = {
            "task_name": self.task_name,
            "train_rows": len(split.train_df),
            "test_rows": len(split.test_df),
            "test_size": self.test_size,
            "random_state": self.random_state,
            "class_weight": CLASS_WEIGHT_BY_TASK[self.task_name],
        }
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        encoded_predictions = self._predict_encoded(df)
        return self._decode_predictions(encoded_predictions)

    def evaluate(self, df: pd.DataFrame) -> dict[str, float]:
        self.fit(df)
        predictions = self.model.predict(self._require_test_X())
        truth = self._require_test_y()
        return {
            "accuracy": float(accuracy_score(truth, predictions)),
            "macro_f1": float(f1_score(truth, predictions, average="macro")),
        }

    def _predict_encoded(self, df: pd.DataFrame) -> pd.Series:
        X = self.preprocessor.transform(df)
        predictions = self.model.predict(X)
        return pd.Series(predictions, name=self.task_name)

    def _decode_predictions(self, predictions: pd.Series) -> pd.Series:
        if not self.target_mapping_:
            raise ValueError("Target mapping is unavailable. Fit the trainer first.")
        decoded = predictions.map(lambda value: self.target_mapping_[int(value)])
        return pd.Series(decoded, name=self.task_name)

    def _require_test_X(self) -> csr_matrix:
        if self._test_X is None:
            raise ValueError("testation features are unavailable. Fit the trainer first.")
        return self._test_X

    def _require_test_y(self) -> pd.Series:
        if self._test_y is None:
            raise ValueError("testation targets are unavailable. Fit the trainer first.")
        return self._test_y
