"""Regression training workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .preprocessing import ResolutionTimePreprocessor
from .training_utils import make_holdout_split


@dataclass
class RegressionTrainer:
    """Trainer for resolution-time regression."""

    random_state: int = 42
    test_size: float = 0.2
    model: RandomForestRegressor = field(init=False)
    preprocessor: ResolutionTimePreprocessor = field(init=False)
    feature_names_: list[str] = field(default_factory=list, init=False)
    split_metadata_: dict[str, int | float | str] = field(
        default_factory=dict, init=False
    )
    _test_X: csr_matrix | None = field(default=None, init=False, repr=False)
    _test_y: pd.Series | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.preprocessor = ResolutionTimePreprocessor()

    def fit(self, df: pd.DataFrame) -> RegressionTrainer:
        usable_df = df[df["resolution_time_hours"].notna()].reset_index(drop=True)
        split = make_holdout_split(
            frame=usable_df,
            target_column="resolution_time_hours",
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=False,
        )
        train_data = self.preprocessor.fit_transform(split.train_df)
        self.model.fit(train_data.X, train_data.y)

        self._test_X = self.preprocessor.transform(split.test_df)
        self._test_y = pd.to_numeric(
            split.test_df["resolution_time_hours"], errors="coerce"
        ).reset_index(drop=True)
        self.feature_names_ = train_data.feature_names
        self.split_metadata_ = {
            "task_name": "resolution_time_hours",
            "train_rows": len(split.train_df),
            "test_rows": len(split.test_df),
            "test_size": self.test_size,
            "random_state": self.random_state,
        }
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        X = self.preprocessor.transform(df)
        predictions = self.model.predict(X)
        return pd.Series(predictions, name="resolution_time_hours")

    def evaluate(self, df: pd.DataFrame) -> dict[str, float]:
        self.fit(df)
        predictions = self.model.predict(self._require_test_X())
        truth = self._require_test_y()
        mse = mean_squared_error(truth, predictions)
        return {
            "mae": float(mean_absolute_error(truth, predictions)),
            "rmse": float(sqrt(mse)),
            "r2": float(r2_score(truth, predictions)),
        }

    def _require_test_X(self) -> csr_matrix:
        if self._test_X is None:
            raise ValueError("testation features are unavailable. Fit the trainer first.")
        return self._test_X

    def _require_test_y(self) -> pd.Series:
        if self._test_y is None:
            raise ValueError("testation targets are unavailable. Fit the trainer first.")
        return self._test_y
