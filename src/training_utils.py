"""Shared helpers for model training workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


@dataclass
class HoldoutSplit:
    """Container for train/validation dataframes."""

    train_df: pd.DataFrame
    test_df: pd.DataFrame


def make_stratification_labels(
    frame: pd.DataFrame,
    target_columns: str | Sequence[str],
) -> pd.Series:
    """Build a single stratification label from one or more target columns."""
    if isinstance(target_columns, str):
        columns = (target_columns,)
    else:
        columns = tuple(target_columns)

    if not columns:
        raise ValueError("At least one target column is required for stratification.")

    missing_columns = [column for column in columns if column not in frame.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise KeyError(f"Missing target columns for stratification: {missing_text}")

    labels = frame[columns[0]].fillna("__missing__").astype(str)
    for column in columns[1:]:
        next_labels = frame[column].fillna("__missing__").astype(str)
        labels = labels.str.cat(next_labels, sep="__|__")
    return labels


def make_holdout_split(
    frame: pd.DataFrame,
    target_column: str | Sequence[str],
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = False,
) -> HoldoutSplit:
    """Create a holdout split from a raw dataframe."""
    stratify_values = (
        make_stratification_labels(frame, target_column) if stratify else None
    )
    train_df, valid_df = train_test_split(
        frame,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_values,
    )
    return HoldoutSplit(
        train_df=train_df.reset_index(drop=True),
        test_df=valid_df.reset_index(drop=True),
    )


def make_stratified_folds(
    frame: pd.DataFrame,
    target_columns: str | Sequence[str],
    n_splits: int = 5,
    random_state: int = 42,
    shuffle: bool = True,
) -> list[HoldoutSplit]:
    """Create shared stratified folds from one or more target columns."""
    stratify_labels = make_stratification_labels(frame, target_columns)
    min_class_count = int(stratify_labels.value_counts().min())
    if min_class_count < n_splits:
        raise ValueError(
            "Cannot build stratified folds because at least one class has fewer "
            f"than {n_splits} samples."
        )

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )

    folds: list[HoldoutSplit] = []
    for train_idx, test_idx in splitter.split(frame, stratify_labels):
        folds.append(
            HoldoutSplit(
                train_df=frame.iloc[train_idx].reset_index(drop=True),
                test_df=frame.iloc[test_idx].reset_index(drop=True),
            )
        )
    return folds
