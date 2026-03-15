"""Shared helpers for model training workflows."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class HoldoutSplit:
    """Container for train/validation dataframes."""

    train_df: pd.DataFrame
    test_df: pd.DataFrame


def make_holdout_split(
    frame: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = False,
) -> HoldoutSplit:
    """Create a holdout split from a raw dataframe."""
    if target_column not in frame.columns:
        raise KeyError(f"Missing target column: {target_column}")

    stratify_values = frame[target_column] if stratify else None
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
