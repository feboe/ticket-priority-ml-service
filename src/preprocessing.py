"""TF-IDF preprocessing workflows for ticket models."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


@dataclass
class VectorizedDataset:
    """Container returned by each preprocessor after vectorization."""

    X: csr_matrix
    y: pd.Series
    frame: pd.DataFrame
    feature_names: list[str]
    target_mapping: dict[int, str] | None = None


@dataclass
class TextPreparationPipeline:
    """Prepare raw ticket text for TF-IDF vectorization."""

    text_column: str = "initial_message"
    cleaned_column: str = "initial_message_clean"
    length_column: str = "initial_message_length"

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.copy()

        if self.text_column not in df.columns:
            raise KeyError(f"Missing text column: {self.text_column}")

        cleaned_text = df[self.text_column].fillna("").map(self._normalize_text)
        df[self.cleaned_column] = cleaned_text
        df[self.length_column] = cleaned_text.str.len()
        return df

    @staticmethod
    def _normalize_text(value: object) -> str:
        text = unicodedata.normalize("NFKC", str(value or ""))
        text = text.lower()
        text = re.sub(r"https?://\S+|www\.\S+", " url ", text)
        text = re.sub(r"\b\S+@\S+\b", " email ", text)
        text = re.sub(r"\d+", " number ", text)
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"_+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


@dataclass
class TfidfFeatureExtractor:
    """Wrap a configured TF-IDF vectorizer."""

    max_features: int = 5000
    min_df: int = 2
    max_df: float = 0.95
    ngram_range: tuple[int, int] = (1, 2)
    stop_words: str | None = "english"
    sublinear_tf: bool = True
    vectorizer: TfidfVectorizer = field(init=False)

    def __post_init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range,
            stop_words=self.stop_words,
            sublinear_tf=self.sublinear_tf,
        )

    def fit_transform(self, text: pd.Series) -> csr_matrix:
        return self.vectorizer.fit_transform(text)

    def transform(self, text: pd.Series) -> csr_matrix:
        return self.vectorizer.transform(text)

    def get_feature_names(self) -> list[str]:
        return self.vectorizer.get_feature_names_out().tolist()


@dataclass
class TargetEncoder:
    """Encode string targets into integer class ids."""

    encoder: LabelEncoder = field(init=False)

    def __post_init__(self) -> None:
        self.encoder = LabelEncoder()

    def fit_transform(self, target: pd.Series) -> pd.Series:
        encoded = self.encoder.fit_transform(target)
        return pd.Series(encoded, index=target.index, name=target.name)

    def transform(self, target: pd.Series) -> pd.Series:
        encoded = self.encoder.transform(target)
        return pd.Series(encoded, index=target.index, name=target.name)

    def get_mapping(self) -> dict[int, str]:
        return {idx: label for idx, label in enumerate(self.encoder.classes_)}


@dataclass
class TfidfTargetPreprocessor:
    """Shared TF-IDF preprocessing logic for a single target."""

    target_column: str
    numeric_target: bool = False
    text_pipeline: TextPreparationPipeline = field(
        default_factory=TextPreparationPipeline
    )
    feature_extractor: TfidfFeatureExtractor = field(
        default_factory=TfidfFeatureExtractor
    )
    target_encoder: TargetEncoder | None = None

    def fit_transform(self, frame: pd.DataFrame) -> VectorizedDataset:
        prepared = self._prepare_frame(frame)
        X = self.feature_extractor.fit_transform(
            prepared[self.text_pipeline.cleaned_column]
        )
        y = prepared[self.target_column].reset_index(drop=True)
        target_mapping = None
        if self.target_encoder is not None:
            y = self.target_encoder.fit_transform(y)
            target_mapping = self.target_encoder.get_mapping()
        return VectorizedDataset(
            X=X,
            y=y,
            frame=prepared.reset_index(drop=True),
            feature_names=self.feature_extractor.get_feature_names(),
            target_mapping=target_mapping,
        )

    def transform(self, frame: pd.DataFrame) -> csr_matrix:
        prepared = self.text_pipeline.transform(frame)
        return self.feature_extractor.transform(
            prepared[self.text_pipeline.cleaned_column]
        )

    def _prepare_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        if self.target_column not in frame.columns:
            raise KeyError(f"Missing target column: {self.target_column}")

        df = frame.copy()
        if self.numeric_target:
            df[self.target_column] = pd.to_numeric(
                df[self.target_column], errors="coerce"
            )

        df = df[df[self.target_column].notna()].copy()
        if df.empty:
            raise ValueError(
                f"No rows with non-null target values for '{self.target_column}'."
            )

        return self.text_pipeline.transform(df)


@dataclass
class ProductAreaPreprocessor:
    """Preprocessing workflow for product area prediction."""

    pipeline: TfidfTargetPreprocessor = field(
        default_factory=lambda: TfidfTargetPreprocessor(
            target_column="product_area",
            target_encoder=TargetEncoder(),
        )
    )

    def fit_transform(self, frame: pd.DataFrame) -> VectorizedDataset:
        return self.pipeline.fit_transform(frame)

    def transform(self, frame: pd.DataFrame) -> csr_matrix:
        return self.pipeline.transform(frame)


@dataclass
class PriorityPreprocessor:
    """Preprocessing workflow for priority prediction."""

    pipeline: TfidfTargetPreprocessor = field(
        default_factory=lambda: TfidfTargetPreprocessor(
            target_column="priority",
            target_encoder=TargetEncoder(),
        )
    )

    def fit_transform(self, frame: pd.DataFrame) -> VectorizedDataset:
        return self.pipeline.fit_transform(frame)

    def transform(self, frame: pd.DataFrame) -> csr_matrix:
        return self.pipeline.transform(frame)


@dataclass
class ResolutionTimePreprocessor:
    """Preprocessing workflow for resolution time regression."""

    pipeline: TfidfTargetPreprocessor = field(
        default_factory=lambda: TfidfTargetPreprocessor(
            target_column="resolution_time_hours",
            numeric_target=True,
        )
    )

    def fit_transform(self, frame: pd.DataFrame) -> VectorizedDataset:
        return self.pipeline.fit_transform(frame)

    def transform(self, frame: pd.DataFrame) -> csr_matrix:
        return self.pipeline.transform(frame)
