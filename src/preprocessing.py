"""TF-IDF preprocessing workflows for ticket models."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
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


class AuxiliaryFeatureExtractor(Protocol):
    """Interface for sparse auxiliary feature blocks appended after TF-IDF."""

    def fit_transform(self, frame: pd.DataFrame) -> csr_matrix: ...

    def transform(self, frame: pd.DataFrame) -> csr_matrix: ...

    def get_feature_names(self) -> list[str]: ...


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
class LengthFeatureExtractor:
    """Create a single numeric feature from cleaned message length."""

    length_column: str = "initial_message_length"
    feature_name: str = "initial_message_length"
    scale_: float = field(default=1.0, init=False)

    def fit_transform(self, frame: pd.DataFrame) -> csr_matrix:
        values = self._get_values(frame)
        self.scale_ = max(float(values.max()), 1.0)
        return csr_matrix((values / self.scale_).to_numpy().reshape(-1, 1))

    def transform(self, frame: pd.DataFrame) -> csr_matrix:
        values = self._get_values(frame)
        return csr_matrix((values / self.scale_).to_numpy().reshape(-1, 1))

    def get_feature_names(self) -> list[str]:
        return [self.feature_name]

    def _get_values(self, frame: pd.DataFrame) -> pd.Series:
        if self.length_column not in frame.columns:
            raise KeyError(f"Missing length column: {self.length_column}")
        return frame[self.length_column].astype(float)


@dataclass
class PlatformFeatureExtractor:
    """One-hot encode the platform column as sparse features."""

    platform_column: str = "platform"
    unknown_value: str = "unknown"
    categories_: list[str] = field(default_factory=list, init=False)

    def fit_transform(self, frame: pd.DataFrame) -> csr_matrix:
        values = self._prepare_values(frame)
        self.categories_ = sorted(set(values.tolist()) | {self.unknown_value})
        return self._to_sparse(values)

    def transform(self, frame: pd.DataFrame) -> csr_matrix:
        if not self.categories_:
            raise ValueError(
                "Platform categories are unavailable. Fit the extractor first."
            )
        values = self._prepare_values(frame)
        values = values.where(values.isin(self.categories_), self.unknown_value)
        return self._to_sparse(values)

    def get_feature_names(self) -> list[str]:
        return [f"platform_{category}" for category in self.categories_]

    def _prepare_values(self, frame: pd.DataFrame) -> pd.Series:
        if self.platform_column not in frame.columns:
            raise KeyError(f"Missing platform column: {self.platform_column}")
        return frame[self.platform_column].fillna(self.unknown_value).astype(str)

    def _to_sparse(self, values: pd.Series) -> csr_matrix:
        categorical = pd.Categorical(values, categories=self.categories_)
        encoded = pd.get_dummies(categorical)
        return csr_matrix(encoded.to_numpy(dtype=float))


@dataclass
class ChannelFeatureExtractor:
    """One-hot encode the channel column as sparse features."""

    channel_column: str = "channel"
    unknown_value: str = "unknown"
    categories_: list[str] = field(default_factory=list, init=False)

    def fit_transform(self, frame: pd.DataFrame) -> csr_matrix:
        values = self._prepare_values(frame)
        self.categories_ = sorted(set(values.tolist()) | {self.unknown_value})
        return self._to_sparse(values)

    def transform(self, frame: pd.DataFrame) -> csr_matrix:
        if not self.categories_:
            raise ValueError(
                "Channel categories are unavailable. Fit the extractor first."
            )
        values = self._prepare_values(frame)
        values = values.where(values.isin(self.categories_), self.unknown_value)
        return self._to_sparse(values)

    def get_feature_names(self) -> list[str]:
        return [f"channel_{category}" for category in self.categories_]

    def _prepare_values(self, frame: pd.DataFrame) -> pd.Series:
        if self.channel_column not in frame.columns:
            raise KeyError(f"Missing channel column: {self.channel_column}")
        return frame[self.channel_column].fillna(self.unknown_value).astype(str)

    def _to_sparse(self, values: pd.Series) -> csr_matrix:
        categorical = pd.Categorical(values, categories=self.categories_)
        encoded = pd.get_dummies(categorical)
        return csr_matrix(encoded.to_numpy(dtype=float))


@dataclass
class CustomerSegmentFeatureExtractor:
    """One-hot encode the customer segment column as sparse features."""

    segment_column: str = "customer_segment"
    unknown_value: str = "unknown"
    categories_: list[str] = field(default_factory=list, init=False)

    def fit_transform(self, frame: pd.DataFrame) -> csr_matrix:
        values = self._prepare_values(frame)
        self.categories_ = sorted(set(values.tolist()) | {self.unknown_value})
        return self._to_sparse(values)

    def transform(self, frame: pd.DataFrame) -> csr_matrix:
        if not self.categories_:
            raise ValueError(
                "Customer segment categories are unavailable. Fit the extractor first."
            )
        values = self._prepare_values(frame)
        values = values.where(values.isin(self.categories_), self.unknown_value)
        return self._to_sparse(values)

    def get_feature_names(self) -> list[str]:
        return [f"customer_segment_{category}" for category in self.categories_]

    def _prepare_values(self, frame: pd.DataFrame) -> pd.Series:
        if self.segment_column not in frame.columns:
            raise KeyError(f"Missing customer segment column: {self.segment_column}")
        return frame[self.segment_column].fillna(self.unknown_value).astype(str)

    def _to_sparse(self, values: pd.Series) -> csr_matrix:
        categorical = pd.Categorical(values, categories=self.categories_)
        encoded = pd.get_dummies(categorical)
        return csr_matrix(encoded.to_numpy(dtype=float))


@dataclass
class RegionFilledFeatureExtractor:
    """One-hot encode a filled region feature as sparse columns."""

    region_filled_column: str = "region_filled"
    region_column: str = "region"
    unknown_value: str = "unknown"
    categories_: list[str] = field(default_factory=list, init=False)

    def fit_transform(self, frame: pd.DataFrame) -> csr_matrix:
        values = self._prepare_values(frame)
        self.categories_ = sorted(set(values.tolist()) | {self.unknown_value})
        return self._to_sparse(values)

    def transform(self, frame: pd.DataFrame) -> csr_matrix:
        if not self.categories_:
            raise ValueError(
                "Region categories are unavailable. Fit the extractor first."
            )
        values = self._prepare_values(frame)
        values = values.where(values.isin(self.categories_), self.unknown_value)
        return self._to_sparse(values)

    def get_feature_names(self) -> list[str]:
        return [f"region_filled_{category}" for category in self.categories_]

    def _prepare_values(self, frame: pd.DataFrame) -> pd.Series:
        if self.region_filled_column in frame.columns:
            values = frame[self.region_filled_column]
        elif self.region_column in frame.columns:
            values = frame[self.region_column]
        else:
            raise KeyError(
                f"Missing region columns: {self.region_filled_column} or {self.region_column}"
            )
        return values.fillna(self.unknown_value).astype(str)

    def _to_sparse(self, values: pd.Series) -> csr_matrix:
        categorical = pd.Categorical(values, categories=self.categories_)
        encoded = pd.get_dummies(categorical)
        return csr_matrix(encoded.to_numpy(dtype=float))


@dataclass
class HasAttachmentFeatureExtractor:
    """Create a single numeric feature from the has_attachment column."""

    attachment_column: str = "has_attachment"
    feature_name: str = "has_attachment"

    def fit_transform(self, frame: pd.DataFrame) -> csr_matrix:
        return self._to_sparse(frame)

    def transform(self, frame: pd.DataFrame) -> csr_matrix:
        return self._to_sparse(frame)

    def get_feature_names(self) -> list[str]:
        return [self.feature_name]

    def _to_sparse(self, frame: pd.DataFrame) -> csr_matrix:
        if self.attachment_column not in frame.columns:
            raise KeyError(f"Missing attachment column: {self.attachment_column}")
        values = pd.to_numeric(frame[self.attachment_column], errors="coerce").fillna(0.0)
        return csr_matrix(values.to_numpy(dtype=float).reshape(-1, 1))


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
    auxiliary_extractors: list[AuxiliaryFeatureExtractor] = field(default_factory=list)
    target_encoder: TargetEncoder | None = None

    def fit_transform(self, frame: pd.DataFrame) -> VectorizedDataset:
        prepared = self._prepare_frame(frame)
        X = self.feature_extractor.fit_transform(
            prepared[self.text_pipeline.cleaned_column]
        )
        feature_names = self.feature_extractor.get_feature_names()
        for extractor in self.auxiliary_extractors:
            X_aux = extractor.fit_transform(prepared)
            X = hstack([X, X_aux], format="csr")
            feature_names = feature_names + extractor.get_feature_names()

        y = prepared[self.target_column].reset_index(drop=True)
        target_mapping = None
        if self.target_encoder is not None:
            y = self.target_encoder.fit_transform(y)
            target_mapping = self.target_encoder.get_mapping()
        return VectorizedDataset(
            X=X,
            y=y,
            frame=prepared.reset_index(drop=True),
            feature_names=feature_names,
            target_mapping=target_mapping,
        )

    def transform(self, frame: pd.DataFrame) -> csr_matrix:
        prepared = self.text_pipeline.transform(frame)
        X = self.feature_extractor.transform(prepared[self.text_pipeline.cleaned_column])
        for extractor in self.auxiliary_extractors:
            X_aux = extractor.transform(prepared)
            X = hstack([X, X_aux], format="csr")
        return X

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
            auxiliary_extractors=[
                LengthFeatureExtractor(),
            ],
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
            auxiliary_extractors=[
                LengthFeatureExtractor(),
                PlatformFeatureExtractor(),
                CustomerSegmentFeatureExtractor(),
                RegionFilledFeatureExtractor(),
            ],
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
            auxiliary_extractors=[
                LengthFeatureExtractor(),
            ],
        )
    )

    def fit_transform(self, frame: pd.DataFrame) -> VectorizedDataset:
        data = self.pipeline.fit_transform(frame)
        transformed_y = pd.Series(
            np.log1p(data.y.to_numpy(dtype=float)),
            index=data.y.index,
            name=data.y.name,
        )
        return VectorizedDataset(
            X=data.X,
            y=transformed_y,
            frame=data.frame,
            feature_names=data.feature_names,
            target_mapping=data.target_mapping,
        )

    def transform(self, frame: pd.DataFrame) -> csr_matrix:
        return self.pipeline.transform(frame)

    @staticmethod
    def inverse_transform_target(
        values: pd.Series | np.ndarray,
    ) -> pd.Series | np.ndarray:
        if isinstance(values, pd.Series):
            return pd.Series(
                np.expm1(values.to_numpy(dtype=float)),
                index=values.index,
                name=values.name,
            )
        return np.expm1(np.asarray(values, dtype=float))


RESOLUTION_TIME_BUCKET_LABELS = [
    "< 4h",
    "4-24h",
    "1-3 Tage",
    "> 3 Tage",
]


@dataclass
class ResolutionTimeBucketPreprocessor:
    """Preprocessing workflow for resolution time bucket classification."""

    pipeline: TfidfTargetPreprocessor = field(
        default_factory=lambda: TfidfTargetPreprocessor(
            target_column="resolution_time_bucket",
            auxiliary_extractors=[
                LengthFeatureExtractor(),
            ],
            target_encoder=TargetEncoder(),
        )
    )

    def fit_transform(self, frame: pd.DataFrame) -> VectorizedDataset:
        prepared = self.prepare_training_frame(frame)
        return self.pipeline.fit_transform(prepared)

    def transform(self, frame: pd.DataFrame) -> csr_matrix:
        return self.pipeline.transform(frame)

    def prepare_training_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        prepared = self._with_bucket_target(frame)
        return prepared[prepared["resolution_time_bucket"].notna()].copy()

    @staticmethod
    def _bucketize_resolution_time(values: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(values, errors="coerce")
        buckets = pd.Series(index=values.index, dtype="object")
        buckets.loc[numeric < 4] = "< 4h"
        buckets.loc[(numeric >= 4) & (numeric < 24)] = "4-24h"
        buckets.loc[(numeric >= 24) & (numeric <= 72)] = "1-3 Tage"
        buckets.loc[numeric > 72] = "> 3 Tage"
        return buckets

    def _with_bucket_target(self, frame: pd.DataFrame) -> pd.DataFrame:
        if "resolution_time_hours" not in frame.columns:
            raise KeyError("Missing target column: resolution_time_hours")
        df = frame.copy()
        df["resolution_time_bucket"] = self._bucketize_resolution_time(
            df["resolution_time_hours"]
        )
        return df
