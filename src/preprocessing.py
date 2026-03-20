"""Compact TF-IDF preprocessing for queue and priority classification."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache

import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords

TFIDF_MAX_FEATURES = None
TFIDF_MIN_DF = 1
TFIDF_MAX_DF = 0.95
TFIDF_NGRAM_RANGE = (1, 3)
TFIDF_ANALYZER = "word"
TFIDF_STOP_WORDS = None
TFIDF_SUBLINEAR_TF = False

STOP_WORD_LANGUAGE_BY_PREFIX = {
    "de": "german",
    "en": "english",
}

PRIORITY_CLASS_ORDER = ("low", "medium", "high")

DEFAULT_LENGTH_FEATURE_ENABLED = False
LENGTH_FEATURE_ENABLED_BY_TARGET = {
    "queue": False,
    "priority": False,
}


@lru_cache(maxsize=4)
def _load_nltk_stop_words(language: str) -> frozenset[str]:
    try:
        return frozenset(stopwords.words(language))
    except LookupError:
        return frozenset()


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
    """Combine subject and body into a normalized text field."""

    subject_column: str = "subject"
    body_column: str = "body"
    language_column: str = "language"
    combined_column: str = "ticket_text"
    cleaned_column: str = "ticket_text_clean"
    length_column: str = "ticket_text_length"

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.copy()
        if self.body_column not in df.columns:
            raise KeyError(f"Missing text column: {self.body_column}")

        subject = (
            df[self.subject_column].fillna("").astype(str)
            if self.subject_column in df.columns
            else pd.Series("", index=df.index, dtype="object")
        )
        body = df[self.body_column].fillna("").astype(str)
        languages = (
            df[self.language_column].fillna("unknown").astype(str).str.lower()
            if self.language_column in df.columns
            else pd.Series("unknown", index=df.index, dtype="object")
        )
        combined_text = (subject.str.strip() + " " + body.str.strip()).str.strip()

        df[self.combined_column] = combined_text
        cleaned_text = pd.Series(
            [
                self._normalize_text(text, language)
                for text, language in zip(combined_text, languages)
            ],
            index=df.index,
            dtype="object",
        )
        df[self.cleaned_column] = cleaned_text
        df[self.length_column] = cleaned_text.str.len()
        return df

    @staticmethod
    def _normalize_text(value: object, language: str = "unknown") -> str:
        text = unicodedata.normalize("NFKC", str(value or ""))
        text = text.lower()
        text = re.sub(r"https?://\S+|www\.\S+", " url ", text)
        text = re.sub(r"\b\S+@\S+\b", " email ", text)
        text = re.sub(r"\d+", " number ", text)
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"_+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        stop_words = TextPreparationPipeline._get_stop_words(language)
        if stop_words:
            text = " ".join(token for token in text.split() if token not in stop_words)
        return text

    @staticmethod
    def _get_stop_words(language: str) -> frozenset[str]:
        for prefix, nltk_language in STOP_WORD_LANGUAGE_BY_PREFIX.items():
            if language.startswith(prefix):
                return _load_nltk_stop_words(nltk_language)
        return frozenset()


@dataclass
class TfidfFeatureExtractor:
    """Wrap a configured TF-IDF vectorizer."""

    max_features: int | None = TFIDF_MAX_FEATURES
    min_df: int = TFIDF_MIN_DF
    max_df: float = TFIDF_MAX_DF
    ngram_range: tuple[int, int] = TFIDF_NGRAM_RANGE
    analyzer: str = TFIDF_ANALYZER
    stop_words: str | None = TFIDF_STOP_WORDS
    sublinear_tf: bool = TFIDF_SUBLINEAR_TF
    vectorizer: TfidfVectorizer = field(init=False)

    def __post_init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range,
            analyzer=self.analyzer,
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
    """Create a single numeric feature from cleaned ticket length."""

    length_column: str = "ticket_text_length"
    feature_name: str = "ticket_text_length"
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
class OrderedTargetEncoder:
    """Encode string targets with an explicit class order."""

    class_order: tuple[str, ...]

    def fit_transform(self, target: pd.Series) -> pd.Series:
        return self.transform(target)

    def transform(self, target: pd.Series) -> pd.Series:
        categories = pd.Categorical(target, categories=self.class_order)
        if categories.isna().any():
            unknown_labels = sorted(set(target[categories.isna()].astype(str)))
            raise ValueError(
                "Found labels outside the configured order: " + ", ".join(unknown_labels)
            )
        return pd.Series(categories.codes, index=target.index, name=target.name)

    def get_mapping(self) -> dict[int, str]:
        return {idx: label for idx, label in enumerate(self.class_order)}


@dataclass
class TfidfTargetPreprocessor:
    """Shared TF-IDF preprocessing logic for a single target."""

    target_column: str
    length_feature_enabled: bool = DEFAULT_LENGTH_FEATURE_ENABLED
    text_pipeline: TextPreparationPipeline = field(
        default_factory=TextPreparationPipeline
    )
    feature_extractor: TfidfFeatureExtractor = field(
        default_factory=TfidfFeatureExtractor
    )
    target_encoder: TargetEncoder | OrderedTargetEncoder | None = None
    length_extractor: LengthFeatureExtractor | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.length_feature_enabled:
            self.length_extractor = LengthFeatureExtractor()

    def fit_transform(self, frame: pd.DataFrame) -> VectorizedDataset:
        prepared = self._prepare_frame(frame)
        X_text = self.feature_extractor.fit_transform(
            prepared[self.text_pipeline.cleaned_column]
        )
        feature_names = self.feature_extractor.get_feature_names()
        if self.length_extractor is None:
            X = X_text
        else:
            X_length = self.length_extractor.fit_transform(prepared)
            X = hstack([X_text, X_length], format="csr")
            feature_names = feature_names + self.length_extractor.get_feature_names()

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
        X_text = self.feature_extractor.transform(
            prepared[self.text_pipeline.cleaned_column]
        )
        if self.length_extractor is None:
            return X_text
        X_length = self.length_extractor.transform(prepared)
        return hstack([X_text, X_length], format="csr")

    def _prepare_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        if self.target_column not in frame.columns:
            raise KeyError(f"Missing target column: {self.target_column}")

        df = frame.copy()
        df = df[df[self.target_column].notna()].copy()
        if df.empty:
            raise ValueError(
                f"No rows with non-null target values for '{self.target_column}'."
            )
        return self.text_pipeline.transform(df)


@dataclass
class QueuePreprocessor:
    """Preprocessing workflow for queue prediction."""

    pipeline: TfidfTargetPreprocessor = field(init=False)

    def __post_init__(self) -> None:
        self.pipeline = TfidfTargetPreprocessor(
            target_column="queue",
            length_feature_enabled=LENGTH_FEATURE_ENABLED_BY_TARGET["queue"],
            target_encoder=TargetEncoder(),
        )

    def fit_transform(self, frame: pd.DataFrame) -> VectorizedDataset:
        return self.pipeline.fit_transform(frame)

    def transform(self, frame: pd.DataFrame) -> csr_matrix:
        return self.pipeline.transform(frame)


@dataclass
class PriorityPreprocessor:
    """Preprocessing workflow for priority prediction."""

    pipeline: TfidfTargetPreprocessor = field(init=False)

    def __post_init__(self) -> None:
        self.pipeline = TfidfTargetPreprocessor(
            target_column="priority",
            length_feature_enabled=LENGTH_FEATURE_ENABLED_BY_TARGET["priority"],
            target_encoder=OrderedTargetEncoder(PRIORITY_CLASS_ORDER),
        )

    def fit_transform(self, frame: pd.DataFrame) -> VectorizedDataset:
        return self.pipeline.fit_transform(frame)

    def transform(self, frame: pd.DataFrame) -> csr_matrix:
        return self.pipeline.transform(frame)
