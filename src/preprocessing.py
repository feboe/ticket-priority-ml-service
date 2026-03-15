"""Compact TF-IDF preprocessing for queue and priority classification."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Protocol

import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


EN_STOP_WORDS = frozenset(ENGLISH_STOP_WORDS)
GERMAN_STOP_WORDS = frozenset(
    {
        "aber", "alle", "allem", "allen", "aller", "alles", "als", "also", "am", "an",
        "ander", "andere", "anderem", "anderen", "anderer", "anderes", "anderm", "andern",
        "anderr", "anders", "auch", "auf", "aus", "bei", "bin", "bis", "bist", "da",
        "damit", "dann", "das", "dass", "da?", "dein", "deine", "dem", "den", "der",
        "des", "dess", "deshalb", "die", "dies", "dieser", "dieses", "doch", "dort",
        "du", "durch", "ein", "eine", "einem", "einen", "einer", "eines", "er", "es",
        "euer", "eure", "f?r", "hatte", "hatten", "hattest", "hattet", "hier", "hinter",
        "ich", "ihr", "ihre", "im", "in", "ist", "ja", "jede", "jedem", "jeden", "jeder",
        "jedes", "jener", "jenes", "jetzt", "kann", "kannst", "k?nnen", "k?nnt", "machen",
        "mein", "meine", "mit", "mu?", "mu?t", "musst", "m?ssen", "m??t", "nach", "nachdem",
        "nein", "nicht", "nun", "oder", "seid", "sein", "seine", "sich", "sie", "sind",
        "soll", "sollen", "sollst", "sollt", "sonst", "soweit", "sowie", "und", "unser",
        "unsere", "unter", "vom", "von", "vor", "wann", "warum", "was", "weiter", "weitere",
        "wenn", "wer", "werde", "werden", "werdet", "weshalb", "wie", "wieder", "wieso",
        "wir", "wird", "wirst", "wo", "woher", "wohin", "zu", "zum", "zur", "?ber"
    }
)


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
        language = (
            df[self.language_column].fillna("unknown").astype(str).str.lower()
            if self.language_column in df.columns
            else pd.Series("unknown", index=df.index, dtype="object")
        )
        combined_text = (subject.str.strip() + " " + body.str.strip()).str.strip()

        df[self.combined_column] = combined_text
        cleaned_text = pd.Series(
            [self._normalize_text(text, lang) for text, lang in zip(combined_text, language)],
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
        if not stop_words:
            return text
        tokens = [token for token in text.split() if token not in stop_words]
        return " ".join(tokens)

    @staticmethod
    def _get_stop_words(language: str) -> frozenset[str]:
        if language.startswith("de"):
            return GERMAN_STOP_WORDS
        if language.startswith("en"):
            return EN_STOP_WORDS
        return frozenset()


@dataclass
class TfidfFeatureExtractor:
    """Wrap a configured TF-IDF vectorizer."""

    max_features: int = 20000
    min_df: int = 2
    max_df: float = 0.95
    ngram_range: tuple[int, int] = (1, 2)
    stop_words: str | None = None
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
class CategoricalFeatureExtractor:
    """One-hot encode an arbitrary categorical column as sparse features."""

    column_name: str
    feature_prefix: str
    unknown_value: str = "unknown"
    required: bool = False
    categories_: list[str] = field(default_factory=list, init=False)

    def fit_transform(self, frame: pd.DataFrame) -> csr_matrix:
        values = self._prepare_values(frame)
        if values is None:
            return self._empty_matrix(len(frame))
        self.categories_ = sorted(set(values.tolist()) | {self.unknown_value})
        return self._to_sparse(values)

    def transform(self, frame: pd.DataFrame) -> csr_matrix:
        values = self._prepare_values(frame)
        if values is None:
            return self._empty_matrix(len(frame))
        if not self.categories_:
            raise ValueError(
                f"Categories are unavailable for column '{self.column_name}'. Fit the extractor first."
            )
        values = values.where(values.isin(self.categories_), self.unknown_value)
        return self._to_sparse(values)

    def get_feature_names(self) -> list[str]:
        return [f"{self.feature_prefix}_{category}" for category in self.categories_]

    def _prepare_values(self, frame: pd.DataFrame) -> pd.Series | None:
        if self.column_name not in frame.columns:
            if self.required:
                raise KeyError(f"Missing categorical column: {self.column_name}")
            return None
        return frame[self.column_name].fillna(self.unknown_value).astype(str)

    def _to_sparse(self, values: pd.Series) -> csr_matrix:
        categorical = pd.Categorical(values, categories=self.categories_)
        encoded = pd.get_dummies(categorical)
        return csr_matrix(encoded.to_numpy(dtype=float))

    @staticmethod
    def _empty_matrix(num_rows: int) -> csr_matrix:
        return csr_matrix((num_rows, 0), dtype=float)


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
                "Found labels outside the configured order: "
                + ", ".join(unknown_labels)
            )
        return pd.Series(categories.codes, index=target.index, name=target.name)

    def get_mapping(self) -> dict[int, str]:
        return {idx: label for idx, label in enumerate(self.class_order)}


@dataclass
class TfidfTargetPreprocessor:
    """Shared TF-IDF preprocessing logic for a single target."""

    target_column: str
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
        df = df[df[self.target_column].notna()].copy()
        if df.empty:
            raise ValueError(
                f"No rows with non-null target values for '{self.target_column}'."
            )
        return self.text_pipeline.transform(df)


@dataclass
class QueuePreprocessor:
    """Preprocessing workflow for queue prediction."""

    pipeline: TfidfTargetPreprocessor = field(
        default_factory=lambda: TfidfTargetPreprocessor(
            target_column="queue",
            auxiliary_extractors=[
                LengthFeatureExtractor(),
                CategoricalFeatureExtractor(
                    column_name="language",
                    feature_prefix="language",
                    required=False,
                ),
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
                CategoricalFeatureExtractor(
                    column_name="language",
                    feature_prefix="language",
                    required=False,
                ),
            ],
            target_encoder=OrderedTargetEncoder(("low", "medium", "high")),
        )
    )

    def fit_transform(self, frame: pd.DataFrame) -> VectorizedDataset:
        return self.pipeline.fit_transform(frame)

    def transform(self, frame: pd.DataFrame) -> csr_matrix:
        return self.pipeline.transform(frame)
