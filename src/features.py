"""
Text-centric feature extraction for hate-speech classification.

This module focuses on the text-only feature groups outlined in the
project plan. It provides a light-weight API that can be reused across
training scripts or notebooks.  The implementation is intentionally
transparent and pragmatic—something a strong master's student might
write for a course project.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import spacy
    from spacy.language import Language
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "spaCy is required for POS features. Install it via `pip install spacy` "
        "and download the English model with `python -m spacy download en_core_web_sm`."
    ) from exc

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "sentence-transformers is required for semantic embeddings. "
        "Install it via `pip install sentence-transformers`."
    ) from exc

try:
    from textblob import TextBlob
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "TextBlob is required for sentiment features. "
        "Install it via `pip install textblob`."
    ) from exc

try:
    from textstat import textstat
except ImportError:  # pragma: no cover - optional dependency
    textstat = None


LOGGER = logging.getLogger(__name__)

URL_PATTERN = re.compile(
    r"(https?://|www\.)\S+", flags=re.IGNORECASE
)
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "]+",
    flags=re.UNICODE,
)
NON_ALPHANUMERIC_PATTERN = re.compile(r"[^a-z0-9\s]")
MULTISPACE_PATTERN = re.compile(r"\s+")

SOURCE_VOCAB = ("twitter", "reddit", "4chan")


def load_dataset(
    path: str | Path,
    text_column: str = "text",
    label_column: str = "label",
) -> pd.DataFrame:
    """
    Load a CSV or XLSX dataset that contains text and label columns.

    Args:
        path: Path to the dataset file.
        text_column: Name of the text column in the raw file.
        label_column: Name of the label column in the raw file.

    Returns:
        DataFrame with two columns: ``text`` and ``label``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() in {".csv", ".tsv"}:
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:  # pragma: no cover - defensive branch
        raise ValueError(
            f"Unsupported file extension {path.suffix!r}. Expected CSV or XLSX."
        )

    missing = {text_column, label_column} - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {', '.join(sorted(missing))}"
        )

    result = pd.DataFrame(
        {
            "text": df[text_column].fillna("").astype(str),
            "label": df[label_column],
        }
    )
    return result


def clean_text(text: str) -> str:
    """
    Basic normalisation pipeline matching the project requirements.
    """
    text = text.lower()
    text = URL_PATTERN.sub(" ", text)
    text = EMOJI_PATTERN.sub(" ", text)
    text = NON_ALPHANUMERIC_PATTERN.sub(" ", text)
    text = MULTISPACE_PATTERN.sub(" ", text).strip()
    return text


def get_hate_lexicon() -> List[str]:
    """
    Load hate lexicon from external dataset file (Kaggle/HuggingFace).
    REQUIRES: data/hate_lexicon.txt must exist.
    Run scripts/download_hate_lexicon.py to generate it.
    """
    lexicon_path = Path("data/hate_lexicon.txt")
    
    if not lexicon_path.exists():
        raise FileNotFoundError(
            f"Hate lexicon not found at {lexicon_path}\n"
            f"Please run: python scripts/download_hate_lexicon.py"
        )
    
    try:
        with open(lexicon_path, 'r', encoding='utf-8') as f:
            terms = [line.strip().lower() for line in f if line.strip()]
        LOGGER.info(f"Loaded {len(terms)} hate terms from external lexicon: {lexicon_path}")
        return terms
    except Exception as e:
        raise RuntimeError(f"Failed to load hate lexicon from {lexicon_path}: {e}")



def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Safe cosine similarity that returns 0.0 for zero vectors.
    """
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - _cosine_similarity(a, b)


def _estimate_syllables(word: str) -> int:
    """
    Naive syllable estimator used only when textstat is unavailable.
    """
    word = word.lower()
    word = re.sub(r"[^a-z]", "", word)
    if not word:
        return 0
    vowels = "aeiouy"
    count = 0
    prev_char_is_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_char_is_vowel:
            count += 1
        prev_char_is_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def _language_complexity(text: str) -> float:
    """
    Flesch-Kincaid readability score with a fallback approximation.
    """
    if textstat is not None:
        try:
            return float(textstat.flesch_kincaid_grade(text))
        except Exception:  # pragma: no cover - defensive
            LOGGER.debug("textstat failed to compute FK grade for text.")

    sentences = max(1, len(re.findall(r"[.!?]", text)) or 1)
    words = clean_text(text).split()
    num_words = len(words) or 1
    syllables = sum(_estimate_syllables(word) for word in words) or 1
    return 0.39 * (num_words / sentences) + 11.8 * (syllables / num_words) - 15.59


@dataclass
class TextFeatureExtractor:
    """
    Encapsulates stateful components such as vectorisers and language models.
    """

    spacy_model: str = "en_core_web_sm"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_tfidf_features: int = 10000
    tfidf_selection_method: str = "random_forest"
    tfidf_n_selected: int = 1000
    batch_size: int = 64
    random_state: int = 42

    def __post_init__(self) -> None:
        self.hate_lexicon: List[str] = get_hate_lexicon()
        self.hate_lexicon_set = {token for token in self.hate_lexicon}

        try:
            self.nlp: Language = spacy.load(self.spacy_model, disable=["ner"])
        except OSError as exc:  # pragma: no cover - environment guard
            raise OSError(
                f"spaCy model '{self.spacy_model}' is not installed. "
                "Install it with `python -m spacy download en_core_web_sm`."
            ) from exc

        self.embedding_model = SentenceTransformer(
            self.embedding_model_name, device="cpu"
        )
        self.lexicon_embedding = self.embedding_model.encode(
            self.hate_lexicon,
            show_progress_bar=False,
            batch_size=min(len(self.hate_lexicon), self.batch_size),
            convert_to_numpy=True,
        ).mean(axis=0)
        
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self._is_fitted = False
        self.tfidf_selector = None
        self.selected_tfidf_feature_names = None
    def fit(
        self,
        df: pd.DataFrame,
        source_name: str,
        text_column: str = "text",
        label_column: str = "label",
    ) -> "TextFeatureExtractor":
        """
        Fit the feature extractor on training data.
        
        This method learns parameters from the training data (primarily the TF-IDF
        vocabulary) that will be used when transforming both training and test data.
        
        Args:
            df: Training DataFrame
            source_name: Name of the dataset source (e.g., 'reddit', '4chan', 'twitter')
            text_column: Name of the text column
            label_column: Name of the label column
            
        Returns:
            self (for method chaining)
        """
        if text_column not in df.columns or label_column not in df.columns:
            raise ValueError(
                f"DataFrame must contain '{text_column}' and '{label_column}' columns."
            )
        
        print(f"Fitting feature extractor on {source_name} training data...")
        
        # Prepare text
        frame = df[[text_column, label_column]].copy()
        frame[text_column] = frame[text_column].fillna("").astype(str)
        raw_text = frame[text_column]
        cleaned_text = raw_text.apply(clean_text)
        
        print(f"  Fitting TF-IDF vectorizer (max_features={self.max_tfidf_features})...")
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=self.max_tfidf_features,
            sublinear_tf=True,
        )
        self.tfidf_vectorizer.fit(cleaned_text.tolist())
        print(f"  TF-IDF vocabulary size: {len(self.tfidf_vectorizer.vocabulary_)}")
        
        self._is_fitted = True
        return self
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        source_name: str,
        text_column: str = "text",
        label_column: str = "label",
    ) -> pd.DataFrame:
        """
        Fit the extractor and transform the data in one call.
        
        This is a convenience method equivalent to calling fit() then transform().
        Use this for training data, then use transform() for test data.
        
        Args:
            df: Training DataFrame
            source_name: Name of the dataset source
            text_column: Name of the text column
            label_column: Name of the label column
            
        Returns:
            DataFrame with extracted features
        """
        self.fit(df, source_name, text_column, label_column)
        features = self.transform(df, source_name, text_column, label_column)
        
        if self.tfidf_selection_method != "none":
            tfidf_cols = [col for col in features.columns if col.startswith('tfidf_')]
            if tfidf_cols:
                tfidf_df = features[tfidf_cols]
                labels = features[label_column].values
                
                if self.tfidf_selection_method == "random_forest":
                    selected_tfidf = self._select_tfidf_features_rf(
                        tfidf_df, 
                        labels, 
                        n_select=self.tfidf_n_selected
                    )
                else:
                    LOGGER.warning(f"Unknown selection method: {self.tfidf_selection_method}. Using all features.")
                    selected_tfidf = tfidf_df
                
                features = features.drop(columns=tfidf_cols)
                features = pd.concat([features, selected_tfidf], axis=1)
        
        return features
    
    def transform(
        self,
        df: pd.DataFrame,
        source_name: str,
        text_column: str = "text",
        label_column: str = "label",
    ) -> pd.DataFrame:
        """
        Extract all text-based feature groups for the provided dataframe.
        
        The extractor must be fitted before calling transform. Use fit() on training
        data first, then transform() on both training and test data.
        
        Args:
            df: DataFrame to transform
            source_name: Name of the dataset source
            text_column: Name of the text column
            label_column: Name of the label column
            
        Returns:
            DataFrame with extracted features
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Feature extractor must be fitted before transform. "
                "Call fit() on training data first."
            )
        
        if text_column not in df.columns or label_column not in df.columns:
            raise ValueError(
                f"DataFrame must contain '{text_column}' and '{label_column}' columns."
            )

        frame = df[[text_column, label_column]].copy()
        frame[text_column] = frame[text_column].fillna("").astype(str)

        raw_text = frame[text_column]
        cleaned_text = raw_text.apply(clean_text)
        tokens = cleaned_text.str.split()

        lexical = self._lexical_features(raw_text, cleaned_text, tokens)
        sentiment = self._sentiment_features(raw_text)
        hate_lexicon = self._hate_lexicon_features(tokens, raw_text)
        
        tfidf_full = self._tfidf_features(cleaned_text)
        if self.tfidf_selector is not None and self.selected_tfidf_feature_names:
            tfidf = tfidf_full[self.selected_tfidf_feature_names]
        else:
            tfidf = tfidf_full
        
        embeddings = self._sentence_embeddings(cleaned_text)
        semantic = self._semantic_features(tokens, embeddings)
        experimental = self._experimental_features(
            raw_text, tokens, embeddings, frame[label_column].to_numpy()
        )

        features = pd.concat(
            [lexical, sentiment, semantic, hate_lexicon, experimental, tfidf],
            axis=1,
        )
        features[label_column] = frame[label_column].values
        return features

    def _lexical_features(
        self,
        raw_text: pd.Series,
        cleaned_text: pd.Series,
        tokens: pd.Series,
    ) -> pd.DataFrame:
        """
        Lexical and simple textual statistics.
        
        Returns 14 features:
        - num_words, avg_word_len
        - uppercase_ratio, exclamation_ratio, question_ratio, period_ratio
        - punctuation_count, multiple_count
        - char_count, repeated_chars
        - unique_words, word_repetition
        - longest_word_length, shortest_word_length
        - language_complexity_index
        """
        data = pd.DataFrame(index=raw_text.index)

        num_words = tokens.apply(len)
        avg_word_len = tokens.apply(
            lambda toks: float(np.mean([len(t) for t in toks])) if toks else 0.0
        )
        alphabetic_counts = raw_text.apply(
            lambda text: sum(char.isalpha() for char in text)
        )
        uppercase_counts = raw_text.apply(
            lambda text: sum(char.isupper() for char in text)
        )
        exclam_counts = raw_text.str.count("!")
        question_counts = raw_text.str.count(r"\?")
        period_counts = raw_text.str.count(r"\.")
        char_count = raw_text.str.len()

        # Existing features
        data["num_words"] = num_words
        data["avg_word_len"] = avg_word_len
        data["uppercase_ratio"] = np.divide(
            uppercase_counts,
            alphabetic_counts.replace(0, np.nan),
        ).fillna(0.0)
        data["exclamation_ratio"] = np.divide(
            exclam_counts,
            char_count.replace(0, np.nan),
        ).fillna(0.0)
        
        data["question_ratio"] = np.divide(
            question_counts,
            char_count.replace(0, np.nan),
        ).fillna(0.0)
        
        data["period_ratio"] = np.divide(
            period_counts,
            char_count.replace(0, np.nan),
        ).fillna(0.0)
        
        data["punctuation_count"] = raw_text.str.count(r'[!?.]')
        
        data["multiple_count"] = raw_text.str.count(r'[!?.]{2,}')
        
        data["char_count"] = char_count
        
        data["repeated_chars"] = raw_text.str.count(r'(.)\1{2,}')
        
        data["unique_words"] = tokens.apply(lambda toks: len(set(toks)) if toks else 0)
        
        data["word_repetition"] = num_words - data["unique_words"]
        
        data["longest_word_length"] = tokens.apply(
            lambda toks: max([len(t) for t in toks]) if toks else 0
        )
        
        data["shortest_word_length"] = tokens.apply(
            lambda toks: min([len(t) for t in toks]) if toks else 0
        )

        return data

    def _hate_lexicon_features(
        self,
        tokens: pd.Series,
        raw_text: pd.Series,
    ) -> pd.DataFrame:
        """
        Hate speech lexicon matching features (independent group).
        """
        data = pd.DataFrame(index=tokens.index)
        
        data["hate_word_count"] = tokens.apply(
            lambda toks: sum(token in self.hate_lexicon_set for token in toks)
        )
        
        data["hate_word_ratio"] = tokens.apply(
            lambda toks: sum(token in self.hate_lexicon_set for token in toks) / len(toks) if toks else 0.0
        )
        
        data["contains_hate_word"] = (data["hate_word_count"] > 0).astype(int)
        
        return data
    
    def _sentiment_features(
        self,
        raw_text: pd.Series,
    ) -> pd.DataFrame:
        """
        Sentiment analysis features using TextBlob (independent group).
        
        This group captures the emotional tone and subjectivity of text,
        which can be indicators of hate speech patterns.
        """
        data = pd.DataFrame(index=raw_text.index)
        
        # Polarity: -1 (negative) to +1 (positive)
        data["sentiment_polarity"] = raw_text.apply(
            lambda text: float(TextBlob(text).sentiment.polarity)
        )
        
        # Subjectivity: 0 (objective) to 1 (subjective)
        data["sentiment_subjectivity"] = raw_text.apply(
            lambda text: float(TextBlob(text).sentiment.subjectivity)
        )
        
        # Additional sentiment features for more comprehensive analysis
        # Sentiment magnitude (absolute polarity)
        data["sentiment_magnitude"] = data["sentiment_polarity"].abs()
        
        # Sentiment category (negative, neutral, positive)
        data["sentiment_negative"] = (data["sentiment_polarity"] < -0.1).astype(int)
        data["sentiment_neutral"] = ((data["sentiment_polarity"] >= -0.1) & 
                                     (data["sentiment_polarity"] <= 0.1)).astype(int)
        data["sentiment_positive"] = (data["sentiment_polarity"] > 0.1).astype(int)
        
        # Extreme sentiment flags
        data["sentiment_very_negative"] = (data["sentiment_polarity"] < -0.5).astype(int)
        data["sentiment_very_positive"] = (data["sentiment_polarity"] > 0.5).astype(int)
        
        # Subjectivity category
        data["highly_subjective"] = (data["sentiment_subjectivity"] > 0.7).astype(int)
        data["highly_objective"] = (data["sentiment_subjectivity"] < 0.3).astype(int)
        
        return data

    def _tfidf_features(self, cleaned_text: pd.Series) -> pd.DataFrame:
        """
        TF-IDF features with unigrams and bigrams.
        
        Uses the fitted TF-IDF vectorizer to transform the text. This ensures
        that both training and test data use the same vocabulary.
        """
        if self.tfidf_vectorizer is None:
            raise RuntimeError("TF-IDF vectorizer not fitted. Call fit() first.")
        
        # Transform using the fitted vectorizer (not fit_transform!)
        matrix = self.tfidf_vectorizer.transform(cleaned_text.tolist())
        feature_names = [f"tfidf_{name}" for name in self.tfidf_vectorizer.get_feature_names_out()]
        tfidf_df = pd.DataFrame.sparse.from_spmatrix(
            matrix,
            index=cleaned_text.index,
            columns=feature_names,
        )
        return tfidf_df

    def _select_tfidf_features_rf(
        self, 
        tfidf_df: pd.DataFrame, 
        labels: np.ndarray, 
        n_select
    ) -> pd.DataFrame:
        """
        Select top N TF-IDF features using Random Forest feature importance.
        
        Args:
            tfidf_df: DataFrame with all TF-IDF features
            labels: Target labels for supervised selection
            n_select: Number of features to keep
            
        Returns:
            DataFrame with selected TF-IDF features only
        """
        from sklearn.ensemble import RandomForestClassifier
        
        LOGGER.info(f"Selecting top {n_select} TF-IDF features using Random Forest...")
        
        X = tfidf_df.values
        y = labels
        
        # Limit selection if fewer features available
        n_select = min(n_select, tfidf_df.shape[1])
        
        # Train RF for feature importance
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        rf.fit(X, y)
        
        # Get top N features by importance
        importances = rf.feature_importances_
        top_indices = np.argsort(importances)[::-1][:n_select]
        
        self.selected_tfidf_feature_names = tfidf_df.columns[top_indices].tolist()
        self.tfidf_selector = {
            'method': 'random_forest',
            'n_selected': n_select,
            'importances': dict(zip(
                self.selected_tfidf_feature_names,
                importances[top_indices]
            ))
        }
        
        LOGGER.info(f"Selected {len(self.selected_tfidf_feature_names)} TF-IDF features")
        LOGGER.info(f"Top 5 features: {[name.replace('tfidf_', '') for name in self.selected_tfidf_feature_names[:5]]}")
        
        return tfidf_df[self.selected_tfidf_feature_names]

    def _sentence_embeddings(self, cleaned_text: pd.Series) -> np.ndarray:
        """
        Encode each text into a sentence-level embedding.
        """
        embeddings = self.embedding_model.encode(
            cleaned_text.tolist(),
            show_progress_bar=False,
            batch_size=self.batch_size,
            convert_to_numpy=True,
        )
        return embeddings

    def _semantic_features(
        self,
        tokens: pd.Series,
        embeddings: np.ndarray,
    ) -> pd.DataFrame:
        """
        POS ratios, pronoun usage, and raw embedding vectors.
        """
        docs = list(
            self.nlp.pipe(
                tokens.apply(lambda tok_list: " ".join(tok_list)).tolist(),
                batch_size=self.batch_size,
            )
        )

        noun_ratio: List[float] = []
        verb_ratio: List[float] = []
        adj_ratio: List[float] = []

        for doc in docs:
            content_tokens = [token for token in doc if not token.is_space]
            denom = len(content_tokens) or 1
            noun_ratio.append(sum(token.pos_ == "NOUN" for token in content_tokens) / denom)
            verb_ratio.append(sum(token.pos_ == "VERB" for token in content_tokens) / denom)
            adj_ratio.append(sum(token.pos_ == "ADJ" for token in content_tokens) / denom)

        pronoun_counts = {
            "they": tokens.apply(lambda toks: toks.count("they")),
            "us": tokens.apply(lambda toks: toks.count("us")),
            "you": tokens.apply(lambda toks: toks.count("you")),
        }
        total_words = tokens.apply(len).replace(0, np.nan)

        data = pd.DataFrame(index=tokens.index)
        data["noun_ratio"] = noun_ratio
        data["verb_ratio"] = verb_ratio
        data["adj_ratio"] = adj_ratio

        for pronoun, counts in pronoun_counts.items():
            data[f"{pronoun}_ratio"] = (counts / total_words).fillna(0.0)

        embed_cols = [f"embed_{idx}" for idx in range(embeddings.shape[1])]
        embed_df = pd.DataFrame(embeddings, index=tokens.index, columns=embed_cols)

        data = pd.concat([data, embed_df], axis=1)
        return data

    def _experimental_features(
        self,
        raw_text: pd.Series,
        tokens: pd.Series,
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> pd.DataFrame:
        """
        Experimental features derived from embeddings and readability.
        """
        data = pd.DataFrame(index=raw_text.index)

        lexicon_similarity = [
            _cosine_similarity(vec, self.lexicon_embedding) for vec in embeddings
        ]
        data["hate_lexicon_similarity"] = lexicon_similarity
        data["language_complexity_index"] = raw_text.apply(_language_complexity)

        labels_series = pd.Series(labels)
        normalized_labels = labels_series.astype(str).str.strip().str.lower()
        neutral_mask = normalized_labels.isin({"0", "0.0", "neutral"}).to_numpy()
        if neutral_mask.any():
            neutral_mean = embeddings[neutral_mask].mean(axis=0)
            distances = [
                _cosine_distance(vec, neutral_mean) for vec in embeddings
            ]
        else:
            distances = [math.nan] * len(raw_text)
        data["embedding_distance_to_neutral"] = distances

        return data

    def _platform_flags(self, index: pd.Index, source_name: str) -> pd.DataFrame:
        """
        One-hot encode the dataset source.
        """
        columns = {f"source_{name}": 0 for name in SOURCE_VOCAB}
        columns[f"source_{source_name}"] = 1
        data = pd.DataFrame(columns, index=index)
        return data


def extract_features(
    df: pd.DataFrame,
    source_name: str,
    text_column: str = "text",
    label_column: str = "label",
    extractor: Optional[TextFeatureExtractor] = None,
) -> pd.DataFrame:
    """
    Convenience wrapper around :class:`TextFeatureExtractor`.
    
    **DEPRECATED USAGE WARNING**: This function now requires a fitted extractor.
    
    For proper usage, use the fit/transform pattern:
    
    ```python
    # Create and fit extractor on training data
    extractor = TextFeatureExtractor()
    train_features = extractor.fit_transform(train_df, source_name='reddit')
    
    # Transform test data with same extractor
    test_features = extractor.transform(test_df, source_name='reddit')
    ```
    
    Args:
        df: DataFrame to transform
        source_name: Name of the dataset source
        text_column: Name of the text column
        label_column: Name of the label column
        extractor: Pre-fitted TextFeatureExtractor instance. If None, creates
                   and fits a new one (NOT recommended - will cause data leakage)
    
    Returns:
        DataFrame with extracted features
    """
    if extractor is None:
        # Backward compatibility mode - creates and fits new extractor
        # This should be avoided as it can cause data leakage
        LOGGER.warning(
            "Creating and fitting new extractor in extract_features(). "
            "This is deprecated. Use fit/transform pattern instead."
        )
        extractor = TextFeatureExtractor()
        return extractor.fit_transform(df, source_name, text_column, label_column)
    
    # Use provided fitted extractor
    return extractor.transform(df, source_name, text_column, label_column)


if __name__ == "__main__":  # pragma: no cover - manual usage example
    logging.basicConfig(level=logging.INFO)

    project_root = Path(__file__).resolve().parents[2]
    reddit_path = project_root / "data" / "raw" / "reddit_dataset.csv"
    chan_path = project_root / "data" / "raw" / "4chan_dataset.csv"

    reddit_df = load_dataset(reddit_path, text_column="Comment", label_column="Hateful")
    reddit_features = extract_features(reddit_df, source_name="reddit")
    LOGGER.info("Reddit features shape: %s", reddit_features.shape)

    chan_df = load_dataset(chan_path, text_column="Comment", label_column="Hateful")
    chan_features = extract_features(chan_df, source_name="4chan")
    LOGGER.info("4chan features shape: %s", chan_features.shape)

