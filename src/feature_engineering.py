"""Feature engineering pipelines for PySpark ML."""

from pyspark.ml.feature import (
    Tokenizer, StopWordsRemover, CountVectorizer,
    HashingTF, IDF, NGram
)

from src.config import FeatureConfig


def count_vectorizer_stages(config: FeatureConfig = None) -> list:
    """Tokenizer -> StopWordsRemover -> CountVectorizer."""
    if config is None:
        config = FeatureConfig()
    return [
        Tokenizer(inputCol="combined_text", outputCol="tokens"),
        StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens"),
        CountVectorizer(inputCol="filtered_tokens", outputCol="features",
                        vocabSize=config.vocab_size, minDF=config.min_df),
    ]


def tfidf_stages(config: FeatureConfig = None) -> list:
    """Tokenizer -> StopWordsRemover -> HashingTF -> IDF."""
    if config is None:
        config = FeatureConfig()
    return [
        Tokenizer(inputCol="combined_text", outputCol="tokens"),
        StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens"),
        HashingTF(inputCol="filtered_tokens", outputCol="raw_features", numFeatures=config.vocab_size),
        IDF(inputCol="raw_features", outputCol="features"),
    ]


def ngram_cv_stages(config: FeatureConfig = None) -> list:
    """Tokenizer -> StopWordsRemover -> NGram -> CountVectorizer (bigrams)."""
    if config is None:
        config = FeatureConfig()
    return [
        Tokenizer(inputCol="combined_text", outputCol="tokens"),
        StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens"),
        NGram(n=config.ngram_range, inputCol="filtered_tokens", outputCol="ngrams"),
        CountVectorizer(inputCol="ngrams", outputCol="features",
                        vocabSize=config.vocab_size, minDF=config.min_df),
    ]


def ngram_tfidf_stages(config: FeatureConfig = None) -> list:
    """Tokenizer -> StopWordsRemover -> NGram -> HashingTF -> IDF (bigrams)."""
    if config is None:
        config = FeatureConfig()
    return [
        Tokenizer(inputCol="combined_text", outputCol="tokens"),
        StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens"),
        NGram(n=config.ngram_range, inputCol="filtered_tokens", outputCol="ngrams"),
        HashingTF(inputCol="ngrams", outputCol="raw_features", numFeatures=config.vocab_size),
        IDF(inputCol="raw_features", outputCol="features"),
    ]


FEATURE_PIPELINES = {
    "count_vectorizer": count_vectorizer_stages,
    "tfidf": tfidf_stages,
    "ngram_cv": ngram_cv_stages,
    "ngram_tfidf": ngram_tfidf_stages,
}


def get_feature_pipeline(feature_type: str, config: FeatureConfig = None) -> list:
    """Factory function to get feature pipeline stages by name."""
    if feature_type not in FEATURE_PIPELINES:
        raise ValueError(f"Unknown feature type: {feature_type}. Choose from: {list(FEATURE_PIPELINES.keys())}")
    return FEATURE_PIPELINES[feature_type](config)
