"""Tests for feature_engineering module."""

import pytest
from src.feature_engineering import get_feature_pipeline, FEATURE_PIPELINES
from src.config import FeatureConfig


def test_all_pipeline_types_exist():
    expected = ["count_vectorizer", "tfidf", "ngram_cv", "ngram_tfidf"]
    for name in expected:
        assert name in FEATURE_PIPELINES


def test_get_feature_pipeline_returns_stages():
    config = FeatureConfig()
    for name in FEATURE_PIPELINES:
        stages = get_feature_pipeline(name, config)
        assert isinstance(stages, list)
        assert len(stages) >= 3


def test_invalid_pipeline_raises():
    with pytest.raises(ValueError, match="Unknown feature type"):
        get_feature_pipeline("invalid_pipeline")


def test_custom_config():
    config = FeatureConfig(vocab_size=1000, min_df=2)
    stages = get_feature_pipeline("count_vectorizer", config)
    # CountVectorizer is the last stage
    cv = stages[-1]
    assert cv.getVocabSize() == 1000
    assert cv.getMinDF() == 2
