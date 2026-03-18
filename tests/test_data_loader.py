"""Tests for data_loader module."""

import pytest
from unittest.mock import patch, MagicMock
from src.config import SparkConfig, DataConfig


def test_spark_config_defaults():
    config = SparkConfig()
    assert config.app_name == "AmazonReviewsSentiment"
    assert config.driver_memory == "8g"


def test_data_config_defaults():
    config = DataConfig()
    assert config.train_path == "data/train.parquet"
    assert config.sample_fraction == 1.0


def test_data_config_schema():
    config = DataConfig()
    assert "polarity" in config.schema
    assert "title" in config.schema
    assert "text" in config.schema
