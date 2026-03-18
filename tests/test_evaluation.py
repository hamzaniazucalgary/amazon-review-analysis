"""Tests for evaluation module."""

import pytest
import numpy as np
import pandas as pd
from src.evaluation import create_comparison_table


def test_create_comparison_table():
    results = [
        {"model": "lr", "feature_type": "cv", "accuracy": 0.9, "f1": 0.89, "auc": 0.95, "training_time": 120.5},
        {"model": "nb", "feature_type": "tfidf", "accuracy": 0.85, "f1": 0.84, "auc": 0.92, "training_time": 60.2},
    ]
    df = create_comparison_table(results)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "model" in df.columns
    assert "accuracy" in df.columns


def test_comparison_table_formatting():
    results = [
        {"model": "lr", "feature_type": "cv", "accuracy": 0.9017, "f1": 0.9015, "auc": 0.9616, "training_time": 120.5},
    ]
    df = create_comparison_table(results)
    assert df.iloc[0]["accuracy"] == "0.9017"
    assert df.iloc[0]["training_time"] == "120.5s"


def test_comparison_table_handles_none_auc():
    results = [
        {"model": "gbt", "feature_type": "cv", "accuracy": 0.88, "f1": 0.87, "auc": None, "training_time": 300.0},
    ]
    df = create_comparison_table(results)
    assert df.iloc[0]["auc"] == "N/A"
