"""Model definitions and pipeline builder."""

from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    LogisticRegression, NaiveBayes,
    RandomForestClassifier, GBTClassifier,
)

from src.config import FeatureConfig
from src.feature_engineering import get_feature_pipeline

MODEL_REGISTRY = {
    "logistic_regression": lambda: LogisticRegression(
        featuresCol="features", labelCol="label", maxIter=100
    ),
    "naive_bayes": lambda: NaiveBayes(
        featuresCol="features", labelCol="label", smoothing=1.0
    ),
    "random_forest": lambda: RandomForestClassifier(
        featuresCol="features", labelCol="label", numTrees=100, maxDepth=10
    ),
    "gbt": lambda: GBTClassifier(
        featuresCol="features", labelCol="label", maxIter=50, maxDepth=5
    ),
}


def get_model(model_name: str):
    """Get a model instance by name."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name]()


def build_full_pipeline(feature_type: str, model_name: str, config: FeatureConfig = None) -> Pipeline:
    """Build a complete Pipeline combining feature stages and a model."""
    stages = get_feature_pipeline(feature_type, config)
    model = get_model(model_name)
    return Pipeline(stages=stages + [model])
