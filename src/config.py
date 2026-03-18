"""Central configuration for the project."""

import os
from dataclasses import dataclass, field


def is_colab() -> bool:
    """Detect if running in Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


@dataclass
class SparkConfig:
    app_name: str = "AmazonReviewsSentiment"
    driver_memory: str = "8g"
    executor_memory: str = "8g"


@dataclass
class DataConfig:
    train_path: str = "data/train.parquet"
    test_path: str = "data/test.parquet"
    schema: str = "polarity INT, title STRING, text STRING"
    sample_fraction: float = 1.0

    def __post_init__(self):
        if is_colab():
            drive_base = "/content/drive/MyDrive/amazon-reviews-sentiment-analysis"
            self.train_path = os.path.join(drive_base, "data/train.parquet")
            self.test_path = os.path.join(drive_base, "data/test.parquet")


@dataclass
class FeatureConfig:
    vocab_size: int = 65536
    min_df: int = 5
    use_tfidf: bool = False
    use_ngrams: bool = False
    ngram_range: int = 2


@dataclass
class TransformerConfig:
    model_name: str = "distilbert-base-uncased"
    sample_size: int = 200_000
    max_length: int = 256
    batch_size: int = 32
    epochs: int = 3
    lr: float = 2e-5


@dataclass
class ProjectConfig:
    spark: SparkConfig = field(default_factory=SparkConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    mlflow_tracking_uri: str = "mlruns"
    mlflow_experiment_name: str = "amazon-reviews-sentiment"
