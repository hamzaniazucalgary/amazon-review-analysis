"""Data loading utilities for PySpark and Pandas."""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, concat_ws, when, lower

from src.config import SparkConfig, DataConfig


def get_spark_session(config: SparkConfig = None) -> SparkSession:
    """Create and return a SparkSession."""
    if config is None:
        config = SparkConfig()
    return (
        SparkSession.builder
        .appName(config.app_name)
        .config("spark.driver.memory", config.driver_memory)
        .config("spark.executor.memory", config.executor_memory)
        .getOrCreate()
    )


def load_data(spark: SparkSession, config: DataConfig = None) -> tuple[DataFrame, DataFrame]:
    """Load and preprocess data with PySpark.

    Returns (train_df, test_df) with columns: polarity, title, text, combined_text, label.
    """
    if config is None:
        config = DataConfig()

    # Read data (auto-detect format from file extension)
    if config.train_path.endswith(".parquet"):
        train_df = spark.read.parquet(config.train_path)
        test_df = spark.read.parquet(config.test_path)
    else:
        train_df = spark.read.csv(config.train_path, schema=config.schema, header=False, quote='"', escape='"')
        test_df = spark.read.csv(config.test_path, schema=config.schema, header=False, quote='"', escape='"')

    # Preprocess
    for name, df in [("train", train_df), ("test", test_df)]:
        df = df.na.drop()
        df = df.withColumn("title", lower(col("title"))).withColumn("text", lower(col("text")))
        df = df.withColumn("combined_text", concat_ws(" ", col("title"), col("text")))
        df = df.withColumn("label", when(col("polarity") == 1, 0.0).otherwise(1.0))

        if config.sample_fraction < 1.0:
            df = df.sample(fraction=config.sample_fraction, seed=42)

        if name == "train":
            train_df = df
        else:
            test_df = df

    return train_df, test_df


def load_data_pandas(config: DataConfig = None, sample_size: int = None):
    """Load and preprocess data with Pandas (for EDA and transformers)."""
    import pandas as pd

    if config is None:
        config = DataConfig()

    if config.train_path.endswith(".parquet"):
        train_df = pd.read_parquet(config.train_path)
        test_df = pd.read_parquet(config.test_path)
    else:
        train_df = pd.read_csv(config.train_path, names=["polarity", "title", "text"], header=None)
        test_df = pd.read_csv(config.test_path, names=["polarity", "title", "text"], header=None)

    for df in [train_df, test_df]:
        df.dropna(inplace=True)
        df["title"] = df["title"].str.lower()
        df["text"] = df["text"].str.lower()
        df["combined_text"] = df["title"] + " " + df["text"]
        df["label"] = df["polarity"].map({1: 0, 2: 1})

    if sample_size:
        train_df = train_df.sample(n=min(sample_size, len(train_df)), random_state=42)
        test_df = test_df.sample(n=min(sample_size, len(test_df)), random_state=42)

    return train_df, test_df
