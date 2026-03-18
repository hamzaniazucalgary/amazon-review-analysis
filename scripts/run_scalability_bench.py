"""CLI script to run scalability benchmarks."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ProjectConfig
from src.data_loader import get_spark_session, load_data
from src.models import build_full_pipeline
from src.evaluation import compute_metrics
from src.utils import timer


def main():
    config = ProjectConfig()
    spark = get_spark_session(config.spark)
    train_df, test_df = load_data(spark, config.data)
    train_df.cache()
    test_df.cache()
    full_count = train_df.count()
    print(f"Full training set: {full_count:,} rows")

    fractions = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
    results = []

    for frac in fractions:
        sample = train_df.sample(fraction=frac, seed=42)
        n_rows = sample.count()
        print(f"\n--- Fraction: {frac} ({n_rows:,} rows) ---")

        pipeline = build_full_pipeline("count_vectorizer", "logistic_regression", config.features)

        with timer(f"Training on {n_rows:,} rows") as elapsed:
            model = pipeline.fit(sample)
        training_time = elapsed()

        predictions = model.transform(test_df)
        metrics = compute_metrics(predictions)
        throughput = n_rows / training_time

        results.append({
            "fraction": frac, "n_rows": n_rows,
            "training_time": training_time,
            "accuracy": metrics["accuracy"],
            "throughput": throughput,
        })
        print(f"Accuracy: {metrics['accuracy']:.4f} | Time: {training_time:.1f}s | Throughput: {throughput:,.0f} rows/sec")

    print("\n" + "="*60)
    print("SCALABILITY RESULTS")
    print("="*60)
    import pandas as pd
    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    spark.stop()


if __name__ == "__main__":
    main()
