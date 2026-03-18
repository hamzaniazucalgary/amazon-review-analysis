"""CLI script to run the full PySpark experiment grid."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ProjectConfig
from src.data_loader import get_spark_session, load_data
from src.models import build_full_pipeline
from src.evaluation import compute_metrics, create_comparison_table, plot_comparison_chart
from src.experiment_tracker import init_mlflow, log_experiment
from src.utils import timer


def main():
    config = ProjectConfig()
    spark = get_spark_session(config.spark)
    train_df, test_df = load_data(spark, config.data)
    train_df.cache()
    test_df.cache()
    print(f"Train: {train_df.count():,} | Test: {test_df.count():,}")

    init_mlflow(config.mlflow_tracking_uri, config.mlflow_experiment_name)

    experiments = [
        ("logistic_regression", "count_vectorizer"),
        ("logistic_regression", "tfidf"),
        ("logistic_regression", "ngram_cv"),
        ("logistic_regression", "ngram_tfidf"),
        ("naive_bayes", "count_vectorizer"),
        ("naive_bayes", "tfidf"),
        ("random_forest", "count_vectorizer"),
        ("gbt", "count_vectorizer"),
    ]

    results = []
    for model_name, feature_type in experiments:
        run_name = f"{model_name}_{feature_type}"
        print(f"\n{'='*60}\nTraining: {run_name}\n{'='*60}")

        pipeline = build_full_pipeline(feature_type, model_name, config.features)

        with timer(run_name) as elapsed:
            model = pipeline.fit(train_df)
        training_time = elapsed()

        predictions = model.transform(test_df)
        include_auc = model_name != "gbt"
        metrics = compute_metrics(predictions, include_auc=include_auc)

        result = {"model": model_name, "feature_type": feature_type, **metrics, "training_time": training_time}
        results.append(result)

        log_experiment(run_name=run_name, model_name=model_name, feature_type=feature_type,
                       hyperparams={"vocab_size": config.features.vocab_size, "min_df": config.features.min_df},
                       metrics=metrics, training_time=training_time)

        print(f"Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}")

    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    comparison = create_comparison_table(results)
    print(comparison.to_string(index=False))

    os.makedirs("docs", exist_ok=True)
    plot_comparison_chart(results, save_path="docs/model_comparison.png")
    print("\nComparison chart saved to docs/model_comparison.png")

    spark.stop()


if __name__ == "__main__":
    main()
