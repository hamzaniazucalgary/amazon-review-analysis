"""Export PySpark model as sklearn pipeline for lightweight Streamlit inference."""

import sys
import os
import pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SklearnPipeline


def export_from_spark_model(spark_model_path: str, output_path: str):
    """Extract vocabulary and coefficients from PySpark model and rebuild as sklearn."""
    from pyspark.ml import PipelineModel
    from src.data_loader import get_spark_session

    spark = get_spark_session()
    model = PipelineModel.load(spark_model_path)

    # Extract CountVectorizer vocabulary (stage index 2)
    cv_model = model.stages[2]
    vocabulary = cv_model.vocabulary

    # Extract LogisticRegression coefficients (last stage)
    lr_model = model.stages[-1]
    coefficients = lr_model.coefficients.toArray()
    intercept = lr_model.intercept

    spark.stop()

    # Rebuild as sklearn pipeline
    sklearn_cv = CountVectorizer(vocabulary={word: i for i, word in enumerate(vocabulary)},
                                  lowercase=True)
    sklearn_lr = LogisticRegression()
    sklearn_lr.classes_ = np.array([0, 1])
    sklearn_lr.coef_ = coefficients.reshape(1, -1)
    sklearn_lr.intercept_ = np.array([intercept])

    sklearn_pipeline = SklearnPipeline([
        ("vectorizer", sklearn_cv),
        ("classifier", sklearn_lr),
    ])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(sklearn_pipeline, f)

    print(f"sklearn model exported to {output_path}")
    print(f"Vocabulary size: {len(vocabulary):,}")


if __name__ == "__main__":
    spark_path = sys.argv[1] if len(sys.argv) > 1 else "app/model_artifacts/spark_model"
    output = sys.argv[2] if len(sys.argv) > 2 else "app/model_artifacts/sklearn_model.pkl"
    export_from_spark_model(spark_path, output)
