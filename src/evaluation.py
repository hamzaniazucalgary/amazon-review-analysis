"""Evaluation metrics and visualization."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator,
)


def compute_metrics(predictions, include_auc: bool = True) -> dict:
    """Compute classification metrics from PySpark predictions DataFrame."""
    metrics = {}

    evaluators = {
        "accuracy": ("accuracy", MulticlassClassificationEvaluator),
        "precision": ("weightedPrecision", MulticlassClassificationEvaluator),
        "recall": ("weightedRecall", MulticlassClassificationEvaluator),
        "f1": ("f1", MulticlassClassificationEvaluator),
    }
    for name, (metric_name, evaluator_cls) in evaluators.items():
        evaluator = evaluator_cls(labelCol="label", predictionCol="prediction", metricName=metric_name)
        metrics[name] = evaluator.evaluate(predictions)

    if include_auc:
        try:
            auc_eval = BinaryClassificationEvaluator(
                labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
            )
            metrics["auc"] = auc_eval.evaluate(predictions)
        except Exception:
            metrics["auc"] = None

    return metrics


def get_confusion_matrix(predictions) -> np.ndarray:
    """Extract a 2x2 confusion matrix from PySpark predictions."""
    pdf = predictions.select("label", "prediction").toPandas()
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(pdf["label"], pdf["prediction"])


def plot_confusion_matrix(cm: np.ndarray, title: str = "Confusion Matrix", save_path: str = None):
    """Plot a confusion matrix as a seaborn heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues",
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_roc_curve(predictions, title: str = "ROC Curve", save_path: str = None):
    """Plot ROC curve by collecting probabilities from PySpark predictions."""
    from pyspark.sql.functions import udf
    from pyspark.sql.types import FloatType
    from sklearn.metrics import roc_curve, auc

    extract_prob = udf(lambda v: float(v[1]), FloatType())
    pdf = predictions.withColumn("prob", extract_prob("probability")) \
        .select("label", "prob").toPandas()

    fpr, tpr, _ = roc_curve(pdf["label"], pdf["prob"])
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_precision_recall_curve(predictions, title: str = "Precision-Recall Curve", save_path: str = None):
    """Plot precision-recall curve."""
    from pyspark.sql.functions import udf
    from pyspark.sql.types import FloatType
    from sklearn.metrics import precision_recall_curve, average_precision_score

    extract_prob = udf(lambda v: float(v[1]), FloatType())
    pdf = predictions.withColumn("prob", extract_prob("probability")) \
        .select("label", "prob").toPandas()

    precision, recall, _ = precision_recall_curve(pdf["label"], pdf["prob"])
    ap = average_precision_score(pdf["label"], pdf["prob"])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, lw=2, label=f"AP = {ap:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="upper right")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def create_comparison_table(results: list[dict]) -> pd.DataFrame:
    """Create a formatted comparison DataFrame from a list of result dicts.

    Each dict should have keys: model, feature_type, accuracy, precision, recall, f1, auc, training_time.
    """
    df = pd.DataFrame(results)
    numeric_cols = ["accuracy", "precision", "recall", "f1", "auc"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = df[c].apply(lambda x: f"{x:.4f}" if x is not None else "N/A")
    if "training_time" in df.columns:
        df["training_time"] = df["training_time"].apply(lambda x: f"{x:.1f}s")
    return df


def plot_comparison_chart(results: list[dict], save_path: str = None):
    """Plot grouped bar chart comparing models."""
    df = pd.DataFrame(results)
    metrics = ["accuracy", "f1"]
    available = [m for m in metrics if m in df.columns and df[m].notna().any()]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df))
    width = 0.35

    labels = df.apply(lambda r: f"{r.get('model', '')} + {r.get('feature_type', '')}", axis=1)
    for i, metric in enumerate(available):
        ax.bar(x + i * width, df[metric].astype(float), width, label=metric.capitalize())

    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0.8, 1.0)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig
