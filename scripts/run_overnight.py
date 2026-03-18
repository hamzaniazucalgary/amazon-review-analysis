"""
Overnight runner: executes the full pipeline from data download to model export.

Usage:
    source .venv/bin/activate
    nohup python scripts/run_overnight.py > overnight.log 2>&1 &

Features:
    - Per-step error handling (one failure won't kill the run)
    - File-based progress log (check overnight_progress.json)
    - Spark sessions cleaned up in finally blocks
    - Reduced RF/GBT params to prevent OOM
    - matplotlib Agg backend (no display needed)
    - Explicit garbage collection between heavy steps
"""

import sys
import os
import gc
import json
import time
import traceback
from datetime import datetime

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

# Force non-interactive matplotlib backend before any imports
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

PROGRESS_FILE = os.path.join(PROJECT_ROOT, "overnight_progress.json")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log(msg: str):
    """Print with timestamp, flush immediately so nohup logs are real-time."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def save_progress(step: str, status: str, detail: str = ""):
    """Append step result to progress file."""
    progress = []
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            progress = json.load(f)
    progress.append({
        "step": step,
        "status": status,
        "detail": detail,
        "timestamp": datetime.now().isoformat(),
    })
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def run_step(name: str, func, *args, **kwargs):
    """Run a step with error handling and progress tracking."""
    log(f"{'='*60}")
    log(f"STARTING: {name}")
    log(f"{'='*60}")
    start = time.time()
    try:
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        save_progress(name, "SUCCESS", f"{elapsed:.1f}s")
        log(f"COMPLETED: {name} ({elapsed:.1f}s)")
        return result
    except Exception as e:
        elapsed = time.time() - start
        tb = traceback.format_exc()
        save_progress(name, "FAILED", f"{elapsed:.1f}s — {str(e)}")
        log(f"FAILED: {name} ({elapsed:.1f}s)")
        log(tb)
        return None
    finally:
        # Force garbage collection between steps to free memory
        gc.collect()


# ---------------------------------------------------------------------------
# Step 0: Prerequisites check
# ---------------------------------------------------------------------------

def step_check_prerequisites():
    """Verify Java and data files exist."""
    # Check Java
    import subprocess
    result = subprocess.run(["java", "-version"], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("Java not found. Install: sudo apt install openjdk-11-jre-headless")
    java_version = result.stderr.split("\n")[0]
    log(f"Java: {java_version}")

    os.makedirs(DOCS_DIR, exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "app", "model_artifacts"), exist_ok=True)


# ---------------------------------------------------------------------------
# Step 1: Data download
# ---------------------------------------------------------------------------

def step_download_data():
    """Download dataset if not already present."""
    train_path = os.path.join(PROJECT_ROOT, "data", "train.parquet")
    test_path = os.path.join(PROJECT_ROOT, "data", "test.parquet")

    if os.path.exists(train_path) and os.path.exists(test_path):
        train_size = os.path.getsize(train_path) / (1024 * 1024)
        test_size = os.path.getsize(test_path) / (1024 * 1024)
        log(f"Data already exists: train={train_size:.0f}MB, test={test_size:.0f}MB")
        return

    log("Downloading dataset from HuggingFace...")
    from data.download_data import download_huggingface
    download_huggingface(os.path.join(PROJECT_ROOT, "data"), "parquet")


# ---------------------------------------------------------------------------
# Step 2: EDA — generate all exploration plots
# ---------------------------------------------------------------------------

def step_eda():
    """Run EDA and generate all plots (Pandas-based, quick)."""
    from src.config import DataConfig
    from src.data_loader import load_data_pandas

    config = DataConfig()
    train_df, test_df = load_data_pandas(config, sample_size=500_000)
    log(f"EDA data loaded: train={len(train_df):,}, test={len(test_df):,}")

    # --- Class distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (name, df) in zip(axes, [("Train", train_df), ("Test", test_df)]):
        counts = df["label"].value_counts().sort_index()
        ax.bar(["Negative", "Positive"], counts.values, color=["#e74c3c", "#2ecc71"])
        ax.set_title(f"{name} Set Class Distribution")
        ax.set_ylabel("Count")
        for i, v in enumerate(counts.values):
            ax.text(i, v + 1000, f"{v:,}", ha="center")
    plt.tight_layout()
    fig.savefig(os.path.join(DOCS_DIR, "class_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log("Saved: class_distribution.png")

    # --- Review length ---
    train_df["char_length"] = train_df["combined_text"].str.len()
    train_df["word_count"] = train_df["combined_text"].str.split().str.len()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(train_df["char_length"].clip(upper=3000), bins=100, color="steelblue", edgecolor="none", alpha=0.8)
    axes[0].set_title("Character Length Distribution")
    axes[0].set_xlabel("Character Length")
    axes[0].set_ylabel("Count")
    axes[1].hist(train_df["word_count"].clip(upper=500), bins=100, color="coral", edgecolor="none", alpha=0.8)
    axes[1].set_title("Word Count Distribution")
    axes[1].set_xlabel("Word Count")
    axes[1].set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(os.path.join(DOCS_DIR, "review_length_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log("Saved: review_length_distribution.png")

    # --- Top words ---
    from sklearn.feature_extraction.text import CountVectorizer as SklearnCV

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, (label, title, color) in zip(axes, [(0, "Negative Reviews", "#e74c3c"), (1, "Positive Reviews", "#2ecc71")]):
        texts = train_df[train_df["label"] == label]["combined_text"].sample(50000, random_state=42)
        cv = SklearnCV(max_features=30, stop_words="english")
        X = cv.fit_transform(texts)
        word_counts = X.sum(axis=0).A1
        words = cv.get_feature_names_out()
        sorted_idx = word_counts.argsort()
        ax.barh(words[sorted_idx], word_counts[sorted_idx], color=color)
        ax.set_title(f"Top 30 Words - {title}")
    plt.tight_layout()
    fig.savefig(os.path.join(DOCS_DIR, "top_words.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log("Saved: top_words.png")

    # --- Word clouds ---
    from wordcloud import WordCloud

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, (label, title, cmap) in zip(axes, [(0, "Negative Reviews", "Reds"), (1, "Positive Reviews", "Greens")]):
        text = " ".join(train_df[train_df["label"] == label]["combined_text"].sample(50000, random_state=42))
        wc = WordCloud(width=800, height=400, background_color="white", colormap=cmap, max_words=200).generate(text)
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(title, fontsize=14)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(os.path.join(DOCS_DIR, "wordclouds.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log("Saved: wordclouds.png")

    del train_df, test_df
    gc.collect()


# ---------------------------------------------------------------------------
# Step 3: PySpark model training — the heavy step
# ---------------------------------------------------------------------------

def step_train_pyspark_models():
    """Train all 8 PySpark model configurations, save best model, generate plots."""
    from src.config import ProjectConfig, FeatureConfig
    from src.data_loader import get_spark_session, load_data
    from src.models import build_full_pipeline, MODEL_REGISTRY
    from src.evaluation import (
        compute_metrics, get_confusion_matrix, plot_confusion_matrix,
        plot_roc_curve, plot_precision_recall_curve,
        create_comparison_table, plot_comparison_chart,
    )
    from src.experiment_tracker import init_mlflow, log_experiment
    from src.utils import timer

    config = ProjectConfig()
    spark = None
    try:
        spark = get_spark_session(config.spark)
        train_df, test_df = load_data(spark, config.data)
        train_df.cache()
        test_df.cache()
        train_count = train_df.count()
        test_count = test_df.count()
        log(f"Data loaded: train={train_count:,}, test={test_count:,}")

        init_mlflow(config.mlflow_tracking_uri, config.mlflow_experiment_name)

        # Use reduced vocab for tree models to avoid OOM/extreme runtimes
        default_config = config.features
        reduced_config = FeatureConfig(vocab_size=16384, min_df=5)

        experiments = [
            ("logistic_regression", "count_vectorizer", default_config),
            ("logistic_regression", "tfidf", default_config),
            ("logistic_regression", "ngram_cv", default_config),
            ("logistic_regression", "ngram_tfidf", default_config),
            ("naive_bayes", "count_vectorizer", default_config),
            ("naive_bayes", "tfidf", default_config),
            ("random_forest", "count_vectorizer", reduced_config),
            ("gbt", "count_vectorizer", reduced_config),
        ]

        results = []
        best_accuracy = 0.0
        best_model = None
        best_predictions = None
        best_name = ""

        for model_name, feature_type, feat_config in experiments:
            run_name = f"{model_name}_{feature_type}"
            log(f"Training: {run_name}")

            try:
                pipeline = build_full_pipeline(feature_type, model_name, feat_config)

                with timer(run_name) as elapsed:
                    model = pipeline.fit(train_df)
                training_time = elapsed()

                predictions = model.transform(test_df)
                predictions.cache()

                include_auc = model_name != "gbt"
                metrics = compute_metrics(predictions, include_auc=include_auc)

                result = {
                    "model": model_name,
                    "feature_type": feature_type,
                    **metrics,
                    "training_time": training_time,
                }
                results.append(result)

                log_experiment(
                    run_name=run_name, model_name=model_name, feature_type=feature_type,
                    hyperparams={"vocab_size": feat_config.vocab_size, "min_df": feat_config.min_df},
                    metrics=metrics, training_time=training_time,
                )

                log(f"  Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | "
                    f"AUC: {metrics.get('auc', 'N/A')} | Time: {training_time:.1f}s")

                # Track best model (by accuracy)
                if metrics["accuracy"] > best_accuracy:
                    # Uncache previous best predictions
                    if best_predictions is not None:
                        best_predictions.unpersist()
                    best_accuracy = metrics["accuracy"]
                    best_model = model
                    best_predictions = predictions
                    best_name = run_name
                else:
                    predictions.unpersist()

            except Exception as e:
                log(f"  FAILED: {run_name} — {e}")
                traceback.print_exc()
                results.append({
                    "model": model_name, "feature_type": feature_type,
                    "accuracy": None, "f1": None, "auc": None,
                    "training_time": None, "error": str(e),
                })

        # --- Results comparison ---
        valid_results = [r for r in results if r.get("accuracy") is not None]
        if valid_results:
            log("\nRESULTS COMPARISON:")
            comparison = create_comparison_table(valid_results)
            log("\n" + comparison.to_string(index=False))

            plot_comparison_chart(valid_results, save_path=os.path.join(DOCS_DIR, "model_comparison.png"))
            log("Saved: model_comparison.png")

        # --- Best model plots + save ---
        if best_model is not None and best_predictions is not None:
            log(f"\nBest model: {best_name} (accuracy: {best_accuracy:.4f})")

            cm = get_confusion_matrix(best_predictions)
            plot_confusion_matrix(cm, title=f"Confusion Matrix - {best_name}",
                                  save_path=os.path.join(DOCS_DIR, "confusion_matrix.png"))
            log("Saved: confusion_matrix.png")

            # ROC and PR curves (only if model outputs probabilities)
            if "probability" in best_predictions.columns:
                plot_roc_curve(best_predictions, title=f"ROC Curve - {best_name}",
                               save_path=os.path.join(DOCS_DIR, "roc_curve.png"))
                log("Saved: roc_curve.png")

                plot_precision_recall_curve(best_predictions, title=f"PR Curve - {best_name}",
                                           save_path=os.path.join(DOCS_DIR, "pr_curve.png"))
                log("Saved: pr_curve.png")

            # Save best model
            model_path = os.path.join(PROJECT_ROOT, "app", "model_artifacts", "spark_model")
            best_model.write().overwrite().save(model_path)
            log(f"Best model saved to {model_path}")

            best_predictions.unpersist()

        # Save results to JSON for later use
        results_path = os.path.join(DOCS_DIR, "pyspark_results.json")
        with open(results_path, "w") as f:
            json.dump(valid_results, f, indent=2, default=str)
        log(f"Results saved to {results_path}")

        train_df.unpersist()
        test_df.unpersist()

        return valid_results

    finally:
        if spark is not None:
            spark.stop()
            log("Spark session stopped.")


# ---------------------------------------------------------------------------
# Step 4: Error analysis
# ---------------------------------------------------------------------------

def step_error_analysis():
    """Run error analysis on the best model's predictions."""
    from src.config import ProjectConfig
    from src.data_loader import get_spark_session, load_data
    from pyspark.ml import PipelineModel
    from pyspark.sql.functions import udf, col, length, when
    from pyspark.sql.types import FloatType

    model_path = os.path.join(PROJECT_ROOT, "app", "model_artifacts", "spark_model")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No saved model at {model_path}. Train models first.")

    config = ProjectConfig()
    spark = None
    try:
        spark = get_spark_session(config.spark)
        _, test_df = load_data(spark, config.data)

        model = PipelineModel.load(model_path)
        predictions = model.transform(test_df)
        predictions.cache()

        total = predictions.count()
        correct = predictions.filter("label = prediction").count()
        incorrect = total - correct
        log(f"Total: {total:,} | Correct: {correct:,} ({correct/total*100:.2f}%) | "
            f"Incorrect: {incorrect:,} ({incorrect/total*100:.2f}%)")

        # --- Confidence distribution ---
        extract_prob = udf(lambda v: float(v[1]), FloatType())
        pdf = predictions.withColumn("confidence", extract_prob("probability")) \
            .withColumn("is_correct", (col("label") == col("prediction")).cast("int")) \
            .select("confidence", "is_correct").toPandas()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(pdf[pdf["is_correct"] == 1]["confidence"], bins=50, alpha=0.6, label="Correct", color="green")
        ax.hist(pdf[pdf["is_correct"] == 0]["confidence"], bins=50, alpha=0.6, label="Incorrect", color="red")
        ax.set_xlabel("Predicted Probability (Positive Class)")
        ax.set_ylabel("Count")
        ax.set_title("Confidence Distribution: Correct vs Incorrect Predictions")
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(DOCS_DIR, "confidence_distribution.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        log("Saved: confidence_distribution.png")

        # --- Accuracy by review length ---
        length_df = predictions.withColumn("text_length", length("combined_text")) \
            .withColumn("length_bucket",
                when(col("text_length") < 100, "0-100")
                .when(col("text_length") < 300, "100-300")
                .when(col("text_length") < 500, "300-500")
                .when(col("text_length") < 1000, "500-1000")
                .otherwise("1000+")
            ) \
            .withColumn("is_correct", (col("label") == col("prediction")).cast("int"))

        bucket_acc = length_df.groupBy("length_bucket").agg({"is_correct": "avg"}).toPandas()
        bucket_acc.columns = ["length_bucket", "accuracy"]
        order = ["0-100", "100-300", "300-500", "500-1000", "1000+"]
        bucket_acc["length_bucket"] = pd.Categorical(bucket_acc["length_bucket"], categories=order, ordered=True)
        bucket_acc = bucket_acc.sort_values("length_bucket")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(bucket_acc["length_bucket"], bucket_acc["accuracy"], marker="o", lw=2, color="steelblue")
        ax.set_xlabel("Review Length (characters)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy by Review Length")
        ax.set_ylim(0.8, 1.0)
        plt.tight_layout()
        fig.savefig(os.path.join(DOCS_DIR, "accuracy_by_length.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        log("Saved: accuracy_by_length.png")

        predictions.unpersist()

    finally:
        if spark is not None:
            spark.stop()
            log("Spark session stopped.")


# ---------------------------------------------------------------------------
# Step 5: Scalability benchmarks
# ---------------------------------------------------------------------------

def step_scalability():
    """Run scalability benchmarks at 7 sample fractions, generate plot."""
    from src.config import ProjectConfig
    from src.data_loader import get_spark_session, load_data
    from src.models import build_full_pipeline
    from src.evaluation import compute_metrics
    from src.utils import timer

    config = ProjectConfig()
    spark = None
    try:
        spark = get_spark_session(config.spark)
        train_df, test_df = load_data(spark, config.data)
        train_df.cache()
        test_df.cache()
        full_count = train_df.count()
        log(f"Full training set: {full_count:,} rows")

        fractions = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
        results = []

        for frac in fractions:
            sample = train_df.sample(fraction=frac, seed=42)
            n_rows = sample.count()
            log(f"Fraction {frac}: {n_rows:,} rows")

            pipeline = build_full_pipeline("count_vectorizer", "logistic_regression", config.features)

            with timer(f"  Training on {n_rows:,} rows") as elapsed:
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
            log(f"  Accuracy: {metrics['accuracy']:.4f} | Time: {training_time:.1f}s | "
                f"Throughput: {throughput:,.0f} rows/sec")

        # --- Dual-axis plot ---
        df = pd.DataFrame(results)

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        ax1.plot(df["n_rows"], df["training_time"], "b-o", lw=2, markersize=8, label="Training Time")
        ax2.plot(df["n_rows"], df["accuracy"], "r-s", lw=2, markersize=8, label="Accuracy")

        ax1.set_xlabel("Dataset Size (rows)")
        ax1.set_ylabel("Training Time (seconds)", color="blue")
        ax2.set_ylabel("Accuracy", color="red")
        ax1.tick_params(axis="y", labelcolor="blue")
        ax2.tick_params(axis="y", labelcolor="red")
        ax1.xaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"))

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
        plt.title("Scalability: Training Time & Accuracy vs Dataset Size")
        plt.tight_layout()
        fig.savefig(os.path.join(DOCS_DIR, "scalability.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        log("Saved: scalability.png")

        # Save results
        with open(os.path.join(DOCS_DIR, "scalability_results.json"), "w") as f:
            json.dump(results, f, indent=2)

        train_df.unpersist()
        test_df.unpersist()

    finally:
        if spark is not None:
            spark.stop()
            log("Spark session stopped.")


# ---------------------------------------------------------------------------
# Step 6: DistilBERT training
# ---------------------------------------------------------------------------

def step_train_distilbert():
    """Fine-tune DistilBERT on 200K samples."""
    import torch
    from torch.utils.data import Dataset as TorchDataset

    from src.config import TransformerConfig, DataConfig
    from src.data_loader import load_data_pandas
    from src.experiment_tracker import init_mlflow, log_experiment

    t_config = TransformerConfig()
    d_config = DataConfig()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")
    if device == "cuda":
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        log(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load data
    train_df, test_df = load_data_pandas(d_config)
    train_df = train_df.sample(n=t_config.sample_size, random_state=42)
    test_df = test_df.sample(n=t_config.sample_size // 4, random_state=42)

    from sklearn.model_selection import train_test_split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_df["combined_text"].tolist(), train_df["label"].tolist(),
        test_size=0.1, random_state=42, stratify=train_df["label"],
    )
    log(f"Train: {len(train_texts):,} | Val: {len(val_texts):,} | Test: {len(test_df):,}")

    # Tokenize
    from transformers import DistilBertTokenizerFast
    tokenizer = DistilBertTokenizerFast.from_pretrained(t_config.model_name)

    log("Tokenizing...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=t_config.max_length)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=t_config.max_length)
    test_encodings = tokenizer(test_df["combined_text"].tolist(), truncation=True, padding=True, max_length=t_config.max_length)
    log("Tokenization complete.")

    # Dataset class
    class ReviewDataset(TorchDataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

    train_dataset = ReviewDataset(train_encodings, train_labels)
    val_dataset = ReviewDataset(val_encodings, val_labels)
    test_dataset = ReviewDataset(test_encodings, test_df["label"].tolist())

    # Train
    from transformers import DistilBertForSequenceClassification, TrainingArguments, Trainer
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    def compute_hf_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        acc = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
        return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

    model = DistilBertForSequenceClassification.from_pretrained(t_config.model_name, num_labels=2)

    output_dir = os.path.join(PROJECT_ROOT, "transformer_results")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=t_config.epochs,
        per_device_train_batch_size=t_config.batch_size,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(PROJECT_ROOT, "transformer_logs"),
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=(device == "cuda"),
        learning_rate=t_config.lr,
        report_to="none",  # don't try to log to wandb etc.
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_hf_metrics,
    )

    log("Starting DistilBERT fine-tuning...")
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    log(f"Fine-tuning complete in {training_time:.1f}s")

    # Evaluate
    test_results = trainer.evaluate(test_dataset)
    log("DistilBERT test results:")
    for k, v in test_results.items():
        log(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Log to MLflow
    init_mlflow()
    log_experiment(
        run_name="distilbert_baseline",
        model_name="distilbert",
        feature_type="transformer_embeddings",
        hyperparams={
            "max_length": t_config.max_length,
            "epochs": t_config.epochs,
            "lr": t_config.lr,
            "sample_size": t_config.sample_size,
        },
        metrics={k.replace("eval_", ""): v for k, v in test_results.items() if isinstance(v, float)},
        training_time=training_time,
    )
    log("DistilBERT results logged to MLflow.")

    # Save results
    with open(os.path.join(DOCS_DIR, "distilbert_results.json"), "w") as f:
        json.dump({k: v for k, v in test_results.items()}, f, indent=2, default=str)

    # Cleanup GPU memory
    del model, trainer, train_dataset, val_dataset, test_dataset
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()


# ---------------------------------------------------------------------------
# Step 7: Export model for Streamlit
# ---------------------------------------------------------------------------

def step_export_model():
    """Export the best PySpark model as a sklearn pickle for Streamlit."""
    model_path = os.path.join(PROJECT_ROOT, "app", "model_artifacts", "spark_model")
    output_path = os.path.join(PROJECT_ROOT, "app", "model_artifacts", "sklearn_model.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No saved model at {model_path}. Train models first.")

    from scripts.export_model_for_app import export_from_spark_model
    export_from_spark_model(model_path, output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    total_start = time.time()

    log("=" * 60)
    log("OVERNIGHT RUN STARTED")
    log(f"Project root: {PROJECT_ROOT}")
    log("=" * 60)

    # Clear previous progress
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)

    run_step("0. Prerequisites check", step_check_prerequisites)
    run_step("1. Data download", step_download_data)

    # Verify data exists before continuing — everything else depends on it
    train_path = os.path.join(PROJECT_ROOT, "data", "train.parquet")
    if not os.path.exists(train_path):
        log("FATAL: Data download failed. Cannot continue without data.")
        log("Fix the issue and re-run: python scripts/run_overnight.py")
        return

    run_step("2. EDA plots", step_eda)
    run_step("3. PySpark model training (8 configs)", step_train_pyspark_models)
    run_step("4. Error analysis", step_error_analysis)
    run_step("5. Scalability benchmarks", step_scalability)
    run_step("6. DistilBERT fine-tuning", step_train_distilbert)
    run_step("7. Export model for Streamlit", step_export_model)

    total_elapsed = time.time() - total_start
    hours = total_elapsed / 3600

    log("")
    log("=" * 60)
    log(f"OVERNIGHT RUN COMPLETE — Total time: {hours:.1f} hours ({total_elapsed:.0f}s)")
    log("=" * 60)

    # Print summary
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            progress = json.load(f)
        log("\nSTEP SUMMARY:")
        for step in progress:
            icon = "OK" if step["status"] == "SUCCESS" else "FAIL"
            log(f"  [{icon}] {step['step']} — {step['detail']}")

    log(f"\nCheck overnight_progress.json for details.")
    log(f"View MLflow: mlflow ui --port 5000")
    log(f"Launch app:  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
