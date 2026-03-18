"""MLflow experiment tracking wrapper."""

import mlflow


def init_mlflow(tracking_uri: str = "mlruns", experiment_name: str = "amazon-reviews-sentiment"):
    """Initialize MLflow tracking."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_experiment(
    run_name: str,
    model_name: str,
    feature_type: str,
    hyperparams: dict,
    metrics: dict,
    training_time: float,
    artifacts: list[str] = None,
):
    """Log a single experiment run to MLflow."""
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("feature_type", feature_type)
        for k, v in hyperparams.items():
            mlflow.log_param(k, v)

        for k, v in metrics.items():
            if v is not None:
                mlflow.log_metric(k, v)
        mlflow.log_metric("training_time_seconds", training_time)

        if artifacts:
            for path in artifacts:
                mlflow.log_artifact(path)
