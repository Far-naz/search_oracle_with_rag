from __future__ import annotations

import argparse

from config import (
    INTENT_MLFLOW_EXPERIMENT,
    INTENT_MLFLOW_TRACKING_URI,
    INTENT_MODEL_PATH,
)
from src.intent.model_store import train_and_store_intent_model
from src.intent.training_data import split_training_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and store the intent model.")
    parser.add_argument("--model-path", default=INTENT_MODEL_PATH)
    parser.add_argument(
        "--model-type",
        default="logistic_regression",
        choices=["logistic_regression", "svm", "xgboost", "neural_network"],
    )
    parser.add_argument(
        "--feature-extractor",
        default="tfidf",
        choices=["tfidf", "word2vec"],
    )
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--mlflow-tracking-uri", default=INTENT_MLFLOW_TRACKING_URI)
    parser.add_argument("--mlflow-experiment", default=INTENT_MLFLOW_EXPERIMENT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    texts, intents = split_training_examples()
    stored_model = train_and_store_intent_model(
        texts=texts,
        intents=intents,
        model_path=args.model_path,
        model_type=args.model_type,
        feature_extractor=args.feature_extractor,
        log_to_mlflow=args.mlflow,
        mlflow_tracking_uri=args.mlflow_tracking_uri or None,
        mlflow_experiment=args.mlflow_experiment,
    )

    print(f"Stored local intent model at: {stored_model.model_path}")
    if stored_model.mlflow_run_id:
        print(f"MLflow run id: {stored_model.mlflow_run_id}")
        print(f"MLflow artifact URI: {stored_model.mlflow_artifact_uri}")


if __name__ == "__main__":
    main()
