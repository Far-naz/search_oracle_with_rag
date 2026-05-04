from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from config import (
    ENABLE_ML_INTENT_RECOGNITION,
    INTENT_MLFLOW_EXPERIMENT,
    INTENT_MLFLOW_MODEL_URI,
    INTENT_MLFLOW_TRACKING_URI,
    INTENT_MODEL_PATH,
    INTENT_MODEL_STORE,
)
from src.intent.ml_recognizer import FeatureExtractor, MLIntentRecognizer, SupportedModel


MLFLOW_ARTIFACT_PATH = "intent_model"
MODEL_FILENAME = "intent_model.pkl"


@dataclass(frozen=True)
class StoredIntentModel:
    model_path: Path
    mlflow_run_id: str | None = None
    mlflow_artifact_uri: str | None = None


def load_configured_intent_model(
    model_path: str | Path = INTENT_MODEL_PATH,
    model_store: str = INTENT_MODEL_STORE,
    mlflow_model_uri: str = INTENT_MLFLOW_MODEL_URI,
    mlflow_tracking_uri: str = INTENT_MLFLOW_TRACKING_URI,
    enable_ml: bool = ENABLE_ML_INTENT_RECOGNITION,
) -> MLIntentRecognizer | None:
    if not enable_ml:
        return None

    normalized_store = (model_store or "local").lower()
    if normalized_store == "mlflow" and mlflow_model_uri:
        return load_intent_model_from_mlflow(
            model_uri=mlflow_model_uri,
            tracking_uri=mlflow_tracking_uri or None,
        )

    path = Path(model_path)
    if not path.exists():
        return None

    return MLIntentRecognizer.load(path)


def train_and_store_intent_model(
    texts: Sequence[str],
    intents: Sequence[str],
    model_path: str | Path = INTENT_MODEL_PATH,
    model_type: SupportedModel = "logistic_regression",
    feature_extractor: FeatureExtractor = "tfidf",
    remove_stopwords: bool = True,
    use_lemmatization: bool = True,
    log_to_mlflow: bool = False,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment: str = INTENT_MLFLOW_EXPERIMENT,
) -> StoredIntentModel:
    recognizer = MLIntentRecognizer(
        model_type=model_type,
        feature_extractor=feature_extractor,
        remove_stopwords=remove_stopwords,
        use_lemmatization=use_lemmatization,
    )
    recognizer.fit(texts, intents)

    path = Path(model_path)
    recognizer.save(path)

    if not log_to_mlflow:
        return StoredIntentModel(model_path=path)

    run_id, artifact_uri = log_intent_model_to_mlflow(
        recognizer=recognizer,
        model_path=path,
        training_example_count=len(texts),
        intent_count=len(set(intents)),
        tracking_uri=mlflow_tracking_uri,
        experiment_name=mlflow_experiment,
    )
    return StoredIntentModel(
        model_path=path,
        mlflow_run_id=run_id,
        mlflow_artifact_uri=artifact_uri,
    )


def log_intent_model_to_mlflow(
    recognizer: MLIntentRecognizer,
    model_path: str | Path,
    training_example_count: int,
    intent_count: int,
    tracking_uri: str | None = None,
    experiment_name: str = INTENT_MLFLOW_EXPERIMENT,
) -> tuple[str, str]:
    mlflow = _import_mlflow()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    if experiment_name:
        mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="intent-recognizer") as run:
        mlflow.log_params(
            {
                "model_type": recognizer.model_type,
                "feature_extractor": recognizer.feature_extractor,
                "remove_stopwords": recognizer.remove_stopwords,
                "use_lemmatization": recognizer.use_lemmatization,
            }
        )
        mlflow.log_metric("training_examples", training_example_count)
        mlflow.log_metric("intent_classes", intent_count)
        mlflow.log_artifact(str(model_path), artifact_path=MLFLOW_ARTIFACT_PATH)
        artifact_uri = mlflow.get_artifact_uri(MLFLOW_ARTIFACT_PATH)
        return run.info.run_id, artifact_uri


def load_intent_model_from_mlflow(
    model_uri: str,
    tracking_uri: str | None = None,
) -> MLIntentRecognizer:
    mlflow = _import_mlflow()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    downloaded_path = Path(mlflow.artifacts.download_artifacts(artifact_uri=model_uri))
    if downloaded_path.is_file():
        return MLIntentRecognizer.load(downloaded_path)

    preferred_path = downloaded_path / MODEL_FILENAME
    if preferred_path.exists():
        return MLIntentRecognizer.load(preferred_path)

    matches = list(downloaded_path.rglob(MODEL_FILENAME))
    if not matches:
        matches = list(downloaded_path.rglob("*.pkl"))
    if not matches:
        raise FileNotFoundError(
            f"Could not find a pickle intent model under MLflow artifact {model_uri}."
        )

    return MLIntentRecognizer.load(matches[0])


def _import_mlflow():
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError(
            "mlflow is required for MLflow-backed intent model storage."
        ) from exc
    return mlflow
