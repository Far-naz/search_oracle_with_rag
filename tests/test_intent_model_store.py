from src.intent.model_store import (
    load_configured_intent_model,
    train_and_store_intent_model,
)
from src.intent.training_data import split_training_examples


def test_train_and_store_intent_model_writes_loadable_artifact(tmp_path) -> None:
    texts, intents = split_training_examples()
    model_path = tmp_path / "intent_model.pkl"

    stored_model = train_and_store_intent_model(
        texts=texts,
        intents=intents,
        model_path=model_path,
        model_type="logistic_regression",
        feature_extractor="tfidf",
    )

    loaded = load_configured_intent_model(
        model_path=stored_model.model_path,
        model_store="local",
        enable_ml=True,
    )

    assert stored_model.model_path.exists()
    assert loaded is not None
    prediction = loaded.predict("advisor who is working on labor")
    assert prediction.intent == "publication_or_expertise_search"


def test_load_configured_intent_model_returns_none_when_disabled(tmp_path) -> None:
    model_path = tmp_path / "missing.pkl"

    loaded = load_configured_intent_model(
        model_path=model_path,
        model_store="local",
        enable_ml=False,
    )

    assert loaded is None


def test_load_configured_intent_model_returns_none_for_missing_local_model(tmp_path) -> None:
    model_path = tmp_path / "missing.pkl"

    loaded = load_configured_intent_model(
        model_path=model_path,
        model_store="local",
        enable_ml=True,
    )

    assert loaded is None
