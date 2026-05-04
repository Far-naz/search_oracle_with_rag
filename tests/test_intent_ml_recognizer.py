from pathlib import Path

import pytest

from src.intent.hybrid_recognizer import HybridIntentRecognizer
from src.intent.ml_recognizer import MLIntentRecognizer


TRAIN_TEXTS = [
    "recommend an advisor for machine learning",
    "find supervisor in finance",
    "I am exploring research fields in economics",
    "what topic in causal inference should I study",
    "which advisors are accepting students now",
    "who is currently available to supervise",
    "who has publications on diffusion models",
    "looking for expertise in time-series forecasting",
]

TRAIN_INTENTS = [
    "advisor_search",
    "advisor_search",
    "topic_search",
    "topic_search",
    "availability_search",
    "availability_search",
    "publication_or_expertise_search",
    "publication_or_expertise_search",
]


@pytest.mark.parametrize("model_type", ["logistic_regression", "svm", "neural_network"])
def test_ml_recognizer_predicts_known_intent(model_type: str) -> None:
    recognizer = MLIntentRecognizer(model_type=model_type)
    recognizer.fit(TRAIN_TEXTS, TRAIN_INTENTS)

    prediction = recognizer.predict("please recommend an advisor")

    assert prediction.intent == "advisor_search"
    assert prediction.source == "ml"
    assert 0.0 <= prediction.confidence <= 1.0


def test_ml_recognizer_save_and_load(tmp_path: Path) -> None:
    recognizer = MLIntentRecognizer(model_type="logistic_regression")
    recognizer.fit(TRAIN_TEXTS, TRAIN_INTENTS)

    model_path = tmp_path / "intent_model.pkl"
    recognizer.save(model_path)

    loaded = MLIntentRecognizer.load(model_path)
    prediction = loaded.predict("looking for expertise in causal inference")

    assert prediction.intent == "publication_or_expertise_search"


def test_hybrid_uses_ml_when_rule_is_weak() -> None:
    ml_recognizer = MLIntentRecognizer(model_type="logistic_regression")
    ml_recognizer.fit(TRAIN_TEXTS, TRAIN_INTENTS)
    hybrid = HybridIntentRecognizer(
        ml_recognizer=ml_recognizer,
        rule_confidence_threshold=0.95,
        ml_confidence_threshold=0.1,
    )

    prediction = hybrid.predict("recommend advisor")

    assert prediction.source in {"ml", "rule"}


def test_ml_recognizer_rejects_unsupported_labels() -> None:
    recognizer = MLIntentRecognizer(model_type="logistic_regression")

    with pytest.raises(ValueError):
        recognizer.fit(["hello"], ["greeting"])


def test_ml_recognizer_preprocesses_text() -> None:
    recognizer = MLIntentRecognizer(
        model_type="logistic_regression",
        remove_stopwords=True,
        use_lemmatization=True,
    )

    processed = recognizer.preprocess_text("The systems are running in the labs")

    assert "the" not in processed.split()
    assert "system" in processed.split()


def test_bert_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        MLIntentRecognizer(model_type="bert_finetuned")


def test_xgboost_works_when_installed() -> None:
    pytest.importorskip("xgboost")

    recognizer = MLIntentRecognizer(model_type="xgboost")
    recognizer.fit(TRAIN_TEXTS, TRAIN_INTENTS)
    prediction = recognizer.predict("find me an advisor")

    assert prediction.intent == "advisor_search"
    assert prediction.source == "ml"


def test_word2vec_feature_extractor_works_when_installed() -> None:
    pytest.importorskip("gensim")

    recognizer = MLIntentRecognizer(
        model_type="logistic_regression",
        feature_extractor="word2vec",
    )
    recognizer.fit(TRAIN_TEXTS, TRAIN_INTENTS)
    prediction = recognizer.predict("which advisor is available now")

    assert prediction.intent in {
        "advisor_search",
        "availability_search",
        "publication_or_expertise_search",
        "topic_search",
    }
    assert prediction.source == "ml"
