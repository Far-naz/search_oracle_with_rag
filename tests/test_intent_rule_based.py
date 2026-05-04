from src.intent.hybrid_recognizer import HybridIntentRecognizer
from src.intent.rule_based_recognizer import RuleBasedIntentRecognizer


def test_rule_based_detects_greeting() -> None:
    recognizer = RuleBasedIntentRecognizer()

    prediction = recognizer.predict("Can you recommend a supervisor for my thesis?")

    assert prediction is not None
    assert prediction.intent == "advisor_search"
    assert prediction.source == "rule"


def test_rule_based_detects_advisor_search() -> None:
    recognizer = RuleBasedIntentRecognizer()

    prediction = recognizer.predict("I am exploring research fields in econometrics")

    assert prediction is not None
    assert prediction.intent == "topic_search"
    assert prediction.confidence > 0.0


def test_rule_based_detects_advisor_working_on_as_expertise_search() -> None:
    recognizer = RuleBasedIntentRecognizer()

    prediction = recognizer.predict("I am looking for advisor who is working on labor")

    assert prediction is not None
    assert prediction.intent == "publication_or_expertise_search"
    assert prediction.confidence >= 0.6


def test_hybrid_returns_fallback_when_no_match_and_no_ml() -> None:
    recognizer = HybridIntentRecognizer()

    prediction = recognizer.predict("zxqv mnrp")

    assert prediction.intent == "unknown"
    assert prediction.source == "fallback"
