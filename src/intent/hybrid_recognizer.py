from src.intent.ml_recognizer import MLIntentRecognizer
from src.intent.rule_based_recognizer import RuleBasedIntentRecognizer
from src.intent.types import Intent, IntentPrediction


class HybridIntentRecognizer:
    """
    Hybrid recognizer that first tries deterministic rules, then falls back to ML.
    """

    def __init__(
        self,
        rule_recognizer: RuleBasedIntentRecognizer | None = None,
        ml_recognizer: MLIntentRecognizer | None = None,
        rule_confidence_threshold: float = 0.6,
        ml_confidence_threshold: float = 0.45,
        fallback_intent: Intent | str = Intent.UNKNOWN,
    ) -> None:
        self.rule_recognizer = rule_recognizer or RuleBasedIntentRecognizer()
        self.ml_recognizer = ml_recognizer
        self.rule_confidence_threshold = rule_confidence_threshold
        self.ml_confidence_threshold = ml_confidence_threshold
        self.fallback_intent = fallback_intent

    def predict(self, text: str) -> IntentPrediction:
        rule_prediction = self.rule_recognizer.predict(text)
        if rule_prediction and rule_prediction.confidence >= self.rule_confidence_threshold:
            return rule_prediction

        if self.ml_recognizer is not None:
            try:
                ml_prediction = self.ml_recognizer.predict(text)
                if ml_prediction.confidence >= self.ml_confidence_threshold:
                    return ml_prediction
            except RuntimeError:
                pass

        if rule_prediction is not None:
            return rule_prediction

        return IntentPrediction(
            intent=self.fallback_intent,
            confidence=0.0,
            source="fallback",
        )
