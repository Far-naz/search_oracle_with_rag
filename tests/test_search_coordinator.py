from src.advisors.models import Advisor, MatchAdvisor
from src.intent.types import IntentPrediction
from src.search_engines.coordinator import AdvisorSearchCoordinator


def _advisor(name: str = "Alice Smith") -> Advisor:
    return Advisor(
        name=name,
        title="Professor",
        section="Economics",
        email=f"{name.lower().replace(' ', '.')}@example.com",
        profile_url=f"https://example.com/{name.lower().replace(' ', '-')}",
        research_output=["Topic A"],
        activities=["Activity A"],
        press_media=["Media A"],
    )


class FakeIntentRecognizer:
    def __init__(self, prediction: IntentPrediction) -> None:
        self.prediction = prediction

    def predict(self, text: str) -> IntentPrediction:
        return self.prediction


class FakeSearchEngine:
    def __init__(
        self,
        results: list[MatchAdvisor],
        advisors: list[Advisor] | None = None,
    ) -> None:
        self.results = results
        self.advisors = advisors or [_advisor()]
        self.search_calls = []

    def search(self, query: str, top_k: int = 5, section_filter=None):
        self.search_calls.append(
            {"query": query, "top_k": top_k, "section_filter": section_filter}
        )
        return self.results[:top_k]

    def get_all_advisors(self):
        return {advisor.email: advisor for advisor in self.advisors}


def test_unknown_intent_with_weak_results_escalates_to_llm(monkeypatch) -> None:
    standard_advisor = _advisor("Standard Advisor")
    llm_advisor = _advisor("LLM Advisor")
    engine = FakeSearchEngine(
        results=[MatchAdvisor(standard_advisor, score=0.4, document="weak")],
        advisors=[standard_advisor, llm_advisor],
    )
    recognizer = FakeIntentRecognizer(
        IntentPrediction(intent="unknown", confidence=0.0, source="fallback")
    )
    llm_calls = []

    def fake_llm_search_advisors(query, advisors, top_k, api_key):
        llm_calls.append(
            {"query": query, "advisors": advisors, "top_k": top_k, "api_key": api_key}
        )
        return [MatchAdvisor(llm_advisor, score=0.88, document="LLM fit")], None

    monkeypatch.setattr(
        "src.search_engines.coordinator.llm_search_advisors",
        fake_llm_search_advisors,
    )

    coordinator = AdvisorSearchCoordinator(
        search_engine=engine,
        intent_recognizer=recognizer,
        enable_llm_search=True,
        llm_escalation_min_score=0.65,
    )

    results = coordinator.search("hard to classify query", top_k=3, api_key="key")

    assert results[0].advisor.name == "LLM Advisor"
    assert coordinator.last_llm_attempted is True
    assert coordinator.last_llm_error is None
    assert llm_calls == [
        {
            "query": "hard to classify query",
            "advisors": [standard_advisor, llm_advisor],
            "top_k": 3,
            "api_key": "key",
        }
    ]


def test_unknown_intent_with_strong_results_does_not_escalate(monkeypatch) -> None:
    advisor = _advisor("Strong Advisor")
    engine = FakeSearchEngine(
        results=[MatchAdvisor(advisor, score=0.9, document="strong")]
    )
    recognizer = FakeIntentRecognizer(
        IntentPrediction(intent="unknown", confidence=0.0, source="fallback")
    )

    def fake_llm_search_advisors(*args, **kwargs):
        raise AssertionError("LLM search should not run for strong standard results")

    monkeypatch.setattr(
        "src.search_engines.coordinator.llm_search_advisors",
        fake_llm_search_advisors,
    )

    coordinator = AdvisorSearchCoordinator(
        search_engine=engine,
        intent_recognizer=recognizer,
        enable_llm_search=True,
        llm_escalation_min_score=0.65,
    )

    results = coordinator.search("hard to classify query", top_k=3, api_key="key")

    assert results[0].advisor.name == "Strong Advisor"
    assert coordinator.last_llm_attempted is False
    assert coordinator.last_llm_error is None


def test_known_intent_with_weak_results_does_not_escalate(monkeypatch) -> None:
    advisor = _advisor("Known Intent Advisor")
    engine = FakeSearchEngine(
        results=[MatchAdvisor(advisor, score=0.4, document="weak")]
    )
    recognizer = FakeIntentRecognizer(
        IntentPrediction(intent="advisor_search", confidence=0.8, source="rule")
    )

    def fake_llm_search_advisors(*args, **kwargs):
        raise AssertionError("LLM search should not run for known intents")

    monkeypatch.setattr(
        "src.search_engines.coordinator.llm_search_advisors",
        fake_llm_search_advisors,
    )

    coordinator = AdvisorSearchCoordinator(
        search_engine=engine,
        intent_recognizer=recognizer,
        enable_llm_search=True,
        llm_escalation_min_score=0.65,
    )

    results = coordinator.search("find an advisor", top_k=3, api_key="key")

    assert results[0].advisor.name == "Known Intent Advisor"
    assert coordinator.last_llm_attempted is False
    assert coordinator.last_llm_error is None


def test_publication_or_expertise_search_uses_cleaned_query() -> None:
    advisor = _advisor("Labor Advisor")
    engine = FakeSearchEngine(
        results=[MatchAdvisor(advisor, score=0.8, document="labor")]
    )
    recognizer = FakeIntentRecognizer(
        IntentPrediction(
            intent="publication_or_expertise_search",
            confidence=0.7,
            source="rule",
        )
    )
    coordinator = AdvisorSearchCoordinator(
        search_engine=engine,
        intent_recognizer=recognizer,
        enable_llm_search=True,
        llm_escalation_min_score=0.65,
    )

    results = coordinator.search(
        "I am looking for advisor who is working on labor",
        top_k=3,
        api_key="key",
    )

    assert results[0].advisor.name == "Labor Advisor"
    assert engine.search_calls == [
        {"query": "labor", "top_k": 3, "section_filter": None}
    ]
    assert coordinator.last_llm_attempted is False
