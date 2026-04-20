from src.advisors.models import Advisor
from src.search_engines.llm_search import llm_search_advisors


def _advisor(name: str, email: str) -> Advisor:
    return Advisor(
        name=name,
        title="Professor",
        section="Economics",
        email=email,
        profile_url=f"https://example.com/{name.lower().replace(' ', '-')}",
        research_output=["Topic A"],
        activities=["Activity A"],
        press_media=["Media A"],
    )


def test_llm_search_uses_id_mapping(monkeypatch) -> None:
    advisors = [
        _advisor("Alice Smith", "alice@example.com"),
        _advisor("Bob Jones", "bob@example.com"),
    ]

    def fake_openrouter(*args, **kwargs):
        return (
            '{"matches": [{"id": 2, "name": "Alice Smith", "score": 0.82, "reason": "Strong fit"}]}',
            None,
        )

    monkeypatch.setattr("src.search_engines.llm_search.openrouter_chat_completion", fake_openrouter)

    results, error = llm_search_advisors("query", advisors, top_k=5, api_key="key")

    assert error is None
    assert len(results) == 1
    assert results[0].advisor.name == "Bob Jones"
    assert results[0].score == 0.82
    assert results[0].document == "Strong fit"


def test_llm_search_returns_error_for_ambiguous_name_without_id(monkeypatch) -> None:
    advisors = [
        _advisor("Anna Lee", "anna@example.com"),
        _advisor("Anne Lee", "anne@example.com"),
    ]

    def fake_openrouter(*args, **kwargs):
        return ('{"matches": [{"name": "Ann Lee", "score": 0.9, "reason": "close"}]}', None)

    monkeypatch.setattr("src.search_engines.llm_search.openrouter_chat_completion", fake_openrouter)

    results, error = llm_search_advisors("query", advisors, top_k=5, api_key="key")

    assert results == []
    assert error == "LLM returned matches that did not map to advisor names."
