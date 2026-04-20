from src.advisors.models import Advisor
from src.explanations.advisor_explainability import (
    generate_rag_explanation,
    get_highlight_terms,
    get_matched_terms,
    highlight_html,
    render_term_pills,
    split_explanations_by_advisor,
)


def _advisor() -> Advisor:
    return Advisor(
        name="Alice Example",
        title="Energy Systems Professor",
        section="Energy Economics",
        email="alice@example.com",
        profile_url="https://example.com/alice",
        research_output=["Energy transition in Europe"],
        activities=["Thesis supervision"],
        press_media=["Podcast on energy markets"],
    )


def test_get_matched_terms_finds_relevant_fields() -> None:
    advisor = _advisor()
    matched = get_matched_terms("energy markets", advisor)

    assert any(item.startswith("Title:") for item in matched)
    assert any(item.startswith("Research Output:") for item in matched)
    assert any(item.startswith("Press/Media:") for item in matched)


def test_get_highlight_terms_returns_unique_terms() -> None:
    advisor = _advisor()
    highlights = get_highlight_terms("energy energy markets", advisor)

    assert "energy" in highlights
    assert highlights.count("energy") == 1


def test_highlight_html_escapes_and_marks() -> None:
    html = highlight_html("Energy <script>alert(1)</script>", ["energy"])

    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert '<mark class="highlight-term">Energy</mark>' in html


def test_render_term_pills_deduplicates_terms() -> None:
    pills = render_term_pills(["energy", "energy", "markets"])

    assert pills.count("match-pill") == 2
    assert "energy" in pills
    assert "markets" in pills


def test_generate_rag_explanation_returns_none_without_api_key() -> None:
    advisor = _advisor()
    result = generate_rag_explanation("energy", [{"advisor": advisor}], api_key=None)
    assert result is None


def test_generate_rag_explanation_success(monkeypatch) -> None:
    advisor = _advisor()

    def fake_openrouter(*args, **kwargs):
        return "Advisor explanation", None

    monkeypatch.setattr("src.explanations.advisor_explainability.openrouter_chat_completion", fake_openrouter)

    result = generate_rag_explanation("energy", [{"advisor": advisor}], api_key="test-key")
    assert result == "Advisor explanation"


def test_generate_rag_explanation_non_200(monkeypatch) -> None:
    advisor = _advisor()

    def fake_openrouter(*args, **kwargs):
        return None, "error"

    monkeypatch.setattr("src.explanations.advisor_explainability.openrouter_chat_completion", fake_openrouter)

    result = generate_rag_explanation("energy", [{"advisor": advisor}], api_key="test-key")
    assert result is None


def test_split_explanations_by_advisor_maps_sections() -> None:
    advisor_a = _advisor()
    advisor_b = Advisor(
        name="Bob Scholar",
        title="Professor",
        section="Finance",
        email="bob@example.com",
        profile_url="https://example.com/bob",
        research_output=[],
        activities=[],
        press_media=[],
    )

    explanation_text = """
### 1. Alice Example
Alice matches energy systems research.

### 2. Bob Scholar
Bob matches finance and market analytics.
"""

    mapped = split_explanations_by_advisor(explanation_text, [advisor_a, advisor_b])
    assert "Alice matches energy systems research." in mapped["Alice Example"]
    assert "Bob matches finance and market analytics." in mapped["Bob Scholar"]
