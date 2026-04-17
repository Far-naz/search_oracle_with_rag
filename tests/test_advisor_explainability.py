from src.advisors.models import Advisor
from src.explanations.advisor_explainability import (
    generate_rag_explanation,
    get_highlight_terms,
    get_matched_terms,
    highlight_html,
    render_term_pills,
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

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json() -> dict:
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "Advisor explanation"}],
                        }
                    }
                ]
            }

    def fake_post(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr("src.explanations.advisor_explainability.requests.post", fake_post)

    result = generate_rag_explanation("energy", [{"advisor": advisor}], api_key="test-key")
    assert result == "Advisor explanation"


def test_generate_rag_explanation_non_200(monkeypatch) -> None:
    advisor = _advisor()

    class FakeResponse:
        status_code = 500

        @staticmethod
        def json() -> dict:
            return {}

    def fake_post(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr("src.explanations.advisor_explainability.requests.post", fake_post)

    result = generate_rag_explanation("energy", [{"advisor": advisor}], api_key="test-key")
    assert result is None
