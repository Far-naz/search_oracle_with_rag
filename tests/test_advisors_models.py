from src.advisors.models import (
    Advisor,
    build_advisor_document,
    build_advisor_metadata,
    reconstruct_advisor,
)


def _sample_advisor() -> Advisor:
    return Advisor(
        name="Alice Example",
        title="Associate Professor",
        section="Energy Economics",
        email="alice@example.com",
        profile_url="https://example.com/alice",
        research_output=["Energy Transition", "Grid Markets"],
        activities=["Supervision"],
        press_media=["Interview on Energy"],
    )


def test_build_advisor_document_includes_expected_sections() -> None:
    advisor = _sample_advisor()
    doc = build_advisor_document(advisor)

    assert "Alice Example - Associate Professor, Energy Economics" in doc
    assert "Email: alice@example.com" in doc
    assert "Profile URL: https://example.com/alice" in doc
    assert "Research output:" in doc
    assert "Activities:" in doc
    assert "Press/Media:" in doc


def test_build_metadata_and_reconstruct_roundtrip() -> None:
    advisor = _sample_advisor()
    metadata = build_advisor_metadata(advisor)
    reconstructed = reconstruct_advisor(metadata)

    assert reconstructed == advisor


def test_reconstruct_advisor_defaults_when_keys_missing() -> None:
    reconstructed = reconstruct_advisor({})

    assert reconstructed.name == ""
    assert reconstructed.title == ""
    assert reconstructed.section == ""
    assert reconstructed.email == ""
    assert reconstructed.profile_url == ""
    assert reconstructed.research_output == []
    assert reconstructed.activities == []
    assert reconstructed.press_media == []
