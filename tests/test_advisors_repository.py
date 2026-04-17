import json

from src.advisors.repository import load_available_advisors


def test_load_available_advisors_reads_records(tmp_path) -> None:
    data = [
        {
            "name": "Alice Example",
            "title": "Associate Professor",
            "section": "Energy Economics",
            "email": "alice@example.com",
            "profile_url": "https://example.com/alice",
            "research_output": ["Energy Transition"],
            "activities": ["Supervision"],
            "press_media": ["Interview"],
        },
        {
            "name": "Bob Example",
            "title": "Professor",
            "section": "Finance",
        },
    ]
    path = tmp_path / "advisors.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    advisors = load_available_advisors(path)

    assert len(advisors) == 2
    assert advisors[0].name == "Alice Example"
    assert advisors[0].research_output == ["Energy Transition"]
    assert advisors[1].name == "Bob Example"
    assert advisors[1].email == ""
    assert advisors[1].profile_url == ""
    assert advisors[1].research_output == []
    assert advisors[1].activities == []
    assert advisors[1].press_media == []
